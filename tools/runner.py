import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import math
import cv2
import numpy as np
import wandb
from easydict import EasyDict

def compute_loss(loss_1, loss_2, config, niter):
    '''
    compute the final loss for optimization
    For dVAE: loss_1 : reconstruction loss, loss_2 : kld loss
    '''
    start = config.kldweight.start
    target = config.kldweight.target
    ntime = config.kldweight.ntime

    _niter = niter - 10000
    if _niter > ntime:
        kld_weight = target
    elif _niter < 0:
        kld_weight = 0.
    else:
        kld_weight = target + (start - target) *  (1. + math.cos(math.pi * float(_niter) / ntime)) / 2.

    loss = loss_1 + kld_weight * loss_2

    return loss

def get_temp(config, niter):
    if config.get('temp') is not None:
        start = config.temp.start
        target = config.temp.target
        ntime = config.temp.ntime
        if niter > ntime:
            return target
        else:
            temp = target + (start - target) *  (1. + math.cos(math.pi * float(niter) / ntime)) / 2.
            return temp
    else:
        return 0 

def run_net(args, config):
    logger = get_logger(args.log_name)
    # build dataset
    
    train_sampler, train_dataloader = builder.dataset_builder(
        args, config.dataset, mode="train", bs=config.bs
    )
    _, val_dataloader = builder.dataset_builder(args, config.dataset, mode="val", bs=1)
    _, test_dataloader = builder.dataset_builder(
        args, config.dataset, mode="test", bs=1
    )
 
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    if args.log_data:
        wandb.watch(base_model)

    # trainval
    # training
    

    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        # metrics = validate(base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2, args, config, logger = None)
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(config.loss_metrics)

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (data, tooth) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)

            points = data.to(args.device)

            temp = get_temp(config, n_itr)

            ret = base_model(points, temperature = temp, hard = False)

            loss_1, loss_2 = base_model.module.get_loss(ret, points)

            _loss = compute_loss(loss_1, loss_2, config, n_itr)

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_1 = dist_utils.reduce_tensor(loss_1, args)
                loss_2 = dist_utils.reduce_tensor(loss_2, args)
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])
            else:
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if not args.sweep and idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
                
        if config.scheduler.type != 'function':
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2, args, config, logger = None)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                if args.save_checkpoints and args.log_data:
                    builder.save_checkpoint(
                        base_model,
                        optimizer,
                        epoch,
                        metrics,
                        best_metrics,
                        f"ckpt-best-{wandb.run.name}",
                        args,
                    )


        if args.save_checkpoints and not args.save_only_best and args.log_data:
            builder.save_checkpoint(
                base_model, optimizer, epoch, metrics, best_metrics, f"ckpt-last-{wandb.run.name}", args
            )
            # save every 100 epoch
            if epoch % 100 == 0:
                builder.save_checkpoint(
                    base_model,
                    optimizer,
                    epoch,
                    metrics,
                    best_metrics,
                    f"ckpt-epoch-{epoch:03d}-{wandb.run.name}",
                    args,
                )

            if (config.max_epoch - epoch) < 2:
                builder.save_checkpoint(
                    base_model,
                    optimizer,
                    epoch,
                    metrics,
                    best_metrics,
                    f"ckpt-epoch-{epoch:03d}-{wandb.run.name}",
                    args,
                )

        if args.log_data:
            log_dict = EasyDict()
            log_dict.epoch = epoch

            for idx, loss in enumerate(config.loss_metrics):
                log_dict[f"train/{loss}"] = losses.avg()[idx]

            wandb.log(log_dict, step=epoch)



def validate(base_model, val_dataloader, epoch, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    # test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    # test_metrics = AverageMeter(config.val_metrics)
    val_metrics = {metric: [] for metric in config.val_metrics}
    # category_metrics = dict()
    n_samples = len(val_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (data, tooth) in enumerate(val_dataloader):
            tooth = tooth[0].item()
            points = data.cuda()

            ret = base_model(inp = points, hard=True, eval=True)
            coarse_points = ret[0]
            dense_points = ret[1]

            # sparse_loss_l1 =  ChamferDisL1(coarse_points, points)
            # sparse_loss_l2 =  ChamferDisL2(coarse_points, points)
            # dense_loss_l1 =  ChamferDisL1(dense_points, points)
            # dense_loss_l2 =  ChamferDisL2(dense_points, points)

            # if args.distributed:
            #     sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
            #     sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
            #     dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
            #     dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            # test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, points, metrics=config.val_metrics)
            for metric, value in _metrics.items():
                _metrics[metric] = value.item()
            for metric in val_metrics:
                val_metrics[metric].append(_metrics[metric])

            if (val_dataloader.dataset.patient == "0538") and args.log_data and str(tooth)[-1] in ['1', '6']:
                print('Logging PCDs')

                wandb.log(
                    {
                        f"val/pcd/dense/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": dense_points[0].detach().cpu().numpy(),
                            }
                        ),
                        f"val/pcd/coarse/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": coarse_points[0].detach().cpu().numpy(),
                            }
                        ),
                        f"val/pcd/gt/{val_dataloader.dataset.tooth}": wandb.Object3D(
                            {
                                "type": "lidar/beta",
                                "points": points[0].detach().cpu().numpy(),
                            }
                        ),
                    },
                    step=epoch,
                )


            # if tooth not in category_metrics:
            #     category_metrics[tooth] = AverageMeter(config.val_metrics)
            # category_metrics[tooth].update(_metrics)

       
            # if (idx+1) % 2000 == 0:
            #     print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
            #                 (idx + 1, n_samples, tooth,  ['%.4f' % l for l in test_losses.val()], 
            #                 ['%.4f' % m for m in _metrics]), logger=logger)
                
        # for _,v in category_metrics.items():
        #     test_metrics.update(v.avg())
        # print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    log_dict = {}  
    print("============================ VAL RESULTS ============================")
    print(f"Epoch: {epoch}")

    mean_val_metrics = []
    for metric, values in val_metrics.items():
        values = torch.tensor(values)
        mean = torch.mean(values)
        mean_val_metrics.append(mean)
        median = torch.median(values)
        log_dict[f"val/{metric}"] = mean
        log_dict[f"val/{metric}-median"] = median
        print(f"{metric}: {mean:.6f}/{median:.6f}")

    if args.log_data:
        wandb.log(log_dict, step=epoch)
 
    return Metrics(
        metric_name=config.consider_metric,
        values=mean_val_metrics,
        metrics=config.val_metrics,
    )

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156",
        "02818832",
        "04379243",
        "04099429",
        "03948459",
        "03790512",
        "03642806",
        "03467517",
        "03261776",
        "03001627",
        "02958343",
        "03759954"
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if  taxonomy_ids[0] not in useful_cate:
                continue
    
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')


            ret = base_model(inp = points, hard=True, eval=True)
            dense_points = ret[1]

            final_image = []

            data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points)
            final_image.append(points)

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points)
            final_image.append(dense_points)

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)

            if idx > 1000:
                break

        return 
