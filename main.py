import time
import os
import torch
import sys
import torch.multiprocessing as mp
import wandb
import shutil
import os.path as op
import json
from pprint import pprint
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
sys.path.append("/storage/share/repos/code/01_scripts/modules/")

from os_tools.import_dir_path import import_dir_path, convert_path
from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *

def main(rank=0, world_size=1):
    # args
    pada = import_dir_path()
    config = EasyDict({
    "optimizer": {
        "type": "AdamW"
        "kwargs": {
            "lr": 0.0005, #0.0005,
            "weight_decay": 0.0005, # 0.0005
        }
    },
    "scheduler": {
        "type": "CosLR",
        "kwargs": {
            "epochs": 500,
            "initial_epochs": 10,
            "warming_up_init_lr": 0.00005
        }
    },
    "temp": {
        "start": 1,
        "target": 0.0625,
        "ntime": 100000
    },
    "kldweight": {
        "start": 0,
        "target": 0.1,
        "ntime": 100000
    },
    "model": {
        "NAME": "DiscreteVAE",
        "group_size": 32,
        "num_group": 64,
        "encoder_dims": 256,
        "num_tokens": 8192,
        "tokens_dims": 256,
        "decoder_dims": 256
        },
    "total_bs": 256,
    "step_per_update": 1,
    "max_epoch": 500,
    'model_name': 'DiscreteVAE',
    'loss_metrics': ['Loss1', 'Loss2'],
    "consider_metric": "CDL2",
    'val_metrics': [
        "CDL1",
        "CDL2"   ]
    })

    config['dataset'] = EasyDict(
        {
            "num_points": 2048,
            "tooth_range": {
                "teeth": 'full', 
                "jaw": "full",
                "quadrants": "all",
            },
            "data_type": "npy",
            "use_fixed_split": True,
            "enable_cache": True,
            "create_cache_file": True,
            "overwrite_cache_file": False,
            "return_normals": False, 
            "dataset": "orthodental",
            "data_dir": None,
            "datahash": "15c02eb0",
            'normalize_mean': True,
            'normalize_pose': False,
            'normalize_scale': False,
        }
    )

    args = EasyDict(
        {
            "launcher": "pytorch" if world_size > 1 else "none",
            "num_gpus": world_size,
            "local_rank": rank,
            "num_workers": 0,  # only applies to mode='train', set to 0 for val and test
            "seed": 0,
            "deterministic": False,
            "sync_bn": False,
            "experiment_dir": pada.model_base_dir,
            "start_ckpts": None,
            "val_freq": 10,
            "resume": False,
            "mode": None,
            "save_checkpoints": True,
            "save_only_best": False,
            "ckpt_dir": None,
            "cfg_dir": None,
            'log_testdata': True,
            'ckpt_path': convert_path(r"O:\data\models\PoinTr\sweep\PoinTr-InfoCD-CD-downsample1\ckpt\ckpt-best-zany-sweep-2.pth"),
            "gt_partial_saved": False,
            'no_occlusion_val': 100,
            "test": False,
            "log_data": True,  # if true: wandb logger on and save ckpts to local drive
            'log_name': 'DiscreteVAE',
        }
    )


    if args.local_rank is not None:
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(args.local_rank)


    args.use_gpu = torch.cuda.is_available()
    args.use_amp_autocast = False
    args.device = torch.device("cuda" if args.use_gpu else "cpu")


    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False

    
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    if args.distributed:
        assert config.total_bs % world_size == 0
        config.bs = config.total_bs // world_size
    else:
        config.bs = config.total_bs

    wandb_config = None

    if args.log_data:
                    # set the wandb project where this run will be logged, dont set config here, else sweep will fail   
        wandb.init(
         project="AutoEncoder",
        save_code=True,
        )

        # define custom x axis metric
        wandb.define_metric("epoch")

        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

        wandb.define_metric("val/pcd/dense/*", step_metric="epoch")
        wandb.define_metric("val/pcd/coarse/*", step_metric="epoch")
        wandb.define_metric("val/pcd/gt/*", step_metric="epoch")

        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        wandb_config = wandb.config

        # update the model config with wandb config
        for key, value in wandb_config.items():
            if "." in key:
                keys = key.split(".")
                config_temp = config
                for sub_key in keys[:-1]:
                    config_temp = config_temp.setdefault(sub_key, {})
                config_temp[keys[-1]] = value
            else:
                config[key] = value

    # config.model.update(network_config_dict[config.model_name].model)


    if wandb_config is not None:
        args.sweep = True if "sweep" in wandb_config else False
    else:
        args.sweep = False

    args.log_data = args.sweep or args.log_data

    args.experiment_path = os.path.join(args.experiment_dir, config.model_name)
    # os.makedirs(args.experiment_path, exist_ok=True)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    # log 
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation

    if args.sweep:
        args.experiment_path = os.path.join(
            args.experiment_path, "sweep", config.sweepname
        )

    if args.log_data and not args.test:
        if not os.path.exists(args.experiment_path):
            os.makedirs(args.experiment_path, exist_ok=True)
            print("Create experiment path successfully at %s" % args.experiment_path)

        shutil.copy(__file__, args.experiment_path)

        args.cfg_dir = op.join(args.experiment_path, "config")
        args.ckpt_dir = op.join(args.experiment_path, "ckpt")
        os.makedirs(args.cfg_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        cfg_name = f"config-{wandb.run.name}.json"

        with open(os.path.join(args.cfg_dir, cfg_name), "w") as json_file:
            json_file.write(json.dumps(config, indent=4))

    if args.log_data:
        # update wandb config
        wandb.config.update(config, allow_val_change=True)

    pprint(config)
    torch.autograd.set_detect_anomaly(True)

    # # run
    # if args.test:
    #     test_net(args, config)
    # else:
    run_net(args, config)


if __name__ == "__main__":
    # User Input
    num_gpus = 1  # number of gpus, dont use 3
    print("Number of GPUs: ", num_gpus)

    if num_gpus > 1:
        if num_gpus == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        elif num_gpus == 3:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
        elif num_gpus == 4:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"  # Set any free port
        os.environ["WORLD_SIZE"] = str(num_gpus)
        # mp.spawn(main, args=(num_gpus, ), nprocs=num_gpus, join=True)
        mp.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        main(rank=0, world_size=1)
