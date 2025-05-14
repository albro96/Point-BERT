# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32
# @Email:  cshzxie@gmail.com

import logging
import open3d
import torch
from utils.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import os

# from extensions.emd import emd_module as emd
from pytorch3d.loss import chamfer_distance
import sys
from easydict import EasyDict

sys.path.append("/storage/share/repos/code/01_scripts/modules/")

import ml_tools.metrics as ml_metrics
from ml_tools.metrics import ToothMetrics, chamfer_distance_manual


class Metrics(object):
    __version__ = "0.3"
    NO_OCCLUSION_VALUE = 100
    OCCLUSIONFUNCS = {
                    'OcclusionLoss': "cd_occlusion_loss",
                    'PenetrationLoss': "penetration_loss",
                    'ClusterDistLoss': "cluster_dist_loss",
                    'ClusterPosLoss': "cluster_position_loss",
                    'ClusterNumLoss': "cluster_num_loss",
                    'InvIOULoss': "inv_iou_loss",
    }

    ITEMS = [
        {
            "name": "F-Score",
            "enabled": True,
            "eval_func": "cls._get_f_score",
            "is_greater_better": True,
            "init_value": 0,
        },
        {
            "name": "CDL1",
            "enabled": True,
            "eval_func": "cls._get_chamfer_distancel1",
            "eval_object": None,
            "is_greater_better": False,
            "init_value": 32767,
        },
        {
            "name": "CDL2",
            "enabled": True,
            "eval_func": "cls._get_chamfer_distancel2",
            "eval_object": None,
            "is_greater_better": False,
            "init_value": 32767,
        },
        {
            "name": "InfoCDL2",
            "enabled": True,
            "eval_func": "cls._get_info_chamfer_distancel2",
            "eval_object": None,
            "is_greater_better": False,
            "init_value": 32767,
        },
    ]
    ITEMS.extend(
        [
            {
                "name": key,
                "enabled": True,
                "eval_func": val,
                "is_greater_better": False,
                "init_value": 32767,
            }
            for key, val in OCCLUSIONFUNCS.items()
        ]
    )

    @classmethod
    def get(
        cls, pred, gt, partial=None, antagonist=None, metrics=None, requires_grad=False
    ):
        _items = (
            [item for item in cls.items() if item["name"] in metrics]
            if metrics is not None
            else cls.items()
        )

        _values = EasyDict()
        # print(
        #     f"Initialzing ToothMetrics with pred shape: {pred.shape}, gt shape: {gt.shape}, antagonist shape: {antagonist.shape}"
        # )
        if any([item['name'] in cls.OCCLUSIONFUNCS.keys()for item in _items]):
            toothmetrics = ToothMetrics(
                recon=pred, gt=gt, antagonist=antagonist, requires_grad=requires_grad, clear_intermediate_results=True, no_occlusion_value=cls.NO_OCCLUSION_VALUE
            )

        for i, item in enumerate(_items):
            if item['name'] in cls.OCCLUSIONFUNCS.keys():
                assert antagonist is not None
                _values[item["name"]] = toothmetrics.get(item["eval_func"])
            else:
                eval_func = eval(item["eval_func"])
                if "f_score" in item["eval_func"]:
                    full_pred = torch.concatenate([pred, partial], dim=1)
                    full_gt = torch.concatenate([gt, partial], dim=1)
                    _values[item["name"]] = eval_func(full_pred, full_gt)
                else:
                    _values[item["name"]] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i["enabled"]]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i["name"] for i in _items]


    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        device = pred.device
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(
                    cls._get_f_score(pred[idx : idx + 1], gt[idx : idx + 1])
                )
            return sum(f_score_list) / len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            result = (
                2 * recall * precision / (recall + precision)
                if recall + precision
                else 0.0
            )
            result_tensor = torch.tensor(result).to(device)
            return result_tensor

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        # return chamfer_distance(pred, gt, norm=1)[0]
        return chamfer_distance_manual(pred, gt, norm=1, two_sided_reduction='mean', point_reduction='mean', single_directional=False, batch_reduction=None)


    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        # return chamfer_distance(pred, gt, norm=2)[0]
        return chamfer_distance_manual(pred, gt, norm=2, two_sided_reduction='mean', point_reduction='mean', single_directional=False, batch_reduction=None)

    @classmethod
    def _get_info_chamfer_distancel2(cls, pred, gt, norm=2, single_directional=False, point_reduction="mean", square=False, tau=0.5, config=None, two_sided_reduction='mean'):

        if config is not None:
            tau = config.get("tau", tau)
            point_reduction = config.get("point_reduction", point_reduction)
            two_sided_reduction = config.get("two_sided_reduction", two_sided_reduction)
            square = config.get("square", square)
            
        assert point_reduction in ["mean", "sum"]
        assert two_sided_reduction in [None, "mean"]

        red_fun = torch.mean if point_reduction == "mean" else torch.sum

        # chamfer_dist = chamfer_3DDist()
        # dist1, dist2, idx1, idx2 = chamfer_dist(pred, gt)
        dists = chamfer_distance_manual(pred, gt, norm=norm, two_sided_reduction=None, point_reduction=None, single_directional=single_directional, batch_reduction=None)

        if single_directional:
            d1 = torch.clamp(dists, min=1e-9)
            distances1 = - torch.log(torch.exp(-tau * d1)/(torch.sum(torch.exp(-tau * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
            return red_fun(distances1) 
        else:
            dist1, dist2 = dists

            dist1 = torch.clamp(dist1, min=1e-9)
            dist2 = torch.clamp(dist2, min=1e-9)

            d1 = torch.sqrt(dist1) if not square else dist1
            d2 = torch.sqrt(dist2) if not square else dist2

            distances1 = - torch.log(torch.exp(-tau * d1)/(torch.sum(torch.exp(-tau * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)
            distances2 = - torch.log(torch.exp(-tau * d2)/(torch.sum(torch.exp(-tau * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

        dist_red = red_fun(distances1) + red_fun(distances2)

        if two_sided_reduction == 'mean':
            dist_red /= 2

        return dist_red

    @classmethod
    def _get_emd_distance(cls, pred, gt, eps=0.005, iterations=100):
        emd_loss = cls.ITEMS[3]["eval_object"]
        dist, _ = emd_loss(pred, gt, eps, iterations)
        emd_out = torch.mean(torch.sqrt(dist))
        return emd_out * 1

    def __init__(self, metric_name, values, metrics=None):
        # set ITEMS to only include metrics
        if metrics is not None:
            self._items = [i for i in Metrics.items() if i["name"] in metrics]
            self._items = sorted(self._items, key=lambda x: metrics.index(x["name"]))

        assert metric_name in Metrics.names(), f"Invalid metric name: {metric_name}"

        # self._items = Metrics.items() # this is original

        self._values = [item["init_value"] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == "list":
            self._values = values
        elif type(values).__name__ == "dict":
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item["name"]
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn("Ignore Metric[Name=%s] due to disability." % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception("Unsupported value type: %s" % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]["name"]
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item["name"] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception("Invalid metric name to compare.")

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return (
            _value > other_value
            if _metric["is_greater_better"]
            else _value < other_value
        )
