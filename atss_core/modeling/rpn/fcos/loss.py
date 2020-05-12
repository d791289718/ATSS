"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from atss_core.layers import SigmoidFocalLoss
from atss_core.layers import BoxL1Loss
from atss_core.layers import AngL1Loss
from atss_core.layers import IOULoss
from atss_core.modeling.matcher import Matcher
from atss_core.modeling.utils import cat
from atss_core.structures.boxlist_ops import boxlist_iou
from atss_core.structures.boxlist_ops import cat_boxlist
from atss_core.structures.rboxlist_ops import convert_to_ltrb
from ..utils import concat_box_prediction_layers


INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.center_sampling_mode = cfg.MODEL.FCOS.CENTER_SAMPLING_MODE
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        # self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.box_reg_loss_func = BoxL1Loss()
        self.angle_reg_loss_func = AngL1Loss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")       

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            ) # larger
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            ) # larger
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            ) # smaller
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            ) # smaller
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def get_rotated_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0, mode='constant'):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        # convert gt_xs gt_ys to rotated axis [num_points, num_targets]
        ang = gt[..., 4]
        ro_xs = torch.cos(ang)[None] * gt_xs[:, None] - torch.sin(ang)[None] * gt_ys[:, None]
        ro_ys = torch.sin(ang)[None] * gt_xs[:, None] + torch.cos(ang)[None] * gt_ys[:, None]

        gt = gt[None].expand(K, num_gts, 5)
        # all the Xs Ys are in rotated axis
        center_w = gt[..., 2] / 2.
        center_h = gt[..., 3] / 2.
        center_gt = gt.new_zeros(K, num_gts, 4)
        # no gt
        if center_w[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        if mode == 'constant':
            beg = 0
            for level, n_p in enumerate(num_points_per):
                end = beg + n_p
                stride = strides[level] * radius
                stride = gt.new_ones(n_p, num_gts) * stride
                # limit sample region in gt
                center_gt[beg:end, :, 0] = torch.where(
                    -stride > -center_w[beg:end, :], -stride, -center_w[beg:end, :]
                )  # larger
                center_gt[beg:end, :, 1] = torch.where(
                    -stride > -center_h[beg:end, :], -stride, -center_h[beg:end, :]
                )
                center_gt[beg:end, :, 2] = torch.where(
                    stride < center_w[beg:end, :], stride, center_w[beg:end, :]
                )
                center_gt[beg:end, :, 3] = torch.where(
                    stride < center_h[beg:end, :], stride, center_h[beg:end, :]
                )
                beg = end
        elif mode == 'ratio':
            # limit sample region in gt
            center_gt[:, :, 0] = -center_w[:, :] * radius
            center_gt[:, :, 1] = -center_h[:, :] * radius
            center_gt[:, :, 2] = center_w[:, :] * radius
            center_gt[:, :, 3] = center_h[:, :] * radius
        left = ro_xs - center_gt[..., 0]
        right = center_gt[..., 2] - ro_xs
        top = ro_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - ro_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets, rtargets, is_rotated=True):
        # 准备工作
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []

        # over 不同feature level
        for l, points_per_level in enumerate(points):
            # .前面的的type；.后面的data
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])  # 拷贝了一份[-1， 64]
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)  # 2维
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)  # list[Tensor] -> Tensor
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level

        points_all_level = torch.cat(points, dim=0)  # list[Tensor] -> Tensor, all feature map in one tensor

        if not is_rotated:
            labels, reg_targets = self.compute_targets_for_locations(
                points_all_level, targets, expanded_object_sizes_of_interest
            )
        else:
            labels, reg_targets, ang_targets = self.compute_rotated_targets_for_locations(
                points_all_level, rtargets, expanded_object_sizes_of_interest
            )

        # over images
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)  # Tensor -> list[Tensor]
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)  # Tensor -> list[Tensor]

        # change to level first version
        reg_targets_level_first = []
        labels_level_first = []

        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        # angle相关
        if is_rotated:
            for i in range(len(labels)):
                ang_targets[i] = torch.split(ang_targets[i], num_points_per_level, dim=0)  # Tensor -> list[Tensor]

            ang_targets_level_first = []
            for level in range(len(points)):
                ang_targets_per_level = torch.cat([
                    ang_targets_per_im[level]
                    for ang_targets_per_im in ang_targets
                ], dim=0)
                ang_targets_level_first.append(ang_targets_per_level)

            return labels_level_first, reg_targets_level_first, ang_targets_level_first

        return labels_level_first, reg_targets_level_first, None

    def compute_rotated_targets_for_locations(self, locations, rtargets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        ang_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        # over N(=16) images
        for im_i in range(len(rtargets)):
            targets_per_im = rtargets[im_i]
            assert targets_per_im.mode == "xywha"
            bboxes = targets_per_im.rbbox

            if len(bboxes) == 0:
                labels.append(torch.zeros(len(locations), dtype=torch.long, device=bboxes.device))
                reg_targets.append(torch.zeros(len(locations), 4, dtype=torch.float32, device=bboxes.device))
                ang_targets.append(torch.zeros(len(locations), dtype=torch.float32, device=bboxes.device))
                continue

            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l, t, r, b = convert_to_ltrb(
                xs[:, None], ys[:, None], bboxes[:, 0][None], bboxes[:, 1][None],
                bboxes[:, 2][None], bboxes[:, 3][None], bboxes[:, 4][None]
            )

            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            ang_targets_per_im = bboxes[:, 4][None].expand(len(locations), -1)  # 预测的是与x夹角绝对值

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_rotated_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius,
                    mode=self.center_sampling_mode
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            # * 这个操作很妙
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            ang_targets_per_im = ang_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            ang_targets.append(ang_targets_per_im)

        return labels, reg_targets, ang_targets

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)


    def __call__(self, locations, box_cls, box_regression, ang_regression,
    centerness, targets, rtargets, is_rotated):
        """
        Arguments:
            # list的每个元素是feature map
            locations (list[Tensor])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            
            # list的每个元素是images
            targets (list[BoxList])
            rtargets(list[RotatedBoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
            ang_loss(Tensor)
        """
        N = box_cls[0].size(0)  # batch的样本数目
        num_classes = box_cls[0].size(1)
        # if len(rtargets.rbbox) == 0 & is_rotated:
        #     # num_gpus = get_num_gpus()
        #     # # sync num_pos from all gpus
        #     # total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        #     # num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        #     box_cls_flatten = []
        #     for l in range(len(box_cls)):
        #         box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
        #     box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        #     labels_flatten = box_cls_flatten.new_zeros(box_cls_flatten.size())
        #     # focal loss
        #     # cls_loss = self.cls_loss_func(
        #     #     box_cls_flatten,
        #     #     labels_flatten.int()
        #     # ) / num_pos_avg_per_gpu

        #     reg_loss = cls_loss.new_zeros(1)
        #     ang_loss = cls_loss.new_zeros(1)
        #     centerness_loss = cls_loss.new_zeros(1)
        #     return cls_loss, reg_loss, ang_loss, centerness_loss

        labels, reg_targets, ang_targets = self.prepare_targets(
            locations, targets, rtargets, is_rotated)

        box_cls_flatten = []
        box_regression_flatten = []
        ang_regression_flatten = []
        centerness_flatten = []

        labels_flatten = []
        reg_targets_flatten = []
        ang_targets_flatten = []

        # head预测结果整理成合适的格式
        # over feature map
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))            
            centerness_flatten.append(centerness[l].permute(0, 2, 3, 1).reshape(-1))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))

            if is_rotated:
                ang_regression_flatten.append(ang_regression[l].permute(0, 2, 3, 1).reshape(-1))
                ang_targets_flatten.append(ang_targets[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)        
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        if is_rotated:
            ang_regression_flatten = torch.cat(ang_regression_flatten, dim=0)
            ang_targets_flatten = torch.cat(ang_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        # 仅仅对非背景(!=0)的计算centerness，reg
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        if is_rotated:
            ang_regression_flatten = ang_regression_flatten[pos_inds]
            ang_targets_flatten = ang_targets_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # focal loss
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            # reg_loss = self.box_reg_loss_func(
            #     box_regression_flatten,
            #     reg_targets_flatten,
            #     centerness_targets,
            #     is_rotated
            # ) / sum_centerness_targets_avg_per_gpu

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu

            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu

            if is_rotated:
                ang_loss = self.angle_reg_loss_func(
                    ang_regression_flatten,
                    ang_targets_flatten,
                    centerness_targets
                ) / num_pos_avg_per_gpu
            else:
                ang_loss = None

        else:
            # pos_ind = None,这些东西也就是空的
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()
            if is_rotated:
                ang_loss = ang_regression_flatten.sum()
            else:
                ang_loss = None
        ang_loss = ang_loss * 10
        return cls_loss, reg_loss, ang_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
