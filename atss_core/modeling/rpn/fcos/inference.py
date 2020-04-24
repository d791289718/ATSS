import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from atss_core.modeling.box_coder import BoxCoder
from atss_core.modeling.utils import cat
from atss_core.structures.bounding_box import BoxList
from atss_core.structures.boxlist_ops import cat_boxlist
from atss_core.structures.boxlist_ops import boxlist_ml_nms
from atss_core.structures.boxlist_ops import remove_small_boxes
from atss_core.structures.rboxlist_ops import remove_small_rotated_boxes
from atss_core.structures.rboxlist_ops import convert_to_rbox
from atss_core.structures.rboxlist_ops import cat_rboxlist
from atss_core.structures.rboxlist_ops import rboxlist_ml_nms

class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            loactions: Tesnor(num, 2)
            box_cls: tensor of size N, C, H, W
            box_regression: tensor of size N, 4, H, W
        Return:
            list[BoxList]
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()  # N, W*H, 1
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)  # N, W*H, 1
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()  # N, W*H

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)  # True看作1处理
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]  # score

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]  # box location
            per_class = per_candidate_nonzeros[:, 1] + 1  # class name

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def rotated_forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, ang_regression,
            centerness, image_sizes):
        """
        Arguments:
            locations tensor (num, 2)
            box_cls: tensor of size N, C, H, W
            box_regression: tensor of size N, 4, H, W
            ang_regression: tensor of size N, 1, H, W
            centerness: tensor of size N, 1, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()  # N, W*H, C
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)  # N, W*H, 4
        ang_regression = ang_regression.view(N, 1, H, W).permute(0, 2, 3, 1)
        ang_regression = ang_regression.reshape(N, -1)  # N, W*H
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()  # N, W*H

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)  # 每个img > thresh的个数(in the current feature map)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]  # W*H, C
            per_candidate_inds = candidate_inds[i]  # ! if == 0
            per_box_cls = per_box_cls[per_candidate_inds]  # score: (num, )

            per_candidate_nonzeros = per_candidate_inds.nonzero()  # retuen tuple
            per_box_loc = per_candidate_nonzeros[:, 0]  # feature_map上点的索引
            per_class = per_candidate_nonzeros[:, 1] + 1  # class索引

            # 所有符合要求的点的box_reg, size = (符合的数目, 4)
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            # 所有符合要求的点的ang_reg
            per_ang_regression = ang_regression[i]
            per_ang_regression = per_ang_regression[per_box_loc]
            # 所有符合要求的点的locations
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():  # > 1000个
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)  # per_pre_nms_top_n最大为1000
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_ang_regression = per_ang_regression[top_k_indices]

            # l, t, r, b size=(num, ) dim=1
            l = per_box_regression[:, 0]
            t = per_box_regression[:, 1]
            r = per_box_regression[:, 2]
            b = per_box_regression[:, 3]
            xs = per_locations[:, 0]
            ys = per_locations[:, 1]
            ang = per_ang_regression

            rboxlist = convert_to_rbox(l, t, r, b, ang, xs, ys, image_sizes[i])
            rboxlist.add_field("labels", per_class)
            rboxlist.add_field("scores", torch.sqrt(per_box_cls))
            # rboxlist = rboxlist.clip_to_image(remove_empty=True)
            rboxlist = rboxlist.remove_outside_image()
            rboxlist = remove_small_rotated_boxes(rboxlist, self.min_size)
            results.append(rboxlist)

        return results

    def forward(self, locations, box_cls, box_regression, ang_regression, centerness, image_sizes, is_rotated):
        """
        Arguments:
            # over feature map
            locations: list[Tensor]
            box_cls: list[tensor]
            box_regression: list[tensor]

            # over batch
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        # over features
        for _, (l, o, b, a, c) in enumerate(zip(locations, box_cls, box_regression, ang_regression, centerness)):
            if not is_rotated:
                sampled_boxes.append(
                    self.forward_for_single_feature_map(l, o, b, c, image_sizes)
                )  # return 每张图在当前feature的RBoxList[[feture1的[图1], [图2]],[],[]]
            else:
                sampled_boxes.append(
                    self.rotated_forward_for_single_feature_map(l, o, b, a, c, image_sizes)
                )

        boxlists = list(zip(*sampled_boxes))  # list的元素是每个图的
        boxlists = [cat_rboxlist(boxlist) for boxlist in boxlists]  # 每个图的结果cat成一个RBoxList

        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists, is_rotated)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists, is_rotated):
        num_images = len(boxlists)  # N
        results = []
        for i in range(num_images):
            # multiclass nms
            if not is_rotated:
                result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            else:
                result = rboxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:  # 100
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,  # 0.05
        pre_nms_top_n=pre_nms_top_n,  # 1000
        nms_thresh=nms_thresh,  # 0.6
        fpn_post_nms_top_n=fpn_post_nms_top_n,  # 100
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,  # 2
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
