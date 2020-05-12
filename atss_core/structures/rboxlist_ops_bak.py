# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import cv2
import math
import numpy as np
from math import pi
import DOTA_devkit.polyiou as polyiou

from atss_core.structures.rotated_bbox import RotatedBoxList

from atss_core.layers import nms as _box_nms
from atss_core.layers import ml_nms as _box_ml_nms
# from atss_core.layers.poly_nms import poly_nms_cuda
# ! check to support 空的rbolist，为什么boxlist支持空的，还是他不可能出现空的？

def convert_to_ltrb_V3(xs, ys, xc, yc, w, h, ang):
    """
    from (xs, ys) and the rbox get the ltrb&ang
    xs, ys:size=(num_point, 1)
    others:size=(1, num_box)
    """
    num_point = xs.size(0)
    num_box = xc.size(1)
    dx = xs - xc
    dy = ys - yc
    dis = (dx.pow(2) + dy.pow(2)).sqrt()

    theta = ang + torch.atan2(dy, dx)

    # ang > 0 --> -1; ang < 0 --> 1
    flip = torch.zeros(1, num_box)
    flip[ang > 0] = -1
    flip[ang <= 0] = 1

    dw = dis * torch.cos(theta) * flip
    dh = dis * torch.sin(theta) * flip

    l = h/2. - dh
    t = w/2. - dw
    r = h/2. + dh
    b = w/2. + dw
    return l, t, r, b


def convert_to_ltrb(xs, ys, xc, yc, w, h, ang):
    """
    from (xs, ys) and the rbox get the ltrb&ang
    xs, ys:size=(num_point, 1)
    others:size=(1, num_box)
    """
    num_point = xs.size(0)
    num_box = xc.size(1)
    dx = xs - xc
    dy = ys - yc

    flip = xs.new_ones(1, num_box)
    flip[ang > 0] = -1

    pre_loc = torch.stack((dx, dy), dim=2)

    # 旋转矩阵
    ang = ang[0] * -1
    transform_matrix_1 = torch.stack((torch.cos(ang), -1*torch.sin(ang)), dim=1)
    transform_matrix_2 = torch.stack((torch.sin(ang), torch.cos(ang)), dim=1)
    transform_matrix = torch.stack((transform_matrix_1, transform_matrix_2), dim=1)

    transform_matrix = transform_matrix[None, :].expand(num_point, -1, -1, -1)
    transform_matrix = transform_matrix.reshape(-1, 2, 2)
    pre_loc = pre_loc[:, :, :, None].reshape(-1, 2, 1)
    locations = torch.bmm(transform_matrix, pre_loc)
    locations = locations.reshape(num_point, num_box, 2)

    rotated_loc_x = locations[:, :, 0] * flip
    rotated_loc_y = locations[:, :, 1] * flip

    l = h/2. - rotated_loc_y
    t = w/2. - rotated_loc_x
    r = rotated_loc_y + h/2.
    b = rotated_loc_x + w/2.
    return l, t, r, b


def convert_to_rbox(l, t, r, b, ang, xs, ys, image_sizes):
    """
    l,t,r,b,ang,xs,xy dim = 1
    return RotatedBoxlist
    """
    img_h, img_w = image_sizes # imagelist的size是(h, w)
    if len(ang) == 0:
        return RotatedBoxList(torch.tensor([], device=ang.device),
        (int(img_w), int(img_h)), mode="xywha")
    h = l + r
    w = t + b

    flip = ang.new_ones(ang.size())
    flip[ang > 0] = -1

    pre_xc = (t - b) / 2. * flip
    pre_yc = (l - r) / 2. * flip
    pre_loc = torch.stack((pre_xc, pre_yc), dim=1)

    transform_matrix_1 = torch.stack((torch.cos(-ang), -torch.sin(-ang)), dim=1)
    transform_matrix_2 = torch.stack((torch.sin(-ang), torch.cos(-ang)), dim=1)
    transform_matrix = torch.stack((transform_matrix_1, transform_matrix_2), dim=1)

    locations = torch.bmm(transform_matrix, pre_loc[:, :, None])
    locations = torch.squeeze(locations, -1)

    dx = locations[:, 0]
    dy = locations[:, 1]
    xc = xs + dx
    yc = ys - dy  # 因为和图像中的纵坐标走向不一样

    rboxes = torch.stack((xc, yc, w, h, ang), dim=1)
    rboxlist = RotatedBoxList(rboxes, (int(img_w), int(img_h)), mode="xywha")
    return rboxlist


def remove_small_rotated_boxes(rboxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (RotatedBoxlist)
        min_size (int)
    """
    if len(rboxlist.rbbox) == 0:
        return rboxlist
    _, _, ws, hs, _ = rboxlist.rbbox.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return rboxlist[keep]


def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_rboxlist(bboxes):
    """
    Concatenates a list of RotatedBoxList of the same image but in diffrent feature map level
    (having the same image size) into a single RotatedBoxList

    Arguments:
        bboxes (list[RotatedBoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, RotatedBoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = RotatedBoxList(_cat([bbox.rbbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


# TODO：仅实现了一类的nms
def rboxlist_ml_nms(rboxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(RotatedBoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return rboxlist
    if len(rboxlist.rbbox) == 0:
        return rboxlist
    mode = rboxlist.mode
    scores = rboxlist.get_field(score_field)
    labels = rboxlist.get_field(label_field)

    # convert to numpy
    dets = rboxlist.convert("poly").rbbox
    if isinstance(dets, torch.Tensor):
        dets = dets.cpu().numpy().astype(np.float64)
    if isinstance(nms_thresh, torch.Tensor):
        nms_thresh = nms_thresh.cpu().numpy().astype(np.float64)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy().astype(np.float64)

    # 准备工作
    x1 = np.min(dets[:, 0::2], axis=1)
    y1 = np.min(dets[:, 1::2], axis=1)
    x2 = np.max(dets[:, 0::2], axis=1)
    y2 = np.max(dets[:, 1::2], axis=1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]  # scores从大到小的索引

    # 正式开始
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # score最大的索引id
        keep.append(i)  # 加入该id
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 除了最大score的poly，剩下的poly
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)  # numpy
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                assert False
        except:
            pass
        inds = np.where(hbb_ovr <= nms_thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)

    # result_dets = torch.from_numpy(dets[keep, :]).to(device)
    # keep_id = torch.from_numpy(np.array(keep)).to(device)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    rboxlist = rboxlist[keep]
    return rboxlist.convert(mode)


if __name__ == "__main__":
    # # test convert_to_ltrb
    # xc = torch.randn(3)
    # yc = torch.randn(3)
    # w = torch.ones(3)
    # h = torch.ones(3)
    # ang = torch.tensor([pi/4, -pi/4, pi/4])
    # xs = torch.randn(6)
    # ys = torch.randn(6)

    # l,t,r,b = convert_to_ltrb(xs[:,None], ys[:,None],xc[None], yc[None], w[None], h[None], ang[None])
    # l3,t3,r3,b3 = convert_to_ltrb_V3(xs[:,None], ys[:,None],xc[None], yc[None], w[None], h[None], ang[None])

    # test convert_to_rbox
    
    l = torch.rand(3)
    t = torch.rand(3)
    r = torch.rand(3)
    b = torch.rand(3)
    ang = torch.tensor([pi/4, pi/4, pi/4])
    xs = torch.randn(3)
    ys = torch.randn(3)


# def rboxlist_ml_nms(rboxlist, nms_thresh, max_proposals=-1,
#                    score_field="scores", label_field="labels"):
#     """
#     Performs non-maximum suppression on a boxlist, with scores specified
#     in a boxlist field via score_field.

#     Arguments:
#         boxlist(BoxList)
#         nms_thresh (float)
#         max_proposals (int): if > 0, then only the top max_proposals are kept
#             after non-maximum suppression
#         score_field (str)
#     """
#     if nms_thresh <= 0:
#         return rboxlist
#     mode = rboxlist.mode

#     boxes = rboxlist.get_bbox_xyxy()
#     scores = rboxlist.get_field(score_field)
#     labels = rboxlist.get_field(label_field)
#     keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)
#     if max_proposals > 0:
#         keep = keep[: max_proposals]
#     rboxlist = rboxlist[keep]
#     return rboxlist


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
