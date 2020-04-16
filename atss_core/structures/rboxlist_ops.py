# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import cv2
import math
from math import pi

from atss_core.structures.rotated_bbox import RotatedBoxList

from atss_core.layers import nms as _box_nms
from atss_core.layers import ml_nms as _box_ml_nms
# from atss_core.layers.poly_nms import poly_nms_cuda

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
    ang = ang[0]
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
    t = w/2. - rotated_loc_x # budui
    r = rotated_loc_y + h/2.
    b = rotated_loc_x + w/2.
    return l, t, r, b

def convert_to_rbox(l, t, r, b, ang, xs, ys, image_sizes):
    """
    return RotatedBoxlist
    """
    h = l + r
    w = t + b

    flip = torch.ones(ang.size())
    flip[ang > 0] = -1

    pre_xc = (t - b)/2. * flip
    pre_yc = (l - r)/2. * flip
    pre_loc = torch.stack((pre_xc, pre_yc), dim=1)

    ang = -1 * ang
    # 旋转矩阵
    ang.squeeze_()
    transform_matrix_1 = torch.stack((torch.cos(ang), -1*torch.sin(ang)), dim=1)
    transform_matrix_2 = torch.stack((torch.sin(ang), torch.cos(ang)), dim=1)
    transform_matrix = torch.stack((transform_matrix_1, transform_matrix_2), dim=1)

    locations = torch.bmm(transform_matrix, pre_loc[:, :, None])
    locations = torch.squeeze(locations)

    dx = locations[:, 0]
    dy = locations[:, 1]
    xc = xs + dx
    yc = ys - dy  # 因为和图像中的纵坐标走向不一样

    img_h, img_w = image_sizes
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
    _, _, ws, hs, _ = rboxlist.rbbox.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return rboxlist[keep]

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

def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

# TODO: 现在是hbb的iou，改成rbb的iou
def rboxlist_ml_nms(rboxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
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
        return rboxlist
    mode = rboxlist.mode

    boxes = rboxlist.get_bbox()
    scores = rboxlist.get_field(score_field)
    labels = rboxlist.get_field(label_field)
    keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    rboxlist = rboxlist[keep]
    return rboxlist

# TODO: poly版本的nms但是没测试，from DOTA
def poly_nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    # import pdb
    # print('in nms wrapper')
    # pdb.set_trace()
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = poly_nms_cuda.poly_nms(dets_th, iou_thr)
        else:
            raise NotImplementedError

    if is_numpy:
        raise NotImplementedError
    return dets[inds, :], inds

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

# FIXME: 写了个寂寞
def rotated_boxlist_iou(boxlist, rotated_boxlist):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (RotatedBoxList) bounding boxes, sized [M,5].

    Returns:
      (tensor) iou, sized [N,M].
    """
    area1 = boxlist[2] * boxlist[3]
    area2 = rotated_boxlist[2] * rotated_boxlist[3]
    cx = (boxlist[0] + boxlist[2]) / 2.0
    cy = (boxlist[1] + boxlist[3]) / 2.0
    r1 = ((cx, cy), (boxlist[2] - boxlist[0] + 1, boxlist[3] - boxlist[1] + 1), 0)
    r2 = ((rotated_boxlist[0], rotated_boxlist[1]),
             (rotated_boxlist[2], rotated_boxlist[3]), math.degrees(rotated_boxlist[4]))
    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
    if int_pts is not None:
        order_pts = cv2.convexHull(int_pts, returnPoints=True)
        int_area = cv2.contourArea(order_pts)
        # 计算出iou
        ious = int_area * 1.0 / (area1 + area2 - int_area)
    else:
        ious = 0







# def convert_to_ltrb_V2(xs, ys, xc, yc, w, h, ang):
#     """
#     from (xs, ys) and the rbox get the ltrb&ang
#     xs, ys:size=(num_point, 1)
#     others:size=(1, num_box)
#     """
#     num_point = xs.size(0)
#     num_box = xc.size(1)
#     dx = xs - xc
#     dy = ys - yc

#     # print(dx)
#     # print(dy)
#     vec1 = torch.stack((dx, dy), dim = 2)
#     # print(vec1.size())
#     vec2 = torch.stack((torch.ones(ang.size()), -1 * torch.tan(ang)), dim=2)
#     vec2 = vec2.expand(num_point, -1, -1)
#     # print(vec2[:,:,1])
   
#     norm1 = torch.norm(vec1, dim=2, keepdim=True)
#     norm2 = torch.norm(vec2, dim=2, keepdim=True)

#     cos_theta = (vec1 * vec2).sum(2) / torch.reshape((norm1 * norm2),(num_point, num_box))

#     flip2 = torch.zeros(num_point, num_box)
#     flip2[dy > 0] = 1
#     flip2[dy <= 0] = -1
#     theta = torch.acos(cos_theta) * flip2

#     dis = (dx.pow(2) + dy.pow(2)).sqrt()
    
#     # ang > 0 --> -1; ang < 0 --> 1
#     flip = torch.zeros(1, num_box)
#     flip[ang > 0] = -1
#     flip[ang <= 0] = 1

#     dw = dis * torch.sin(theta) * flip
#     dh = dis * torch.cos(theta) * flip

#     l = h/2. - dh
#     t = w/2. - dw
#     r = h/2. + dh
#     b = w/2. + dw
#     return l, t, r, b