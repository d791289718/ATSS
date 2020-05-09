#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from math import pi

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class RotatedBoxList(object):
    """
    This class represents a set of rotated bounding boxes.
    The bounding boxes are represented as a Nx5 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, rbbox, image_size, mode="xywha"):
        device = rbbox.device if isinstance(rbbox, torch.Tensor) else torch.device("cpu")
        rbbox = torch.as_tensor(rbbox, dtype=torch.float32, device=device)

        # 确保rbbox的dim=2，最后一维size=5，mode为正确的mode
        # ! 允许空的构建
        if rbbox.ndimension() != 2:
            raise ValueError(
                "Rotatedbbox should have 2 dimensions, got {}".format(rbbox.ndimension())
            )
        if rbbox.size(-1) != 5 and rbbox.size(-1) != 8:
            raise ValueError(
                "last dimension of Rotatedbbox should have a "
                "size of 5 or 8, got {}".format(rbbox.size(-1))
            )
        if mode not in ("xywha", "poly"):
            raise ValueError("mode should be 'xywha' or 'poly'")

        self.rbbox = rbbox  # [[x1, y1, w1, h1, ang1], [x, y, w, h, ang]...] or [[x1, y1, ...x4, y4]]
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def _get_transform_matrix(self, mode):
        if mode == "r2h":
            ang = self.rbbox[:, -1]
        elif mode == "h2r":
            ang = self.rbbox[:, -1] * -1
        else:
            raise ValueError(
                "transform matrix mode has to be r2h or h2r, got {}".format(mode))
        transform_matrix_1 = torch.stack((torch.cos(ang), -torch.sin(ang)), dim=1)
        transform_matrix_2 = torch.stack((torch.sin(ang), torch.cos(ang)), dim=1)
        transform_matrix = torch.stack((transform_matrix_1, transform_matrix_2), dim=1)
        return transform_matrix

    def get_bbox_xyxy(self):
        if len(self.rbbox) == 0:
            return self.rbbox
        if self.mode == "xywha":
            tmp_box = self.convert("poly").rbbox
        else:
            tmp_box = self.rbbox
        x_min, _ = torch.min(tmp_box[:, 0::2], dim=1)
        y_min, _ = torch.min(tmp_box[:, 1::2], dim=1)
        x_max, _ = torch.max(tmp_box[:, 0::2], dim=1)
        y_max, _ = torch.max(tmp_box[:, 1::2], dim=1)
        return torch.stack((x_min, y_min, x_max, y_max), dim=1)

    def get_bbox_xywh(self):
        if len(self.rbbox) == 0:
            return self.rbbox
        if self.mode == "xywha":
            tmp_box = self.convert("poly").rbbox
        else:
            tmp_box = self.rbbox
        x_min, _ = torch.min(tmp_box[:, 0::2], dim=1)
        y_min, _ = torch.min(tmp_box[:, 1::2], dim=1)

        return torch.stack((x_min, y_min, self.rbbox[:, 2], self.rbbox[:, 3]), dim=1)

    def convert(self, mode):
        if mode not in ("poly", "xywha"):
            raise ValueError("mode should be 'poly' or 'xywha'")
        if mode == self.mode:
            return self

        if mode == "poly":
            # rbbox为空
            if len(self.rbbox) == 0:
                return self
            # 旋转矩阵
            transform_matrix = self._get_transform_matrix("r2h")

            poly_list = []
            for rbox, matrix in zip(self.rbbox, transform_matrix):
                dx = rbox[2] / 2.
                dy = rbox[3] / 2.
                pre_loc = torch.tensor(
                    [[-1*dx, -1*dy], [dx, -1*dy], [dx, dy], [-1*dx, dy]])
                pre_loc = torch.as_tensor(pre_loc, device=rbox.device)
                points = torch.bmm(matrix[None].expand(4, -1, -1), pre_loc[:, :, None])
                points[:, 0, :] += rbox[0]
                points[:, 1, :] += rbox[1]
                loc = points.reshape(1, -1)
                poly_list.append(loc)

            rbbox = RotatedBoxList(torch.cat(poly_list), self.size, mode='poly')
            rbbox._copy_extra_fields(self)
            return rbbox
        elif mode == "xywha":
            raise NotImplementedError
            # if len(self.rbbox) == 0:
            #     return self

            # transform_matrix = self._get_transform_matrix("h2r")

            # box_list = []
            # for poly, matrix in zip(self.rbbox, transform_matrix):
            #     x_s = poly[0::2]
            #     y_s = poly[1::2]
            #     x_min = torch.min(x_s)
            #     y_min = torch.min(y_s)
            #     x_max = torch.max(x_s)
            #     y_max = torch.max(y_s)

            #     xc = (x_min + x_max) / 2.
            #     yc = (y_min + y_max) / 2.

            #     pre_loc = torch.stack((x_s - xc, y_s - yc), dim=1)
            #     pre_loc = torch.as_tensor(pre_loc, device=poly.device)
            #     points = torch.bmm(matrix[None].expand(4, -1, -1), pre_loc[:, :, None])
            #     points[:, 0, :] += xc
            #     points[:, 1, :] += yc
            #     loc = points.reshape(1, -1)
            #     box_list.append(loc)

            # rbbox = RotatedBoxList(torch.cat(box_list), self.size, mode='xywha')
            # rbbox._copy_extra_fields(self)
            # return rbbox
        else:
            raise NotImplementedError

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        # ! assert ratios[0] == ratios[1], "rotatedbbox现在仅支持长宽等比例resize"
        # 长宽比例不变
        ratio = ratios[0]
        # scale the rotated bbox
        scaled_box = self.rbbox[:, :-1] * ratio
        scaled_box = torch.cat((scaled_box, self.rbbox[:, -1, None]), 1)

        rbbox = RotatedBoxList(scaled_box, size, mode=self.mode)
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            rbbox.add_field(k, v)
        return rbbox

        # TODO: 处理长宽ratio不一致的形况
        # ratio_width, ratio_height = ratios
        # xmin, ymin, xmax, ymax = self._split_into_xyxy()
        # scaled_xmin = xmin * ratio_width
        # scaled_xmax = xmax * ratio_width
        # scaled_ymin = ymin * ratio_height
        # scaled_ymax = ymax * ratio_height
        # scaled_box = torch.cat(
        #     (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        # )
        # bbox = BoxList(scaled_box, size, mode="xyxy")
        # # bbox._copy_extra_fields(self)
        # for k, v in self.extra_fields.items():
        #     if not isinstance(v, torch.Tensor):
        #         v = v.resize(size, *args, **kwargs)
        #     bbox.add_field(k, v)

        # return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        x, y, w, h, ang = self.rbbox.split(1, dim=-1)
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_x = image_width - x - TO_REMOVE
            transposed_y = y
            transposed_w = w
            transposed_h = h
            transposed_ang = ang * -1
        elif method == FLIP_TOP_BOTTOM:
            TO_REMOVE = 1
            transposed_x = x
            transposed_y = image_height - y - TO_REMOVE
            transposed_w = w
            transposed_h = h
            transposed_ang = ang * -1

        # assert transposed_x > 0 and transposed_y > 0
        transposed_boxes = torch.cat(
            (transposed_x, transposed_y, transposed_w, transposed_h, transposed_ang), dim=-1
        )

        rbbox = RotatedBoxList(transposed_boxes, self.size, mode="xywha")

        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            rbbox.add_field(k, v)
        return rbbox

    def remove_outside_image(self):
        if len(self.rbbox) == 0:
            return self
        if self.mode == "xywha":
            tmp_box = self.convert("poly").rbbox
        else:
            tmp_box = self.rbbox

        x_s = tmp_box[:, 0::2]
        y_s = tmp_box[:, 1::2]
        keep = (x_s >= 0) & (y_s >= 0) & (x_s <= self.size[0]) & (y_s <= self.size[1])
        keep = keep[:, 0] & keep[:, 1] & keep[:, 2] & keep[:, 3]
        return self[keep]

    def rotate(self, ang):
        # ang is 0~360 in degrees
        device = self.rbbox.device
        assert self.mode == "xywha"
        x_s = self.rbbox[:, 0] - self.size[0] / 2
        y_s = self.rbbox[:, 1] - self.size[1] / 2

        rad = torch.as_tensor(-ang * pi / 180, dtype=torch.float32, device=device)
        transform_matrix_1 = torch.stack((torch.cos(rad), -torch.sin(rad)), dim=0)
        transform_matrix_2 = torch.stack((torch.sin(rad), torch.cos(rad)), dim=0)
        transform_matrix = torch.stack((transform_matrix_1, transform_matrix_2), dim=0)

        pre_loc = torch.stack((x_s, y_s), dim=1)
        pre_loc = torch.as_tensor(pre_loc, device=self.rbbox.device)
        points = torch.bmm(transform_matrix[None].expand(len(pre_loc), -1, -1), pre_loc[:, :, None])
        new_x = points[:, 0, :].reshape(-1) + self.size[0] / 2
        new_y = points[:, 1, :].reshape(-1) + self.size[1] / 2

        ang = torch.as_tensor(ang, dtype=torch.float32, device=device)
        new_ang = ang + -1 * self.rbbox[:, -1] * 180 / pi
        new_ang[new_ang > 0] = new_ang[new_ang > 0] % 180
        new_ang[new_ang > 90] = new_ang[new_ang > 90] - 180
        new_ang = -1 * new_ang * pi / 180

        rbox_list = torch.stack((new_x, new_y, self.rbbox[:, 2], self.rbbox[:, 3], new_ang), dim=1)
        rbbox = RotatedBoxList(rbox_list, self.size, mode="xywha")

        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.rotate()
            rbbox.add_field(k, v)
        return rbbox.convert(self.mode)

    def area(self):
        rbox = self.rbbox
        if self.mode == "xywha":
            area = rbox[:, 2] * rbox[:, 3]
        else:
            raise RuntimeError("mode should be 'xywha'")

        return area

    # Tensor-like methods
    def to(self, device):
        rbbox = RotatedBoxList(self.rbbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            rbbox.add_field(k, v)
        return rbbox

    # 用选定的bbox构建一个新的实例
    def __getitem__(self, item):
        rbbox = RotatedBoxList(self.rbbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            rbbox.add_field(k, v[item])
        return rbbox

    def __len__(self):
        return self.rbbox.shape[0]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    from math import pi
    bbox = RotatedBoxList(
        [[341.2143, 443.3325, 778.4297, 178.2595, -1.122944], [10, 10, 10, 10, pi/4], [10, 10, 5, 5, pi/2]],
        (1172, 816)
    )

    rm_bbox = bbox.remove_outside_image()

    # print(bbox.get_bbox_xyxy())
    # print("==========")
    # print(bbox.get_bbox_xywh())

    p_bbox = bbox.convert('poly')
    print(p_bbox.rbbox)

    ro_bbox = bbox.rotate(20)
    print(ro_bbox.rbbox)

    s_bbox = bbox.resize((50, 50))
    print(s_bbox)
    print(s_bbox.rbbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.rbbox)

    # def copy_with_fields(self, fields, skip_missing=False):
    #     bbox = BoxList(self.bbox, self.size, self.mode)
    #     if not isinstance(fields, (list, tuple)):
    #         fields = [fields]
    #     for field in fields:
    #         if self.has_field(field):
    #             bbox.add_field(field, self.get_field(field))
    #         elif not skip_missing:
    #             raise KeyError("Field '{}' not found in {}".format(field, self))
    #     return bbox

    # def _split_into_xyxy(self):
    #     if self.mode == "xyxy":
    #         xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
    #         return xmin, ymin, xmax, ymax
    #     elif self.mode == "xywh":
    #         TO_REMOVE = 1
    #         xmin, ymin, w, h = self.bbox.split(1, dim=-1)
    #         return (
    #             xmin,
    #             ymin,
    #             xmin + (w - TO_REMOVE).clamp(min=0),
    #             ymin + (h - TO_REMOVE).clamp(min=0),
    #         )
    #     else:
    #         raise RuntimeError("Should not be here")

    # def crop(self, box):
    #     """
    #     Cropss a rectangular region from this bounding box. The box is a
    #     4-tuple defining the left, upper, right, and lower pixel
    #     coordinate.
    #     """
    #     # xmin, ymin, xmax, ymax = self._split_into_xyxy()
    #     w, h = box[2] - box[0], box[3] - box[1]
    #     cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
    #     cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
    #     cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
    #     cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

    #     # TODO should I filter empty boxes here?
    #     if False:
    #         is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

    #     cropped_box = torch.cat(
    #         (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
    #     )
    #     bbox = BoxList(cropped_box, (w, h), mode="xyxy")
    #     # bbox._copy_extra_fields(self)
    #     for k, v in self.extra_fields.items():
    #         if not isinstance(v, torch.Tensor):
    #             v = v.crop(box)
    #         bbox.add_field(k, v)
    #     return bbox.convert(self.mode)
