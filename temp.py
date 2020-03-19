# -*- coding=utf-8 -*-
#!/usr/bin/python

import sys
import os
import os.path as osp
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET

# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {'ship': 1}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # if isinstance(obj, time):
        #     return obj.__str__()
        # else:
        #     return super(NpEncoder, self).default(obj)


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# 得到图片唯一标识号
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, img_list, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    list_fp = xml_list
    # 标注基本结构
    json_dict = {"info":[], "licenses":[],
                 "images":[],
                 "annotations": [],
                 "categories": []
                }
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for xml_f in list_fp:
        # print("Processing {}".format(xml_f))
        # 解析XML
        tree = ET.parse(xml_f)
        root = tree.getroot()

        # get the content of images
        filename = get_and_check(root, 'Img_FileName', 1)
        image_id = get_and_check(root, 'Img_ID', 1)  # 图片ID
        width = int(get_and_check(root, 'Img_SizeWidth', 1).text)
        height = int(get_and_check(root, 'Img_SizeHeight', 1).text)
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)

        # get the content of annotations
        for obj in get(root.find('HRSC_Objects'), 'HRSC_Object'):
            annotation = {}

            # category = get_and_check(obj, 'name', 1).text
            category = 'ship'
            # 更新类别ID字典
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]

            xmin = int(get_and_check(obj, 'box_xmin', 1).text)
            ymin = int(get_and_check(obj, 'box_ymin', 1).text)
            xmax = int(get_and_check(obj, 'box_xmax', 1).text)
            ymax = int(get_and_check(obj, 'box_ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)

            difficult = get_and_check(obj, 'difficult', 1).text

            rbox_cx = float(get_and_check(obj, 'mbox_cx', 1).text)
            rbox_cy = float(get_and_check(obj, 'mbox_cy', 1).text)
            rbox_w = float(get_and_check(obj, 'mbox_w', 1).text)
            rbox_h = float(get_and_check(obj, 'mbox_h', 1).text)
            rbox_ang = float(get_and_check(obj, 'mbox_ang', 1).text)

            annotation['difficult'] = difficult
            annotation['area'] = None
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            annotation['rbox'] = [rbox_cx, rbox_cy, rbox_w, rbox_h, rbox_ang]
            # 设置分割数据，点的顺序为逆时针方向
            # annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]
            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # the content of categories
    for cate, cid in categories.items():
        cat = {'supercategory': 'ship', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict,cls = MyEncoder)
    json_fp.write(json_str)
    json_fp.close()

def cvt_annotations(devkit_path, split, out_file):
    split_txt = osp.join(devkit_path,'ImageSets/{}.txt'.format(split))
    if not osp.isfile(split_txt):
        print('filelist does not exist: {}, skip {}'.format(split_txt, split))
        return

    # 获取要转换的bmp和标注list
    xml_paths = []
    img_paths = []
    with open(split_txt) as f:
        for line in f:
            img_name = line[:-1]
            xml_file = osp.join(devkit_path,'Annotations/{}.xml'.format(img_name))
            img_file = osp.join(devkit_path,'JPEGImages/{}.bmp'.format(img_name))
            xml_paths.append(xml_file)
            img_paths.append(img_file)

    convert(xml_paths, img_paths, out_file)

if __name__ == '__main__':
    root_dir = '/data/datasets/HRSC2016'
    out_dir = '/data/datasets/HRSC2016'
    assert(osp.isdir(out_dir))
    assert(osp.isdir(root_dir))


    for split in ['train', 'val', 'test']:
        dataset_name = split
        print('processing {} ...'.format(dataset_name))
        json_file = './instances_' + dataset_name + '.json'
        cvt_annotations(root_dir, split, osp.join(out_dir, json_file))

    print('Done!')
