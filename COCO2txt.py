# -*- coding:utf-8 -*-

from coco.pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

#the path you want to save
savePath="/data/datasets/HRSC2016/"
imgDir=savePath+'images/'
annDir=savePath+'annotations/'

# where you save the data and .json
clsPicked = ['ship']
annFile_list = ['instances_train.json', 'instances_val.json', 'instances_test.json']
dataDir = '/data/datasets/HRSC2016'
txtDir = dataDir + '/annotations/txtLabel'


class MyCOCO(COCO):
    def __init__(self, annFile):
        super().__init__(annFile)

    def toTxt(self, listFile):

        cls_id = self.getCatIds(catNms=clsPicked)
        img_ids = self.getImgIds(catIds=cls_id)
        print(clsPicked,cls_id,len(img_ids))

        with open(listFile, 'w') as f:
            for imgId in tqdm(img_ids):
                img = self.loadImgs(imgId)[0]
                filename = img['file_name']
                # print(filename)
                f.write(filename + '\n')

        for imgId in tqdm(img_ids):
            img = self.loadImgs(imgId)[0]

            annIds = self.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
            anns = self.loadAnns(annIds)

            filename = img['file_name']
            txtFile = os.path.join(txtDir, "%s.txt"%os.path.splitext(filename)[0])

            with open(txtFile, 'w') as f:
                for ann in anns:
                    f.write(str(cls_id[0]) + " " + " ".join([str(a) for a in ann['bbox']]) + '\n')


def main():
    for annFile in annFile_list:
        annFile = os.path.join(annDir, annFile)
        print('processing %s'%(annFile))

        #COCO API for initializing annotated data
        coco = MyCOCO(annFile)

        print('类别数目：',len(coco.dataset['categories']))
        print('图片数目：',len(coco.dataset['images']))
        print('标注数目：',len(coco.dataset['annotations']))


        listFile = os.path.join(txtDir, '%s.txt'%os.path.splitext(annFile)[0])
        print(listFile)
        coco.toTxt(listFile)

main()

