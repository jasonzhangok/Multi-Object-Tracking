import os
import cv2
from utils.dataloaders import LoadImages,LoadImagesAndLabels

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

import pathlib

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

def main():
	#image path
    img_path = ROOT / 'data' / 'images'
    img_path = str(ROOT / 'data' / 'images')
    #txt path
    label_path = ROOT / 'runs' / 'detect'/'exp13'/'labels'
    label_path = str(label_path)
    # 读取图片，结果为三维数组
    imgs = LoadImages(img_path)
    labels = os.listdir(label_path)
    for i in len(labels):
        img = cv2.imread(img_path)
    # 图片宽度(像素)
    w = img.shape[1]
    # 图片高度(像素)
    h = img.shape[0]
    # 打开文件，编码格式'utf-8','r+'读写
    f = open(label_path, 'r+', encoding='utf-8')
    # 读取txt文件中的第一行,数据类型str
    line = f.readline()
    # 根据空格切割字符串，最后得到的是一个list
    msg = line.split(" ")
    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
    print(x1, ",", y1, ",", x2, ",", y2)
    #裁剪
    img_roi = img[y1:y2,x1:x2]
    save_path='./cutpictures/hg.jpg'
    cv2.imwrite(save_path,img_roi)

if __name__ == '__main__':
    main()

