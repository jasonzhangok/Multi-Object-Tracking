import torch
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

def main():
    video_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MOTdataset/MOT16/MOT16-09.mp4'
    # txt文件路径
    labels_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/yolov5-master/runs/detect/exp27/labels'
    saves_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/yolov5-master/runs/detect/exp27/detection'
    if (not os.path.exists(saves_path)):
        os.makedirs(saves_path)
    if not os.access(video_path, os.F_OK):
        print('测试文件不存在')
        return

    cap = cv2.VideoCapture(video_path)  # 获取视频对象
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    isOpened = cap.isOpened  # 判断是否打开
    frame_count = 1
    frame_batch_num = 1
    stride = 10  # 隔5帧保存一张图片
    if(isOpened):
        for _ in tqdm(range(int(frame_num))):
            (frameState, frame) = cap.read()  # 记录每帧及获取状态
            if frameState == True and (frame_count % stride == 0):
                pedestrian_count = 1
                w = frame.shape[1]
                h = frame.shape[0]
                label_path = labels_path + "/" + "MOT16-09_" + str(frame_batch_num) + ".txt"
                f = open(label_path, 'r+', encoding='utf-8')
                for line in f.readlines():
                    msg = line.split(" ")
                    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                    # print(x1, ",", y1, ",", x2, ",", y2)
                    res = frame[y1:y2, x1:x2]
                    # res = cv2.resize(res, (64, 128), interpolation=cv2.INTER_CUBIC)
                    save_path = saves_path + "/" + "MOT16-09_" + str(frame_batch_num) + '_' + str(pedestrian_count) + ".jpg"
                    pedestrian_count = pedestrian_count + 1
                    cv2.imwrite(save_path, res)

                frame_batch_num = frame_batch_num + 1
            frame_count += 1
    cap.release()
    return
if __name__ == '__main__':
    main()
