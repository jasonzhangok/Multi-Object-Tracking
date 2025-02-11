import os
import cv2

def main():
    # imgs_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/test_medium/0001'
    # #txt文件路径
    # labels_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/yolov5-master/runs/detect/exp13/0001/labels'
    # saves_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/train_modified'

    imgs_path_root = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/train'
    #txt文件路径
    labels_path_root = '/Users/jason/IdeaProjects/PeopleFlowDetection/yolov5-master/runs/detect/exp14'
    save_path_root = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/train_modified'
    for i in os.listdir(imgs_path_root):
        if(i != '.DS_Store'):
            imgs_path = imgs_path_root + '/' + i
            labels_path = labels_path_root + '/' + i + '/labels'
            saves_path = save_path_root + '/' + i
            if(not os.path.exists(saves_path)):
                os.makedirs(saves_path)
            for label in os.listdir(labels_path):
                img_path = os.path.join(imgs_path, label)
                img_path = img_path[0:-4]
                img_path = img_path + '.jpg'
                img = cv2.imread(img_path)
                w = img.shape[1]
                h = img.shape[0]
                label_path = os.path.join(labels_path, label)
                f = open(label_path, 'r+', encoding='utf-8')
                for line in f.readlines():
                    msg = line.split(" ")
                    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                    # print(x1, ",", y1, ",", x2, ",", y2)
                    res = img[y1:y2,x1:x2]
                    res = cv2.resize(res, (64, 128), interpolation=cv2.INTER_CUBIC)
                    save_path = label[0:-4] + '.jpg'
                    save_path = os.path.join(saves_path, save_path)
                    cv2.imwrite(save_path,res)

if __name__ == '__main__':
    main()

