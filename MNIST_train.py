import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import cv2
import numpy as np
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision import models
import random
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import Sampler
from torch import randperm
import random
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyNet(nn.Module):
    def __init__(self):
        # super().__init__()
        # self.c1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        # self.c2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),
        #                          nn.Dropout2d(p=0.1),nn.Flatten(0,-1))
        # self.fc1 = nn.Linear(128*7*7, 1024)
        # self.fc2 = nn.Linear(1024, 128)
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.c2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
                                nn.Dropout2d(p=0.1), nn.Flatten(0, -1))
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, X):
        X = self.c1(X)
        # Pass through c2 (including AdaptiveMaxPool2d and Flatten)
        X = self.c2(X)
        X = self.fc1(X)
        # Pass through fully connected layers with ReLU in between
        X = F.relu(X)
        X = self.fc2(X)
        # Normalize the output
        return X


transform = transforms.Compose([
    transforms.ToTensor(),
])


class NumberWeight():
    def __init__(self):
        self.identity = -1
        self.weight = []
        self.count = 0


def LossFunction(plot1, plot2):
    # EuclideanDis = nn.PairwiseDistance(p=2)
    # loss = EuclideanDis(plot1, plot2)
    loss = F.pairwise_distance(plot1, plot2)
    loss = torch.pow(loss, 2)
    return loss


def contrastive_loss(plot1, plot2, target, margin):
    euclidean_dis = F.pairwise_distance(plot1, plot2)
    target = target.view(-1)
    loss = (1 - target) * torch.pow(euclidean_dis, 2) + target * torch.pow(torch.clamp(margin - euclidean_dis, min=0),
                                                                           2)
    return loss


def triplet_loss(plot1, plot2, plot3, margin):
    # anchor = plot1
    # positive = plot2
    # negative = plot3
    dis_ap = torch.pow(F.pairwise_distance(plot1, plot2), 2)
    dis_an = torch.pow(F.pairwise_distance(plot1, plot3), 2)
    loss = F.relu(dis_ap - dis_an + margin)
    return loss


if __name__ == '__main__':
    # Parameters:
    # data path
    train_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_train'
    # learning rate
    initial_lr = 0.001
    # whether save model parameters and whether load moder
    net_save_flag = 1
    # train epoch
    max_epoch = 10
    # loss function constant
    Margin = 5

    # load train and test data
    train_img_data = torchvision.datasets.ImageFolder(train_data_path, transform=transform)
    # calculate the begin index of each class
    number_image_num = [0 for i in range(len(train_img_data.classes))]
    cur_image_class_num = 0
    for i in range(len(train_img_data)):
        if (i != len(train_img_data) - 1):
            if (train_img_data.targets[i] != train_img_data.targets[i + 1]):
                number_image_num[cur_image_class_num] = i
                cur_image_class_num = cur_image_class_num + 1
        else:
            number_image_num[cur_image_class_num] = i

    # calculate the number of images of each class
    class_image_num = [0 for _ in range(len(train_img_data.classes))]
    class_image_num[0] = number_image_num[0] + 1
    for i in range(1, len(train_img_data.classes)):
        class_image_num[i] = number_image_num[i] - number_image_num[i - 1]

    # new a net
    net = MyNet()
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    print("Training on {}".format(DEVICE))
    # try to print the whold tensor for console
    torch.set_printoptions(profile='full')

    for epoch in range(max_epoch):
        print('epoch {}'.format(epoch + 1))
        net.train()
        times = 0
        for j in range(len(train_img_data.classes) - 1):
            print("Calculating {}-th group of data".format(j))
            # flag = [0] * class_image_num[j]
            # possible combination of two people's images
            possible_com = []
            for i in range(class_image_num[j] // 2):
                # get two possible random index to represent two image of same class
                randindex1 = random.randint(0, class_image_num[j] - 1)
                randindex2 = random.randint(0, class_image_num[j] - 1)
                while (randindex1 == randindex2):
                    randindex1 = random.randint(0, class_image_num[j])
                    randindex2 = random.randint(0, class_image_num[j])

                randindex3 = random.randint(0, len(train_img_data) - 1)
                if (j == 0):
                    while (randindex3 <= number_image_num[j]):
                        randindex3 = random.randint(0, len(train_img_data) - 1)
                elif (randindex3 == len(train_img_data.classes) - 1):
                    while (randindex3 >= number_image_num[j - 1]):
                        randindex3 = random.randint(0, len(train_img_data) - 1)
                else:
                    while (1):
                        if randindex3 >= number_image_num[j - 1] and randindex3 <= number_image_num[j]:
                            randindex3 = random.randint(0, len(train_img_data) - 1)
                        else:
                            break
                if (j == 0):
                    possible_com.append([randindex1, randindex2, randindex3])
                else:
                    possible_com.append(
                        [randindex1 + number_image_num[j - 1], randindex2 + number_image_num[j - 1], randindex3])

            # possible_com = random.sample(possible_com, len(possible_com))
            for i in tqdm(possible_com, desc="Finished: "):
                # print proporation of completeness
                # cur_batch_x = torch.empty((3,28,28),device=DEVICE,dtype=torch.float32)
                # print(train_img_data[i[0]][0])
                # print(cur_batch_x[0])
                cur_batch_x = torch.stack([
                    train_img_data[i[0]][0][0],
                    train_img_data[i[1]][0][0],
                    train_img_data[i[2]][0][0]
                ], dim=0)

                cur_batch_x = Variable(cur_batch_x).to(DEVICE)
                # TODO:Solve the dimension question(3 to 1) Solved

                # temp = net(cur_batch_x)
                out0 = net(torch.reshape(cur_batch_x[0], (1, 1, 28, 28)))
                out1 = net(torch.reshape(cur_batch_x[1], (1, 1, 28, 28)))
                out2 = net(torch.reshape(cur_batch_x[2], (1, 1, 28, 28)))
                loss = triplet_loss(out0, out1, out2, Margin)

                if (times % 100 == 0):
                    print('\nloss:', loss)
                    print('loss1:', LossFunction(out0, out1))
                    print('loss2:', LossFunction(out0, out2))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.mps.empty_cache()
                times = times + 1
            if (net_save_flag == 1):
                weights_save_name = 'weights/weight' + str(epoch) + '.pth'
                torch.save(net.state_dict(), weights_save_name)


