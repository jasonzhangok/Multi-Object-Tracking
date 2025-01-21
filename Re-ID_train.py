import torch
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
import random
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import Sampler
from torch import randperm


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return F.relu(Y)

#We set input channels be 3 and the number channels be 32
class DeepSort(nn.Module):
    def __init__(self):
        super().__init__()
        #first two layers of Conv and 1 layer of max pooling
        self.b1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(32), nn.Sigmoid(),nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(32), nn.Sigmoid(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(Residual(32, 32))
        self.b3 = nn.Sequential(Residual(32, 32))
        self.b4 = nn.Sequential(Residual(32, 64, use_1x1conv=True, strides=2))
        self.b5 = nn.Sequential(Residual(64, 64))
        self.b6 = nn.Sequential(Residual(64, 128, use_1x1conv=True, strides=2))
        self.b7 = nn.Sequential(Residual(128, 128))
        self.b8 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)) , nn.Flatten() , nn.Linear(128, 128))
        # self.b8 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)) , nn.Flatten() , nn.Linear(128, 128), nn.BatchNorm1d(128))

        self.Net = nn.Sequential(self.b1,self.b2,self.b3,self.b4,self.b5,self.b6,self.b7,self.b8)

    def forward(self, X):

        return F.normalize(self.Net(X), p=2, dim=1)

class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)

class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.ReLU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                           )

        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                           Inception(512, 160, (112, 224), (24, 64), 64),
                           Inception(512, 128, (128, 256), (24, 64), 64),
                           Inception(512, 112, (144, 288), (32, 64), 64),
                           Inception(528, 256, (160, 320), (32, 128), 128),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                           Inception(832, 384, (192, 384), (48, 128), 128),
                           nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Flatten())

        self.net1 = nn.Sequential(self.b1, self.b2)
        self.net2 = nn.Sequential(self.net1,self.b3, self.b4, self.b5, nn.Linear(1024, 128))

    def forward(self, X):
        return F.normalize(self.net2(X), p=2, dim=1)

# class LossFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, plot1, plot2):
#         EuclideanDis = nn.PairwiseDistance(p=2)
#         loss = EuclideanDis(plot1, plot2)
#         return loss





transform = transforms.Compose([
    transforms.ToTensor(),
])


def LossFunction(plot1, plot2):
    EuclideanDis = nn.PairwiseDistance(p=2)
    loss = EuclideanDis(plot1, plot2)
    return loss

def PossibleCombination(m,n):
    res = []
    for i in range(m):
        for j in range(n):
            for k in range(i+1,m):
                res.append([i,k,j])
    return res

if __name__ == '__main__':
    #load train and test data
    train_img_data = torchvision.datasets.ImageFolder('/Users/jason/IdeaProjects/PeopleFlowDetection/Marker-1501testlarge/train',transform=transform)
    # train_data1 = torch.utils.data.DataLoader(train_img_data, batch_size=2,shuffle=True,num_workers=1,drop_last=True)
    test_img_data = torchvision.datasets.ImageFolder('/Users/jason/IdeaProjects/PeopleFlowDetection/Marker-1501testlarge/test',transform=transform)
    test_data = torch.utils.data.DataLoader(test_img_data, batch_size=2,shuffle=False, num_workers=1,drop_last=True)

    #the index of each person images
    person_image_num = [0 for i in range(len(train_img_data.classes))]
    cur_image_class_num = 0
    for i in range(len(train_img_data)):
        if(i != len(train_img_data) - 1):
            if(train_img_data.targets[i] != train_img_data.targets[i+1]):
                person_image_num[cur_image_class_num] = i
                cur_image_class_num = cur_image_class_num + 1
        else:
            person_image_num[cur_image_class_num] = i




    #new a net
    # net = DeepSort()
    net = GoogleNet()
    net = net.to(DEVICE)
    initial_lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(),lr=initial_lr)
    # scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda epoch: 1 / (epoch + 1))
    print("Training on {}".format(DEVICE))
    #try to print the whold tensor for console
    torch.set_printoptions(profile='full')


    for epoch in range(10):
        print('epoch {}'.format(epoch + 1))
        net.train()
        # training-----------------------------

        #constant that added to the loss function
        # constant = torch.tensor(1e-5).to(DEVICE)
        # person_diff_dist = torch.zeros(101,device=DEVICE)
        # person_same_dist = torch.ones(101,device=DEVICE)

        times = 0
        for j in range(len(train_img_data.classes)-1):
            print("Calculating {}-th group of data".format(j))
            # possible combination of two people's images
            possible_com = []
            if(j == 0):
                possible_com = PossibleCombination(person_image_num[j] + 1, person_image_num[1] - person_image_num[0])
            else:
                possible_com = PossibleCombination(person_image_num[j] - person_image_num[j-1] , person_image_num[j+1] - person_image_num[j])
            possible_com = random.sample(possible_com, len(possible_com)//2)
            for i in possible_com:
            #print proporation of completeness
                if(times % 100 == 0 and times != 0):
                    print("Finished {} batches".format(times))
                    print("Finished {:.2f}%".format(times/(len(possible_com) * (len(train_img_data.classes)-1))*100))
                    # person_same_dist[100] = min(person_same_dist)
                    # person_diff_dist[100] = max(person_diff_dist)
                    # # constant.detach().fill_(person_diff_dist[100] - person_same_dist[100])
                    # constant.detach().fill_(person_diff_dist[100])
                    # constant = max(person_diff_dist) - min(person_same_dist)


                    # print(class_weight_sum.tolist())

                #get the training data(3, 2 of same type , one of another type)
                #the different person image information
                cur_batch_x = torch.empty((3,3,128,64),device=DEVICE,dtype=torch.float32)
                if(j == 0):
                    cur_batch_x[0] = train_img_data[i[0]][0]
                    cur_batch_x[1] = train_img_data[i[1]][0]
                    cur_batch_x[2] = train_img_data[i[2] + person_image_num[j] + 1][0]
                else:
                    cur_batch_x[0] = train_img_data[i[0] + person_image_num[j-1] + 1][0]
                    cur_batch_x[1] = train_img_data[i[1] + person_image_num[j-1] + 1][0]
                    cur_batch_x[2] = train_img_data[i[2] + person_image_num[j] + 1][0]
                #decide whether the two batch images belongs to same person
                # cur_batch_class = 1 if batch_x[1][0] == batch_x[1][1] else 0
                # print(batch_x[1][0],batch_x[1][1],cur_different_x[1])

                cur_batch_x = Variable(cur_batch_x).to(DEVICE)
                # print(out[0].shape)
                # print(class_weight_sum[cur_class].shape)
                #count the number of each class and calculate loss
                # if(cur_batch_class == 1):
                out = net(cur_batch_x)
                # out = out / torch.norm(out)
                #if the two images belongs to same person, it is still possible that the two images are from the not the same person as before

                # person_diff_dist[times%100] = LossFunction(out[0], out[2])
                # person_same_dist[times%100] = LossFunction(out[0], out[1])
                loss = LossFunction(out[0], out[1]) / LossFunction(out[0], out[2])
                if(times % 10 == 0):
                    print(loss)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                #     class_weight_sum[cur_class] = class_weight_sum[cur_class] + out[0]
                #     image_num[cur_class] = image_num[cur_class] + 1
                #     cur_class = cur_class + 1
                #     class_weight_sum[cur_class] = class_weight_sum[cur_class] + out[1]
                #     image_num[cur_class] = image_num[cur_class] + 1
                #
                #
                #     if(cur_different_x[1] == batch_x[1][0]):
                #         loss = -LossFunction(out[0], out[1]) + LossFunction(out[0], out[2]) - LossFunction(out[1], out[2])
                #     elif(cur_different_x[1] == batch_x[1][1]):
                #         loss = -LossFunction(out[0], out[1]) - LossFunction(out[0], out[2]) + LossFunction(out[1], out[2])
                #     else:
                #         loss = -LossFunction(out[0], out[1]) - LossFunction(out[0], out[2]) - LossFunction(out[1], out[2])
                #     train_correct = (pred == batch_y).sum()
                #     train_acc += train_correct.item()
                #     print(loss)
                torch.mps.empty_cache()
                times = times + 1

        # for batch_x in train_data1:
        #     #print proporation of completeness
        #     if(times % 100 == 0 and times != 0):
        #         print("Finished {} batches".format(times))
        #         print("Finished {:.2f}%".format(times/len(train_data1)*100))
        #         person_same_dist[100] = min(person_same_dist)
        #         person_diff_dist[100] = max(person_diff_dist)
        #         # constant.detach().fill_(person_diff_dist[100] - person_same_dist[100])
        #         constant.detach().fill_(person_diff_dist[100])
        #         # constant = max(person_diff_dist) - min(person_same_dist)
        #
        #
        #         # print(class_weight_sum.tolist())
        #
        #     #get the training data(3, 2 of same type , one of another type)
        #     cur_batch_x = batch_x
        #     #get the another image that is from the same person of batch_x
        #     # print(batch_x[1][0].item())
        #     random_same_index = 0
        #     if(batch_x[1][0].item() == 0):
        #         random_same_index = random.randint(0,person_image_num[batch_x[1][0].item()])
        #     else:
        #         random_same_index = random.randint(person_image_num[batch_x[1][0].item()-1],person_image_num[batch_x[1][0].item()])
        #     cur_batch_x1 = cur_batch_x[0]
        #     #the different person image information
        #     cur_batch_x = torch.empty((3,3,128,64),device=DEVICE,dtype=torch.float32)
        #     cur_batch_x[0] = cur_batch_x1[0]
        #     cur_batch_x[1] = train_img_data[random_same_index][0]
        #     cur_batch_x[2] = cur_batch_x1[1]
        #     #decide whether the two batch images belongs to same person
        #     # cur_batch_class = 1 if batch_x[1][0] == batch_x[1][1] else 0
        #     # print(batch_x[1][0],batch_x[1][1],cur_different_x[1])
        #     cur_batch_x = Variable(cur_batch_x).to(DEVICE)
        #     # print(out[0].shape)
        #     # print(class_weight_sum[cur_class].shape)
        #     #count the number of each class and calculate loss
        #     # if(cur_batch_class == 1):
        #     out = net(cur_batch_x)
        #     # out = out / torch.norm(out)
        #     #if the two images belongs to same person, it is still possible that the two images are from the not the same person as before
        #     class_weight_sum[batch_x[1][0].item()] = class_weight_sum[batch_x[1][0].item()] + out[0]
        #     person_diff_dist[times%100] = LossFunction(out[1], out[2])
        #     person_same_dist[times%100] = LossFunction(out[0], out[1])
        #     loss = LossFunction(out[0], out[1]) - LossFunction(out[0], out[2]) + constant
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     # scheduler.step()
        #     #     class_weight_sum[cur_class] = class_weight_sum[cur_class] + out[0]
        #     #     image_num[cur_class] = image_num[cur_class] + 1
        #     #     cur_class = cur_class + 1
        #     #     class_weight_sum[cur_class] = class_weight_sum[cur_class] + out[1]
        #     #     image_num[cur_class] = image_num[cur_class] + 1
        #     #
        #     #
        #     #     if(cur_different_x[1] == batch_x[1][0]):
        #     #         loss = -LossFunction(out[0], out[1]) + LossFunction(out[0], out[2]) - LossFunction(out[1], out[2])
        #     #     elif(cur_different_x[1] == batch_x[1][1]):
        #     #         loss = -LossFunction(out[0], out[1]) - LossFunction(out[0], out[2]) + LossFunction(out[1], out[2])
        #     #     else:
        #     #         loss = -LossFunction(out[0], out[1]) - LossFunction(out[0], out[2]) - LossFunction(out[1], out[2])
        #     #     train_correct = (pred == batch_y).sum()
        #     #     train_acc += train_correct.item()
        #     #     print(loss)
        #     times = times + 1
        #     torch.mps.empty_cache()
        #calculate current weight of each class after training epoch
        #get the middle point of the classes We can modify this to achieve higher accuracy
        class_weight_sum = torch.zeros(len(train_img_data.classes),128,device=DEVICE)
        class_count_sum = [0 for _ in range(len(train_img_data.classes))]
        for i in range(0,person_image_num[-1]+1,2):
            cur_batch_x = torch.empty((2,3,128,64),device=DEVICE,dtype=torch.float32)
            cur_batch_x[0] = train_img_data[i][0]
            cur_batch_x[1] = train_img_data[i+1][0]
            cur_batch_class = [0] * 2
            cur_batch_class[0] = train_img_data[i][1]
            class_count_sum[cur_batch_class[0]] = class_count_sum[cur_batch_class[0]] + 1
            cur_batch_class[1] = train_img_data[i+1][1]
            class_count_sum[cur_batch_class[1]] = class_count_sum[cur_batch_class[1]] + 1
            cur_batch_x = Variable(cur_batch_x).to(DEVICE)
            out = net(cur_batch_x)
            class_weight_sum[cur_batch_class[0]] = class_weight_sum[cur_batch_class[0]] + out[0]
            class_weight_sum[cur_batch_class[1]] = class_weight_sum[cur_batch_class[1]] + out[1]
            torch.mps.empty_cache()
        for i in range(len(train_img_data.classes)):
            class_weight_sum[i] = class_weight_sum[i] / class_count_sum[i]
            # print(cur_class_weight_point[i])
            # print()

        print("Finished 100%")
        print("Finished Train, begin test:")
        # net.eval()
        eval_acc = 0.

        correct_num = 0
        times = 0
        for batch_x in test_data:
            if(times % 50 == 0 and times != 0):
                print("Finished testing {}% batches".format(times/len(test_data)*100))
            cur_batch_x = batch_x[0]
            # cur_batch_class = 1 if batch_x[1][0] == batch_x[1][1] else 0
            cur_batch_x = Variable(cur_batch_x).to(DEVICE)
            # print(cur_batch_x)
            out = net(cur_batch_x)
            # out = out / torch.norm(out)
            # print(out)
            class_dist0 = [LossFunction(out[0],class_weight_sum[i]) for i in range(len(train_img_data.classes))]
            class_dist1 = [LossFunction(out[1],class_weight_sum[i]) for i in range(len(train_img_data.classes))]
            print(class_dist0)
            print(class_dist1)
            distance = LossFunction(out[0],out[1])
            print(distance)
            pred0 = class_dist0.index(min(class_dist0))
            pred1 = class_dist1.index(min(class_dist1))
            if(pred0 == batch_x[1][0]):
                correct_num = correct_num + 1
            if(pred1 == batch_x[1][1]):
                correct_num = correct_num + 1
            times = times + 1
        print('Acc: {:.6f}'.format(correct_num / (len(train_img_data.classes))))

    # imagepath1 = "/Users/jason/IdeaProjects/PeopleFlowDetection/Market-1501/train/0002/0002_c1s1_000551_01.jpg"
    # imagepath2 = "/Users/jason/IdeaProjects/PeopleFlowDetection/Market-1501/train/0002/0002_c1s1_000776_01.jpg"
    # image1 = cv2.imread(imagepath1)
    # image2 = cv2.imread(imagepath2)
    # rgb1 = np.array(cv2.split(image1))
    # rgb2 = np.array(cv2.split(image2))
    # tensor1 = torch.empty(1,3,128,64)
    # tensor2 = torch.empty(1,3,128,64)
    # tensor1[0] = torch.from_numpy(rgb1)
    # tensor2[0] = torch.from_numpy(rgb2)
    # tensor1 = tensor1.to(torch.float32)
    # tensor2 = tensor2.to(torch.float32)
    # model = DeepSort()
    # res1 = model(tensor1)
    # res2 = model(tensor2)
    # Loss = LossFunction(res1, res2)




