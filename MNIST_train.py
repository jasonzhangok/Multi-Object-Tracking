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
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2))
        self.c2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=2, stride=2),
                                 nn.Dropout2d(p=0.1),nn.Flatten(0,-1))
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.fc2 = nn.Linear(1024, 128)



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
    EuclideanDis = nn.PairwiseDistance(p=2)
    loss = EuclideanDis(plot1, plot2)
    return loss

if __name__ == '__main__':
    #Parameters:
    train_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_train'
    test_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_test'
    initial_lr = 0.001
    net_save_flag = 1
    net_load_flag = 1
    max_epoch = 10
    loss_distance = 5
    threshold = 2



    #load train and test data
    train_img_data = torchvision.datasets.ImageFolder(train_data_path ,transform=transform)
    test_img_data = torchvision.datasets.ImageFolder(test_data_path,transform=transform)
    test_data = torch.utils.data.DataLoader(test_img_data, batch_size=1,shuffle=True, num_workers=1,drop_last=True)
    #calculate the begin index of each class
    number_image_num = [0 for i in range(len(train_img_data.classes))]
    cur_image_class_num = 0
    for i in range(len(train_img_data)):
        if(i != len(train_img_data) - 1):
            if(train_img_data.targets[i] != train_img_data.targets[i+1]):
                number_image_num[cur_image_class_num] = i
                cur_image_class_num = cur_image_class_num + 1
        else:
            number_image_num[cur_image_class_num] = i

    #calculate the number of images of each class
    class_image_num = [0 for _ in range(len(train_img_data.classes))]
    class_image_num[0] = number_image_num[0] + 1
    for i in range(1, len(train_img_data.classes)):
        class_image_num[i] = number_image_num[i] - number_image_num[i - 1]

    #new a net
    net = MyNet()
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(),lr=initial_lr)
    print("Training on {}".format(DEVICE))
    #try to print the whold tensor for console
    torch.set_printoptions(profile='full')

    for epoch in range(max_epoch):
        if (net_load_flag == 1):
            pthfile = 'weights/weight' + str(epoch) + '.pth'
            net.load_state_dict(torch.load(pthfile))
        else:
            print('epoch {}'.format(epoch + 1))
            net.train()
            times = 0
            for j in range(len(train_img_data.classes)-1):
                print("Calculating {}-th group of data".format(j))
                # flag = [0] * class_image_num[j]
                # possible combination of two people's images
                possible_com = []
                for i in range(class_image_num[j]//2):
                    #get two possible random index to represent two image of same class
                    randindex1 = random.randint(0, class_image_num[j]-1)
                    randindex2 = random.randint(0, class_image_num[j]-1)
                    while (randindex1 == randindex2):
                        randindex1 = random.randint(0, class_image_num[j])
                        randindex2 = random.randint(0, class_image_num[j])

                    randindex3 = random.randint(0, len(train_img_data)-1)
                    if(j == 0):
                        while(randindex3 <= number_image_num[j]):
                            randindex3 = random.randint(0, len(train_img_data))
                    elif(randindex3 == len(train_img_data.classes)-1):
                        while(randindex3 >= number_image_num[j-1]):
                            randindex3 = random.randint(0, len(train_img_data))
                    else:
                        while(1):
                            if randindex3 >= number_image_num[j - 1] and randindex3 <= number_image_num[j]:
                                randindex3 = random.randint(0, len(train_img_data))
                            else:
                                break
                    if(j == 0):
                        possible_com.append([randindex1,randindex2,randindex3])
                    else:
                        possible_com.append([randindex1 + number_image_num[j-1],randindex2 + number_image_num[j-1],randindex3])

                # possible_com = random.sample(possible_com, len(possible_com))
                for i in tqdm(possible_com,desc = "Finished: "):
                    #print proporation of completeness
                    # cur_batch_x = torch.empty((3,28,28),device=DEVICE,dtype=torch.float32)
                    # print(train_img_data[i[0]][0])
                    # print(cur_batch_x[0])
                    cur_batch_x = torch.stack([
                        train_img_data[i[0]][0][0],
                        train_img_data[i[1]][0][0],
                        train_img_data[i[2]][0][0]
                    ], dim=0)

                    cur_batch_x = Variable(cur_batch_x).to(DEVICE)
                    #TODO:Solve the dimension question(3 to 1) Solved

                    #temp = net(cur_batch_x)
                    out0 = net(torch.reshape(cur_batch_x[0],(1,28,28)))
                    out1 = net(torch.reshape(cur_batch_x[1],(1,28,28)))
                    out2 = net(torch.reshape(cur_batch_x[2],(1,28,28)))
                    loss = LossFunction(out0, out1) + max(0,loss_distance-LossFunction(out0, out2))

                    if(times % 10 == 0):
                        print('loss:', loss)
                        print('loss1:', LossFunction(out0, out1))
                        print('loss2:', LossFunction(out0, out2))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.mps.empty_cache()
                    times = times + 1
                if(net_save_flag == 1):
                    weights_save_name = 'weights/weight' + str(epoch) + '.pth'
                    torch.save(net.state_dict(), weights_save_name)

        print("Testing:")
        #test
        correct_class_num = 0
        # TODO:Attention: There is not need for us to classify whether the test image belongs to the right class as input We just need to decide how many class we have and when input a new image, we classify it to the correct class
        class_count = 0
        #  class_weight_berycenter: store image tags(class) and the output of the net with the image input to calculate the barycenter
        class_weight_berycenter = []
        for test in test_data:
            test_out = net(torch.reshape(test[0][0][0],(1,28,28)))
            #record current test class number and weight i.e. its NumberWeight
            temp_class = NumberWeight()
            if(len(class_weight_berycenter) == 0):
                class_count = class_count + 1
                temp_class.identity = test[1]
                temp_class.weight = test_out
                temp_class.count = 1
                class_weight_berycenter.append(temp_class)
                correct_class_num = correct_class_num + 1
            else:
                distance = []
                for i in class_weight_berycenter:
                    temp_distance = LossFunction(i.weight, test_out)
                    distance.append(temp_distance)
                min_distance = min(distance)
                #The image input's feature vector min distance is greater than the threahold: the image belongs to a new class
                if(min_distance > threshold):
                    class_count = class_count + 1
                    temp_class.identity = test[1]
                    temp_class.weight = test_out
                    temp_class.count = 1
                    class_weight_berycenter.append(temp_class)
                    correct_class_num = correct_class_num + 1
                else:
                    min_index = distance.index(min_distance)
                    class_weight_berycenter[min_index].weight = class_weight_berycenter[min_index].weight * class_weight_berycenter[min_index].count + test_out
                    class_weight_berycenter[min_index].count = class_weight_berycenter[min_index].count + 1
                    class_weight_berycenter[min_index].weight = class_weight_berycenter[min_index].weight / class_weight_berycenter[min_index].count
                    if(class_weight_berycenter[min_index].identity == test[1]):
                        correct_class_num = correct_class_num + 1
            print(class_count)
        print("epoch %d finished",epoch)








