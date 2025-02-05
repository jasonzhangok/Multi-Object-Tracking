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
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')

class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.b2 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.b3 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1)
        )
        self.b4 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1)
        )
        self.b5 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1)
        )
        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Flatten(),nn.Linear(in_features=512, out_features=1024),
                                nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(in_features=1024, out_features=256))


    def forward(self, x):
        x = self.b1(x)         # 预处理

        x = self.b2(x)          # 四个卷积单元
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)

        x = self.b6(x)            # 池化

        return x


class Bottleneck(nn.Module):  # 卷积3层，F(X)和X的维度不等
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # 此处width=out_channel

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # downsample是用来将残差数据和卷积数据的shape变的相同，可以直接进行相加操作。
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 使用的残差块类型
                 blocks_num,  # 每个卷积层，使用残差块的个数
                 num_classes=1000,  # 训练集标签的分类个数
                 include_top=True,  # 是否在残差结构后接上pooling、fc、softmax
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 第一层卷积输出特征矩阵的深度，也是后面层输入特征矩阵的深度

        self.groups = groups
        self.width_per_group = width_per_group

        # 输入层有RGB三个分量，使得输入特征矩阵的深度是3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)函数：生成多个连续的残差块的残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:  # 默认为True，接上pooling、fc、softmax
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化下采样，无论输入矩阵的shape为多少，output size均为的高宽均为1x1
            # 使矩阵展平为向量，如（W,H,C）->(1,1,W*H*C)，深度为W*H*C
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，512 * block.expansion为输入深度，num_classes为分类类别个数

        for m in self.modules():  # 初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # _make_layer()函数：生成多个连续的残差块，(残差块类型，残差块中第一个卷积层的卷积核个数，残差块个数，残差块中卷积步长)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion
        # 后面的残差块不需要对X下采样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # 以非关键字参数形式，将layers列表，传入Sequential(),使其中残差块串联为一个残差结构
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  # 一般为True
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def ResNet50(include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1024, include_top=include_top)

# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])


transform = transforms.Compose([
    transforms.Resize((128, 64)),  # Resize images to 128x64 (Market-1501 format)
    # transforms.RandomHorizontalFlip(),  # Apply random flipping to augment data
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize between -1 and 1
])

def LossFunction(plot1, plot2):
    # EuclideanDis = nn.PairwiseDistance(p=2)
    # loss = EuclideanDis(plot1, plot2)
    # loss = F.pairwise_distance(plot1, plot2)
    # loss = torch.pow(loss, 2)
    loss = torch.cosine_similarity(plot1, plot2,0)
    return 1 - loss


def contrastive_loss(plot1, plot2, target, margin):
    euclidean_dis = F.pairwise_distance(plot1, plot2)
    target = target.view(-1)
    loss = (1 - target) * torch.pow(euclidean_dis, 2) + target * torch.pow(torch.clamp(margin - euclidean_dis, min=0),
                                                                           2)
    return loss


def triplet_loss(plot1, plot2, plot3, margin,batch_size):
    # anchor = plot1
    # positive = plot2
    # negative = plot3
    if(batch_size==1):
        dis_ap = LossFunction(plot1,plot2)
        dis_an = LossFunction(plot1,plot3)
        # dis_ap = torch.pow(F.pairwise_distance(plot1, plot2), 2)
        # dis_an = torch.pow(F.pairwise_distance(plot1, plot3), 2)
        loss = F.relu(dis_ap - dis_an + margin)
    else:
        loss = torch.tensor(0).to(DEVICE)
        for i in range(batch_size):
            dis_ap = LossFunction(plot1[i], plot2[i])
            dis_an = LossFunction(plot1[i], plot3[i])
            # dis_ap = torch.pow(F.pairwise_distance(plot1, plot2), 2)
            # dis_an = torch.pow(F.pairwise_distance(plot1, plot3), 2)
            temploss = F.relu(dis_ap - dis_an + margin)
            loss = loss + temploss
        loss = loss / batch_size
    return loss

def PossibleCombination(m):
    res = []
    for i in range(m):
        for j in range(i+1):
            res.append([i,j])
    return res

class TripletDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.targets = image_folder.targets
        self.classes = image_folder.classes
        self.class_to_indices = {i: [] for i in range(len(self.classes))}

        # Organize indices by class
        for idx, label in enumerate(self.targets):
            self.class_to_indices[label].append(idx)

    def __getitem__(self, index):
        anchor_idx = index
        anchor_img, anchor_label = self.image_folder[anchor_idx]

        # Positive sample: another image from the same class
        positive_idx = random.choice(self.class_to_indices[anchor_label])
        while positive_idx == anchor_idx:
            positive_idx = random.choice(self.class_to_indices[anchor_label])
        positive_img, _ = self.image_folder[positive_idx]

        # Negative sample: an image from a different class
        negative_label = random.choice([l for l in range(len(self.classes)) if l != anchor_label])
        negative_idx = random.choice(self.class_to_indices[negative_label])
        negative_img, _ = self.image_folder[negative_idx]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.image_folder)

if __name__ == '__main__':
    # # Parameters:
    # # data path
    # train_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/train'
    # # learning rate
    # initial_lr = 0.001
    # # whether save model parameters and whether load moder
    # net_save_flag = 1
    # # train epoch
    # max_epoch = 10
    # # loss function constant
    # Margin = 0.5
    #
    # # load train and test data
    # train_img_data = torchvision.datasets.ImageFolder(train_data_path, transform=transform)
    # # calculate the begin index of each class
    # number_image_num = [0 for i in range(len(train_img_data.classes))]
    # cur_image_class_num = 0
    # for i in range(len(train_img_data)):
    #     if (i != len(train_img_data) - 1):
    #         if (train_img_data.targets[i] != train_img_data.targets[i + 1]):
    #             number_image_num[cur_image_class_num] = i
    #             cur_image_class_num = cur_image_class_num + 1
    #     else:
    #         number_image_num[cur_image_class_num] = i
    #
    # # calculate the number of images of each class
    # class_image_num = [0 for _ in range(len(train_img_data.classes))]
    # class_image_num[0] = number_image_num[0] + 1
    # for i in range(1, len(train_img_data.classes)):
    #     class_image_num[i] = number_image_num[i] - number_image_num[i - 1]
    #
    # # new a net
    # net = ResNet50()
    # net = net.to(DEVICE)
    # optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    # print("Training on {}".format(DEVICE))
    # # try to print the whold tensor for console
    # torch.set_printoptions(profile='full')
    #
    # for epoch in range(max_epoch):
    #     print('epoch {}'.format(epoch + 1))
    #     net.train()
    #     times = 0
    #     for j in range(len(train_img_data.classes)):
    #         print("\nCalculating {}-th group of data".format(j))
    #         # flag = [0] * class_image_num[j]
    #         # possible combination of two people's images
    #         randindex1 = randindex2 = randindex3 = 0
    #         possible_com = []
    #         if(class_image_num[j] // 2 <= 10):
    #             index12 = PossibleCombination(class_image_num[j] // 2)
    #             for i in index12:
    #                 randindex1 = i[0]
    #                 randindex2 = i[1]
    #                 randindex3 = random.randint(0, len(train_img_data) - 1)
    #                 if (j == 0):
    #                     while (randindex3 <= number_image_num[j]):
    #                         randindex3 = random.randint(0, len(train_img_data) - 1)
    #                 elif (j == len(train_img_data.classes) - 1):
    #                     while (randindex3 >= number_image_num[j - 1]):
    #                         randindex3 = random.randint(0, len(train_img_data) - 1)
    #                 else:
    #                     while (1):
    #                         if randindex3 >= number_image_num[j - 1] and randindex3 <= number_image_num[j]:
    #                             randindex3 = random.randint(0, len(train_img_data) - 1)
    #                         else:
    #                             break
    #                 if (j == 0):
    #                     possible_com.append([randindex1, randindex2, randindex3])
    #                 else:
    #                     possible_com.append(
    #                         [randindex1 + number_image_num[j - 1], randindex2 + number_image_num[j - 1], randindex3])
    #         else:
    #             for i in range(class_image_num[j] // 2):
    #                 # get two possible random index to represent two image of same class
    #                 randindex1 = random.randint(0, class_image_num[j] - 1)
    #                 randindex2 = random.randint(0, class_image_num[j] - 1)
    #                 while (randindex1 == randindex2):
    #                     randindex1 = random.randint(0, class_image_num[j])
    #                     randindex2 = random.randint(0, class_image_num[j])
    #
    #                 randindex3 = random.randint(0, len(train_img_data) - 1)
    #                 if (j == 0):
    #                     while (randindex3 <= number_image_num[j]):
    #                         randindex3 = random.randint(0, len(train_img_data) - 1)
    #                 elif (j == len(train_img_data.classes) - 1):
    #                     while (randindex3 >= number_image_num[j - 1]):
    #                         randindex3 = random.randint(0, len(train_img_data) - 1)
    #                 else:
    #                     while (1):
    #                         if randindex3 >= number_image_num[j - 1] and randindex3 <= number_image_num[j]:
    #                             randindex3 = random.randint(0, len(train_img_data) - 1)
    #                         else:
    #                             break
    #                 if (j == 0):
    #                     possible_com.append([randindex1, randindex2, randindex3])
    #                 else:
    #                     possible_com.append(
    #                         [randindex1 + number_image_num[j - 1], randindex2 + number_image_num[j - 1], randindex3])
    #
    #         # possible_com = random.sample(possible_com, len(possible_com))
    #         for i in tqdm(possible_com, desc="Finished: "):
    #             # print proporation of completeness
    #             # cur_batch_x = torch.empty((3,28,28),device=DEVICE,dtype=torch.float32)
    #             # print(train_img_data[i[0]][0])
    #             # print(cur_batch_x[0])
    #             cur_batch_x = torch.stack([
    #                 train_img_data[i[0]][0],
    #                 train_img_data[i[1]][0],
    #                 train_img_data[i[2]][0]
    #             ], dim=0)
    #
    #             cur_batch_x = Variable(cur_batch_x).to(DEVICE)
    #             # TODO:Solve the dimension question(3 to 1) Solved
    #
    #             # temp = net(cur_batch_x)
    #             out0 = net(torch.reshape(cur_batch_x[0], (1, 3, 128, 64)))
    #             out1 = net(torch.reshape(cur_batch_x[1], (1, 3, 128, 64)))
    #             out2 = net(torch.reshape(cur_batch_x[2], (1, 3, 128, 64)))
    #             out0 = out0.squeeze(0)
    #             out1 = out1.squeeze(0)
    #             out2 = out2.squeeze(0)
    #             loss = triplet_loss(out0, out1, out2, Margin)
    #
    #             if (times % 100 == 0):
    #                 print('\nloss:', loss)
    #                 print('loss1:', LossFunction(out0, out1))
    #                 print('loss2:', LossFunction(out0, out2))
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             torch.mps.empty_cache()
    #             times = times + 1
    #             if(net_save_flag == 1 and times % 1000 == 0):
    #                 weights_save_name = 'weights/weightCOS' + str(epoch) + '.pth'
    #                 torch.save(net.state_dict(), weights_save_name)
    #             if(j == len(train_img_data.classes) - 1):
    #                 weights_save_name = 'weights/weightCOS' + str(epoch) + '.pth'
    #                 torch.save(net.state_dict(), weights_save_name)
    #     weights_save_name = 'weights/weightCOS' + str(epoch) + '.pth'
    #     torch.save(net.state_dict(), weights_save_name)


    train_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/train'
    # learning rate
    initial_lr = 0.001
    # whether save model parameters and whether load moder
    net_save_flag = 1
    # train epoch
    max_epoch = 10
    # loss function constant
    Margin = 0.5
    batch_size = 64  # Adjust batch size based on GPU memory
    num_workers = 4  # Set to the number of CPU cores for faster loading

    train_dataset = TripletDataset(torchvision.datasets.ImageFolder(train_data_path, transform=transform))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)

    net = ResNet50()
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    print("Training on {}".format(DEVICE))
    torch.set_printoptions(profile='full')

    # try to print the whold tensor for console
    torch.set_printoptions(profile='full')
    for epoch in range(max_epoch):
        print(f'Epoch {epoch + 1}')
        net.train()

        for batch_idx, (img_a, img_p, img_n) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}'):
            img_a, img_p, img_n = img_a.to(DEVICE), img_p.to(DEVICE), img_n.to(DEVICE)

            out_a = net(img_a)
            out_p = net(img_p)
            out_n = net(img_n)

            loss = triplet_loss(out_a, out_p, out_n, Margin,batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.mps.empty_cache()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item()}')

        torch.save(net.state_dict(), f'weights/weightCOS{epoch}.pth')


