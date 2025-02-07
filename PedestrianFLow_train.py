import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WarmupCosineAnnealingLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=1e-6, base_lr=1e-3, min_lr=1e-6,
                 last_epoch=-1):
        """
        :param optimizer: PyTorch optimizer
        :param warmup_epochs: 预热阶段的 epochs 数
        :param total_epochs: 总的 epochs 数
        :param warmup_start_lr: 预热开始时的学习率
        :param base_lr: 预热结束后的基准学习率
        :param min_lr: 余弦退火最低学习率
        :param last_epoch: 用于恢复训练
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性 Warmup 计算
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            # 余弦退火计算
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        return [lr for _ in self.base_lrs]


class Bottleneck(nn.Module):  # 鍗风Н3灞傦紝F(X)鍜孹鐨勭淮搴︿笉绛?
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # 姝ゅwidth=out_channel

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
        # downsample鏄敤鏉ュ皢娈嬪樊鏁版嵁鍜屽嵎绉暟鎹殑shape鍙樼殑鐩稿悓锛屽彲浠ョ洿鎺ヨ繘琛岀浉鍔犳搷浣溿??
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
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group


        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

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

        if self.include_top:  # 涓?鑸负True
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def ResNet50(include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2048, include_top=include_top)



net = ResNet50()
net = net.to(DEVICE)


transform = transforms.Compose([
    transforms.Resize((128, 64)),  # Resize images to 128x64 (Market-1501 format)
    # transforms.RandomHorizontalFlip(),  # Apply random flipping to augment data
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize between -1 and 1
])

def LossFunction(plot1, plot2):

    loss = torch.cosine_similarity(plot1, plot2,-1)
    return 1 - loss


def contrastive_loss(plot1, plot2, target, margin):
    euclidean_dis = F.pairwise_distance(plot1, plot2)
    target = target.view(-1)
    loss = (1 - target) * torch.pow(euclidean_dis, 2) + target * torch.pow(torch.clamp(margin - euclidean_dis, min=0),
                                                                           2)
    return loss


def triplet_loss(plot1, plot2, plot3, margin):
    dis_ap = LossFunction(plot1, plot2)
    dis_an = LossFunction(plot1, plot3)
    losses = F.relu(dis_ap - dis_an + margin)
    return losses.mean()

class TripletDataset(Dataset):
    def __init__(self, image_folder, margin=0.5):
        self.image_folder = image_folder
        self.targets = image_folder.targets
        self.classes = image_folder.classes
        self.class_to_indices = {i: [] for i in range(len(self.classes))}
        self.margin = margin

        # Organize indices by class
        for idx, label in enumerate(self.targets):
            self.class_to_indices[label].append(idx)

    def __getitem__(self, index):
        # Anchor sample
        anchor_idx = index
        anchor_img, anchor_label = self.image_folder[anchor_idx]
        positive_img,_= self.image_folder[anchor_idx]
        negative_img,_= self.image_folder[anchor_idx]

        if (len(self.class_to_indices[anchor_label]) == 1):
            print(anchor_label)

        # Positive sample: another image from the same class
        if (len(self.class_to_indices[anchor_label]) != 1):
            positive_idx = random.choice(self.class_to_indices[anchor_label])
            while positive_idx == anchor_idx:
                positive_idx = random.choice(self.class_to_indices[anchor_label])
            positive_img, _ = self.image_folder[positive_idx]

            # Negative sample: an image from a different class
            negative_label = random.choice([l for l in range(len(self.classes)) if l != anchor_label])
            # Semi-Hard Negative: A sample from a different class that is still valid
            negative_idx = self._get_semi_hard_negative(anchor_idx,negative_label,positive_idx)
            negative_img, _ = self.image_folder[negative_idx]

        return anchor_img, positive_img, negative_img

    def _get_semi_hard_negative(self, anchor_idx, negative_label,positive_idx):
        """Find a valid negative sample where dist(anchor, negative) > dist(anchor, positive)"""
        net.eval()
        with torch.no_grad():
            anchor_img, _ = self.image_folder[anchor_idx]
            anchor_embedding = net(anchor_img.unsqueeze(0).to(DEVICE)).detach().cpu()  # Get feature representation
            positive_img, _ = self.image_folder[positive_idx]
            positive_embedding = net(positive_img.unsqueeze(0).to(DEVICE)).detach().cpu()
            dist_anchor_positive = LossFunction(anchor_embedding, positive_embedding)

            count = 0
            hardest_negative_idx = random.choice(self.class_to_indices[negative_label])
            for n_idx in self.class_to_indices[negative_label]:
                if(count < 3):
                    negative_img, _ = self.image_folder[n_idx]
                    negative_embedding = net(negative_img.unsqueeze(0).to(DEVICE)).detach().cpu()
                    dist_anchor_negative = LossFunction(anchor_embedding, negative_embedding)
                    if(dist_anchor_positive < dist_anchor_negative and dist_anchor_negative < dist_anchor_positive + self.margin):
                        hardest_negative_idx = n_idx
                else:
                    break
                count = count + 1

        return hardest_negative_idx

    def __len__(self):
        return len(self.image_folder)

if __name__ == '__main__':
    train_data_path = 'H:\\jason\\202501\\train'
    # learning rate
    initial_lr = 0.001
    # whether save model parameters and whether load moder
    net_save_flag = 1
    # train epoch
    max_epoch = 20
    # loss function constant
    Margin = 0.5
    batch_size = 8  # Adjust batch size based on GPU memory
    num_workers = 10  # Set to the number of CPU cores for faster loading

    # net = ResNet50()
    # net = net.to(DEVICE)

    train_dataset = TripletDataset(torchvision.datasets.ImageFolder(train_data_path, transform=transform),
                                    margin=Margin)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=5, total_epochs=max_epoch, base_lr=1e-3, min_lr=1e-6)
    print("Training on {}".format(DEVICE))
    torch.set_printoptions(profile='full')

    for epoch in range(max_epoch):
        print(f'Epoch {epoch + 1}')

        # test_items = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}')

        for batch_idx, (img_a, img_p, img_n) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}'):
            net.train()
            img_a, img_p, img_n = img_a.to(DEVICE), img_p.to(DEVICE), img_n.to(DEVICE)

            out_a = net(img_a)
            out_p = net(img_p)
            out_n = net(img_n)
            loss = triplet_loss(out_a, out_p, out_n, Margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 100 == 0:
                print()
                print(f' Loss: {loss.item()}')
                loss1 = LossFunction(out_a, out_p).mean()
                loss2 = LossFunction(out_a, out_n).mean()
                print(f' Loss1: {loss1.item()}')
                print(f' Loss2: {loss2.item()}')
                print(f"Epoch {epoch + 1}: Learning Rate = {scheduler.get_lr()[0]:.6f}")

        torch.save(net.state_dict(), f'weights/weight2048_{epoch}.pth')


