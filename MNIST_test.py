import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import random
from tqdm import tqdm
import train

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MyNet = train.MyNet
LossFunction = train.LossFunction
contrastive_loss = train.contrastive_loss
triple_loss = train.triplet_loss


transform = train.transforms.Compose([transforms.ToTensor()])

class NumberWeight():
    def __init__(self):
        self.identity = -1
        self.weight = []
        self.count = 0


if __name__ == '__main__':
    #Parameters:
    #data path
    test_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_test'
    #learning rate
    initial_lr = 0.001
    #whether save model parameters and whether load moder
    net_load_flag = 1
    #different class threshold
    threshold = 13
    #whether test or not
    test_index = 1
    #weight path
    weight_index = 9
    pthfile = 'weights/weight' + str(weight_index) + '.pth'
    #train image data class
    total_image_class = len(train.train_img_data.classes)

    #load train and test data
    test_img_data = torchvision.datasets.ImageFolder(test_data_path,transform=transform)
    #test_data = torch.utils.data.DataLoader(test_img_data, batch_size=1,shuffle=True, num_workers=4,drop_last=True)
    #calculate the begin index of each class



    #test
    print("Testing:")
    #enter net testing module
    net = MyNet()
    temp = torch.load(pthfile, map_location=DEVICE)
    net.load_state_dict(temp)
    net.eval()
    correct_class_num = 0
    # TODO:Attention: There is not need for us to classify whether the test image belongs to the right class as input We just need to decide how many class we have and when input a new image, we classify it to the correct class
    class_count = 0

    # TODO:Organize and analysis in_class_dist and between_class_dist
    # Calculate distance There are some problem
    # in_class_dist.sort(reverse=True)
    # between_class_dist.sort()
    # threshold_sup = between_class_dist[10000]
    # threshold_inf = in_class_dist[10000]
    # threshold = threshold_sup + threshold_inf
    #  class_weight_berycenter: store image tags(class) and the output of the net with the image input to calculate the barycenter
    class_weight_berycenter = []
    # shuffle test dataset index
    shuffle_test_index = [i for i in range(len(test_img_data))]
    random.shuffle(shuffle_test_index)
    # sequence test dataset index
    seq_test_index = shuffle_test_index[:int(len(test_img_data)*0.8)]
    seq_test_index.sort()

    # # Test Accuracy: dynamic threshold
    # # TODO: Calculate in-class distance and between-class distance
    # # the index of class(from 0 to 9) in seq_test_index(We calculate this in order to calculate the in-class distance and between class distance)
    # # each class index number is from class_seq_test_index[i-1] to class_seq_test_index[i] - 1
    # class_seq_test_index = []
    # for j in range(len(seq_test_index)-1):
    #     if(test_img_data[seq_test_index[j]][1] != test_img_data[seq_test_index[j+1]][1]):
    #         class_seq_test_index.append(j)
    # in_class_dist = [[] for _ in range(len(train_img_data.classes))]
    # between_class_dist = [[torch.tensor(0) for _ in range(len(train_img_data.classes))] for i in range(len(train_img_data.classes))]
    # # calculate the distance in class(from class 0 to class 9): j class index, i plot1,k plot2,We can get the distance in class between different plots
    # for j in range(len(train_img_data.classes)):
    #     if(j == 0):
    #         for i in range(class_seq_test_index[j]+1):
    #             #calculate barycenter
    #             plot1 = net(torch.reshape(train_img_data[seq_test_index[i]][0][0], (1, 1, 28, 28)))
    #             if(i == 0):
    #                 class_weight_berycenter.append(plot1)
    #             else:
    #                 class_weight_berycenter[j] = class_weight_berycenter[j] + plot1
    #             #calculate the distance in class
    #             for k in range(i+1,class_seq_test_index[j]+1):
    #                 plot2 = net(torch.reshape(train_img_data[seq_test_index[k]][0][0],(1,1,28,28)))
    #                 temp_dist = LossFunction(plot1,plot2)
    #                 in_class_dist[j].append(temp_dist)
    #     else:
    #         for i in range(class_seq_test_index[j-1]+1,class_seq_test_index[j]+1):
    #             #calculate barycenter
    #             plot1 = net(torch.reshape(train_img_data[seq_test_index[i]][0][0], (1, 1, 28, 28)))
    #             if (i == class_seq_test_index[j-1]+1):
    #                 class_weight_berycenter.append(plot1)
    #             else:
    #                 class_weight_berycenter[j] = class_weight_berycenter[j] + plot1
    #             #calculate the distance in class
    #             for k in range(i+1,class_seq_test_index[j]+1):
    #                 plot2 = net(torch.reshape(train_img_data[seq_test_index[k]][0][0],(1,1,28,28)))
    #                 temp_dist = LossFunction(plot1,plot2)
    #                 in_class_dist[j].append(temp_dist)
    #
    # # Calculate the barycenter of each class
    # for j in range(len(train_img_data.classes)):
    #     if(j == 0):
    #         class_weight_berycenter[j] = class_weight_berycenter[j] / (class_seq_test_index[j]+1)
    #     else:
    #         class_weight_berycenter[j] = class_weight_berycenter[j] / (class_seq_test_index[j] - class_seq_test_index[j-1])
    #
    # # Calculate the distance between class
    # for j in range(len(train_img_data.classes)):
    #     if(j == 0):
    #         for i in range(class_seq_test_index[j] + 1):
    #             plot1 = net(torch.reshape(train_img_data[seq_test_index[i]][0][0], (1, 1, 28, 28)))
    #             for k in range(len(train_img_data.classes)):
    #                 if(k != j):
    #                     plot2 = class_weight_berycenter[k]
    #                     temp_dist = LossFunction(plot1,plot2)
    #                     between_class_dist[j][k] = between_class_dist[j][k] + temp_dist
    #     else:
    #         for i in range(class_seq_test_index[j-1]+1,class_seq_test_index[j]+1):
    #             plot1 = net(torch.reshape(train_img_data[seq_test_index[i]][0][0], (1, 1, 28, 28)))
    #             for k in range(len(train_img_data.classes)):
    #                 if(k != j):
    #                     plot2 = class_weight_berycenter[k]
    #                     temp_dist = LossFunction(plot1,plot2)
    #                     between_class_dist[j][k] = between_class_dist[j][k] + temp_dist
    # for i in range(len(train_img_data.classes)):
    #     for j in range(len(train_img_data.classes)):
    #         if (i == 0):
    #             between_class_dist[i][j] = between_class_dist[i][j] / (class_seq_test_index[i]+1)
    #         if(i != 0):
    #             between_class_dist[i][j] = between_class_dist[i][j] / (class_seq_test_index[i] - class_seq_test_index[i-1])
    # Test accuracy: static threshold
    # print_times = 0
    # for j in shuffle_test_index:
    #     print_times = print_times + 1
    #     test = []
    #     test.append(test_img_data[j][0])
    #     test.append(test_img_data[j][1])
    #     test_out = net(torch.reshape(test[0][0],(1,1,28,28)))
    #     #record current test class number and weight i.e. its NumberWeight
    #     temp_class = NumberWeight()
    #     if(len(class_weight_berycenter) == 0):
    #         class_count = class_count + 1
    #         temp_class.identity = test[1]
    #         temp_class.weight = test_out
    #         temp_class.count = 1
    #         class_weight_berycenter.append(temp_class)
    #         correct_class_num = correct_class_num + 1
    #     else:
    #         distance = []
    #         for i in class_weight_berycenter:
    #             temp_distance = LossFunction(i.weight, test_out)
    #             distance.append(temp_distance)
    #         min_distance = min(distance)
    #         #The image input's feature vector min distance is greater than the threahold: the image belongs to a new class
    #         if(min_distance > threshold):
    #             class_count = class_count + 1
    #             temp_class.identity = test[1]
    #             temp_class.weight = test_out
    #             temp_class.count = 1
    #             class_weight_berycenter.append(temp_class)
    #             correct_class_num = correct_class_num + 1
    #         else:
    #             min_index = distance.index(min_distance)
    #             class_weight_berycenter[min_index].weight = class_weight_berycenter[min_index].weight * class_weight_berycenter[min_index].count + test_out
    #             class_weight_berycenter[min_index].count = class_weight_berycenter[min_index].count + 1
    #             class_weight_berycenter[min_index].weight = class_weight_berycenter[min_index].weight / class_weight_berycenter[min_index].count
    #             if(class_weight_berycenter[min_index].identity == test[1]):
    #                 correct_class_num = correct_class_num + 1
    #
    #     if(print_times % 1000 == 0):
    #         print(class_count)
    # print("accuracy = ",correct_class_num/10000 * 100)








