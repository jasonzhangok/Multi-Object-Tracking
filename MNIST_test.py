import copy

import sklearn.manifold
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import random
from tqdm import tqdm
import train
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import copy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MyNet = train.MyNet
LossFunction = train.LossFunction
contrastive_loss = train.contrastive_loss
triple_loss = train.triplet_loss



transform = train.transforms.Compose([transforms.ToTensor()])

class NumberWeight():
    def __init__(self):
        self.identity = -1
        self.feature_vector = torch.empty([16])
        self.count = 0
        self.weight = torch.empty([16])




def MeanAndDeviation(points,number):
    points_2d = []
    length = len(points)
    #reshape:
    for i in range(number):
        temp_points = []
        for j in range(number):
            temp_points.append(points[i * number +j])
        points_2d.append(temp_points)
    #delete 0 and repeat number
    ture_points = []
    for i in range(number):
        for j in range(number):
            if(j > i):
                ture_points.append(points_2d[i][j])
    length = len(ture_points)
    mean = torch.tensor(0.0)
    for i in range(length):
        mean = mean + ture_points[i]
    mean = mean / length
    deviation = torch.tensor(0.0)
    for i in range(length):
        deviation = deviation + (ture_points[i] - mean) ** 2
    deviation = deviation / (length-1)
    deviation = torch.sqrt(deviation)
    return mean, deviation

def InClassDistanceDownwards(points,number):
    points_2d = []
    length = len(points)
    # reshape:
    for i in range(number):
        temp_points = []
        for j in range(number):
            temp_points.append(points[i * number + j])
        points_2d.append(temp_points)
    # delete 0 and repeat number
    ture_points = []
    for i in range(number):
        for j in range(number):
            if (j > i):
                ture_points.append(points_2d[i][j])
    ture_points.sort(reverse=True)
    return ture_points

def BetweenClassDistanceUpwards(points):
    res = points
    res.sort()
    return res


if __name__ == '__main__':
    #Parameters:
    #data path
    test_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_test'
    test_smalldata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_test_small'
    test_mediumdata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_test_medium'
    #learning rate
    initial_lr = 0.001
    #whether save model parameters and whether load moder
    net_load_flag = 1
    #different class threshold
    threshold = 0.68
    #whether test or not
    test_index = 1
    #weight path
    weight_index = 9
    pthfile = 'weights/weightCOS' + str(weight_index) + '.pth'
    #train image data class
    total_image_class = 10

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
    #ATTEN: There is not need for us to classify whether the test image belongs to the right class as input We just need to decide how many class we have and when input a new image, we classify it to the correct class
    class_count = 0

    # SECTION:Organize and analysis in_class_dist and between_class_dist
    # Calculate distance There are some problem
    # in_class_dist.sort(reverse=True)
    # between_class_dist.sort()
    # threshold_sup = between_class_dist[10000]
    # threshold_inf = in_class_dist[10000]
    # threshold = threshold_sup + threshold_inf


    # shuffle test dataset index
    shuffle_test_index = [i for i in range(len(test_img_data))]
    random.shuffle(shuffle_test_index)
    # sequence test dataset index
    seq_test_index = shuffle_test_index[:int(len(test_img_data))]
    # seq_test_index.sort()

    # class_weight_berycenter: store image tags(class) and the output of the net with the image input to calculate the barycenter
    class_weight_berycenter = []

    featurespace = []

    # # SECTION:Distance Calculation:
    # #TODO:
    # # 1. The distance between this point and all the other 8000-1 = 7999 points
    # # 2. The distance between this point and all class barycenters(10)
    # # 3. The distance between all the class barycenters
    # # 4. The distance between this point and all its same class points and the distance between this points and all its different class points
    # # 5. The mean distance  and standard deviation in class
    # # 6. The max 2 distance between class and min 2 distance in class
    # # 7. The mean distance and standard deviation in same class between points and barycenter
    # #Map images to 16-D space
    # for i in seq_test_index:
    #     tempdata = test_img_data[i][0]
    #     test_out = net(torch.reshape(tempdata[0], (1, 1, 28, 28)))
    #     feature = NumberWeight()
    #     feature.feature_vector = test_out
    #     feature.identity = test_img_data[i][1]
    #     #add the feature vector of the test data into feature space
    #     featurespace.append(feature)
    # #1.The distance between this point and all the other 8000-1 = 7999 points
    # point_distance = []
    # for i in seq_test_index:
    #     temp_dist_list = []
    #     for j in seq_test_index:
    #         if( i != j):
    #             dist = LossFunction(featurespace[i].feature_vector, featurespace[j].feature_vector)
    #         else:
    #             dist = torch.tensor(0.0)
    #         temp_dist_list.append(dist)
    #     point_distance.append(temp_dist_list)
    # # point_distance = tuple(point_distance)
    #
    #
    # #2. The distance between this point and all class barycenters(10)
    # # 1)calculate the barycenter:
    # for i in seq_test_index:
    #     if(i != seq_test_index[-1]):
    #         if(featurespace[i].identity != featurespace[i+1].identity):
    #             class_weight_berycenter.append(torch.tensor(0.0))
    # class_weight_berycenter.append(torch.tensor(0.0))
    #
    # class_time_count = [0 for i in range(total_image_class)]
    # for i in seq_test_index:
    #     class_weight_berycenter[featurespace[i].identity] = class_weight_berycenter[featurespace[i].identity] + featurespace[i].feature_vector
    #     class_time_count[featurespace[i].identity] = class_time_count[featurespace[i].identity] + 1
    # for i in range(total_image_class):
    #     class_weight_berycenter[i] = class_weight_berycenter[i] / class_time_count[i]
    #
    # # 3. The distance between all the class barycenters
    # barycenter_distance = []
    # for i in range(total_image_class):
    #     temp_dist_list = []
    #     for j in range(total_image_class):
    #         if( i != j):
    #             dist = LossFunction(class_weight_berycenter[i],class_weight_berycenter[j])
    #         else:
    #             dist = torch.tensor(0.0)
    #         temp_dist_list.append(dist)
    #     barycenter_distance.append(temp_dist_list)
    # # 4. The distance between this point and all its same class points and the distance between this points and all its different class points
    # in_class_distance = []
    # each_class_number = int(len(test_img_data)/10)
    # for i in range(total_image_class):
    #     temp_dist_list = []
    #     for j in range(each_class_number):
    #         temp_dist_list.extend(point_distance[i*each_class_number+j][i*each_class_number:(i+1)*each_class_number])
    #     in_class_distance.append(temp_dist_list)
    # between_class_distance = []
    # for i in range(len(point_distance)):
    #     between_class_distance.append(point_distance[i])
    # # between_class_distance = copy.deepcopy(point_distance.clone())
    # for i in range(total_image_class):
    #     temp_dist_list = []
    #     for j in range(each_class_number):
    #         del between_class_distance[i*each_class_number+j][i*each_class_number:(i+1)*each_class_number]
    #
    # # 5. The mean distance and standard deviation in class
    # class_mean = []
    # class_deviation = []
    # for i in range(total_image_class):
    #     temp_mean,temp_deviation = MeanAndDeviation(in_class_distance[i],each_class_number)
    #     class_mean.append(temp_mean)
    #     class_deviation.append(temp_deviation)
    # # 6. The distance sequence in class(downwards) and the distance sequence between class(upwards)
    # in_class_distance_seq = []
    # for i in range(total_image_class):
    #     in_class_distance_seq.append(InClassDistanceDownwards(in_class_distance[i],each_class_number))
    # between_class_distance_seq = []
    #
    # for i in seq_test_index:
    #     between_class_distance_seq.append(BetweenClassDistanceUpwards(between_class_distance[i]))
    # # 7. The mean distance and standard deviation in same class between points and barycenter
    # point_to_barycenter_mean = []
    # point_to_barycenter_deviation = []
    # for i in range(total_image_class):
    #     temp_dist = torch.tensor(0.0)
    #     for j in range(i*each_class_number,(i+1)*each_class_number):
    #         temp_dist = temp_dist + LossFunction(class_weight_berycenter[i],featurespace[j].feature_vector)
    #     temp_dist = temp_dist / each_class_number
    #     point_to_barycenter_mean.append(temp_dist)
    #
    # for i in range(total_image_class):
    #     temp_dist = torch.tensor(0.0)
    #     for j in range(i*each_class_number,(i+1)*each_class_number):
    #         temp_dist = temp_dist + (LossFunction(class_weight_berycenter[i],featurespace[j].feature_vector) - point_to_barycenter_mean[i])**2
    #     temp_dist = temp_dist / (each_class_number-1)
    #     temp_dist = torch.sqrt(temp_dist)
    #     point_to_barycenter_deviation.append(temp_dist)
    #
    # print("DATA PROCESSING END")

    # SECTION:DBSCAN(Density-Based Spatial Clustering of Application with Noise)
    # for i in range(int(len(test_img_data)*0.1)):
    #     tempdata = test_img_data[i][0]
    #     test_out = net(torch.reshape(tempdata[0], (1, 1, 28, 28)))
    #     feature = NumberWeight()
    #     feature.feature_vector = test_out
    #     feature.identity = test_img_data[i][1]
    #     #add the feature vector of the test data into feature space
    #     featurespace.append(feature)




    # SECTION: Test Accuracy: dynamic threshold
    # EXPLANA: Calculate in-class distance and between-class distance
    # the index of class(from 0 to 9) in seq_test_index(We calculate this in order to calculate the in-class distance and between class distance)
    # each class index number is from class_seq_test_index[i-1] to class_seq_test_index[i] - 1
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




    # SECTION:Test accuracy: static threshold
    print_times = 0
    for j in shuffle_test_index:
        print_times = print_times + 1
        test = []
        test.append(test_img_data[j][0])
        test.append(test_img_data[j][1])
        test_out = net(torch.reshape(test[0][0],(1,1,28,28)))
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

        if(print_times % 1000 == 0):
            print(class_count)
    print("accuracy = %.2f%%"%(correct_class_num/10000 * 100))
    for i in range(len(class_weight_berycenter)):
        print('Number:',class_weight_berycenter[i].identity,'   times:',class_weight_berycenter[i].count)


    # # SECTION:Visualization
    #
    #
    # test_data_featurespace = []
    # for i in range(len(seq_test_index)):
    #     temp_feature = net(torch.reshape(test_img_data[seq_test_index[i]][0][0],(1,1,28,28)))
    #     # original 28*28 to 2D
    #     # temp_feature = torch.reshape(test_img_data[seq_test_index[i]][0][0],(-1,))
    #     temp_feature = temp_feature.detach().numpy()
    #     test_data_featurespace.append(temp_feature)
    #
    # test_data_featurespace_np = np.stack(test_data_featurespace)
    #
    # colors = sns.color_palette("tab10", 10)
    # TSNE_dim = 2
    # tsne = TSNE(n_components=TSNE_dim, init='pca', random_state=501)
    # res = tsne.fit_transform(test_data_featurespace_np)
    # color_flag = [0 for _ in range(10)]
    # # X_tsne = tsne.fit_transform(test_data_featurespace)
    # if(TSNE_dim == 2):
    #     for i in range(len(res)):
    #         for j in range(10):
    #             if(test_img_data[seq_test_index[i]][1]==j):
    #                 if(color_flag[j] == 0):
    #                     plt.scatter(res[i, 0], res[i, 1], c=colors[j], label = f'Class {j}', alpha=0.6)
    #                     color_flag[j] = 1
    #                 else:
    #                     plt.scatter(res[i, 0], res[i, 1], c=colors[j], alpha=0.6)
    #     plt.xlabel('TSNE Component 1')
    #     plt.ylabel('TSNE Component 2')
    #     plt.title('t-SNE Visualization of 16D Data in 2D Original MNIST ')
    #
    # if(TSNE_dim == 3):
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     for i in range(len(res)):
    #         for j in range(10):
    #             if(test_img_data[seq_test_index[i]][1]==j):
    #                 if(color_flag[j] == 0):
    #                     ax.scatter(res[i, 0], res[i, 1], c=colors[j], label = f'Class {j}', alpha=0.6)
    #                     color_flag[j] = 1
    #                 else:
    #                     ax.scatter(res[i, 0], res[i, 1], c=colors[j], alpha=0.6)
    #
    #     ax.set_xlabel('TSNE Component 1')
    #     ax.set_ylabel('TSNE Component 2')
    #     ax.set_zlabel('TSNE Component 3')
    #     ax.set_title('t-SNE Visualization of 16D Data in 3D')
    #
    # plt.legend()
    # plt.show()







