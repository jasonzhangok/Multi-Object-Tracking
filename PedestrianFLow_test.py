import torch
import torchvision
from torchvision import transforms
import random
import train


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
MobileNetV3_Small = train.MobileNetV3_Small(512)
LossFunction = train.LossFunction



transform = transforms.Compose([
    transforms.Resize((128, 64)),  # Resize images to 128x64 (Market-1501 format)
    # transforms.RandomHorizontalFlip(),  # Apply random flipping to augment data
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize between -1 and 1
])


class NumberWeight():
    def __init__(self):
        self.identity = -1
        self.feature_vector = torch.empty([2048])
        self.count = 0
        self.weight = torch.empty([2048])

class PersonImage():
    def __init__(self):
        self.count = 0
        self.featuremap = []


class ImgEmbedding():
    def __init__(self):
        self.cate = -1 #cate 表示预测的类别
        self.embedding = torch.empty([2048])
        self.label = -1 #label 表示真实的类别

#表示划分的类别的数量和是哪一个类别
class ImgClass():
    def __init__(self):
        self.classindex = -1
        self.count = 0

class MyDistance():
    def __init__(self):
        self.distance = -1
        self.identity = -1

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
    # test_smalldata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/test_small'
    test_smalldata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/query_medium_modified'
    test_mediumdata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/query_small'
    test_reality_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/test_real_copy'
    #whether save model parameters and whether load moder
    net_load_flag = 1
    #different class threshold
    threshold = 0.2
    #whether test or not
    test_index = 1
    #weight path
    weight_index = 49
    pthfile = 'weights/weight_mobile_512_' + str(weight_index) + '.pth'
    #train image data class
    total_image_class = 15

    #load train and test data
    test_img_data = torchvision.datasets.ImageFolder(test_mediumdata_path,transform=transform)
    #test_data = torch.utils.data.DataLoader(test_img_data, batch_size=1,shuffle=True, num_workers=4,drop_last=True)
    #calculate the begin index of each class



    #test
    print("Testing:")
    #enter net testing module
    net = MobileNetV3_Small
    temp = torch.load(pthfile, map_location=DEVICE)
    net.load_state_dict(temp)
    net.eval()
    correct_class_num = 0
    #ATTEN: There is not need for us to classify whether the test image belongs to the right class as input We just need to decide how many class we have and when input a new image, we classify it to the correct class
    class_num= 0

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
    seq_test_index.sort()

    # class_weight_berycenter: store image tags(class) and the output of the net with the image input to calculate the barycenter
    # class_weight_berycenter = []
    #
    # SECTION:Distance Calculation:
    #EXPLANA:
    # 1. The distance between this point and all the other 8000-1 = 7999 points
    # 2. The distance between this point and all class barycenters(10)
    # 3. The distance between all the class barycenters
    # 4. The distance between this point and all its same class points and the distance between this points and all its different class points
    # 5. The mean distance  and standard deviation in class
    # 6. The max 2 distance between class and min 2 distance in class
    # 7. The mean distance and standard deviation in same class between points and barycenter
    #Map images to 16-D space
    # featurespace = []
    # for i in seq_test_index:
    #     tempdata = test_img_data[i][0]
    #     test_out = net(torch.reshape(tempdata, (1, 3, 128, 64)))
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
    #             dist = LossFunction(featurespace[i].feature_vector.squeeze(0), featurespace[j].feature_vector.squeeze(0))
    #         else:
    #             dist = torch.tensor(0.0)
    #         temp_dist_list.append(dist)
    #     point_distance.append(temp_dist_list)
    # point_distance = tuple(point_distance)
    #
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
    # #4. The distance between this point and all its same class points and the distance between this points and all its different class points
    # in_class_distance = []
    # each_class_number = int(len(test_img_data)/total_image_class)
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
    # #One matric: The number or percentage of between class distance smaller than the max in class distance
    # print(f"Total test data scale:{each_class_number * (total_image_class - 1) * each_class_number}")
    # total_smaller_number = [0 for _ in range(total_image_class)]
    # for i in range(total_image_class):
    #     combined = torch.cat([t.flatten() for t in in_class_distance[i]])
    #     topk_values, topk_indices = torch.topk(combined, 100)
    #     topk_values = topk_values.tolist()
    #     topk_values = topk_values[0:100:2]
    #     # count how many between class distance are smaller in class distance
    #     cur_class_num = [0 for _ in range(50)]
    #     for k in range(len(topk_values)):
    #         for j in range(i*each_class_number,(i+1)*each_class_number):
    #             for m in range((total_image_class-1) * each_class_number):
    #                 if(between_class_distance[j][m] < topk_values[k]):
    #                     cur_class_num[k] += 1
    #     print(f"Class {i}: Larger count = {cur_class_num}")
    #     total_smaller_number[i] = sum(cur_class_num)
    # print(f"Total smaller number: {total_smaller_number}")
    # print(1)





    # # # 5. The mean distance and standard deviation in class
    # # class_mean = []
    # # class_deviation = []
    # # for i in range(total_image_class):
    # #     temp_mean,temp_deviation = MeanAndDeviation(in_class_distance[i],each_class_number)
    # #     class_mean.append(temp_mean)
    # #     class_deviation.append(temp_deviation)
    # # # 6. The distance sequence in class(downwards) and the distance sequence between class(upwards)
    # # in_class_distance_seq = []
    # # for i in range(total_image_class):
    # #     in_class_distance_seq.append(InClassDistanceDownwards(in_class_distance[i],each_class_number))
    # # between_class_distance_seq = []
    # #
    # # for i in seq_test_index:
    # #     between_class_distance_seq.append(BetweenClassDistanceUpwards(between_class_distance[i]))
    # # # 7. The mean distance and standard deviation in same class between points and barycenter
    # # point_to_barycenter_mean = []
    # # point_to_barycenter_deviation = []
    # # for i in range(total_image_class):
    # #     temp_dist = torch.tensor(0.0)
    # #     for j in range(i*each_class_number,(i+1)*each_class_number):
    # #         temp_dist = temp_dist + LossFunction(class_weight_berycenter[i],featurespace[j].feature_vector)
    # #     temp_dist = temp_dist / each_class_number
    # #     point_to_barycenter_mean.append(temp_dist)
    # #
    # # for i in range(total_image_class):
    # #     temp_dist = torch.tensor(0.0)
    # #     for j in range(i*each_class_number,(i+1)*each_class_number):
    # #         temp_dist = temp_dist + (LossFunction(class_weight_berycenter[i],featurespace[j].feature_vector) - point_to_barycenter_mean[i])**2
    # #     temp_dist = temp_dist / (each_class_number-1)
    # #     temp_dist = torch.sqrt(temp_dist)
    # #     point_to_barycenter_deviation.append(temp_dist)
    # #
    # # print("DATA PROCESSING END")
    #




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
    # EXPLANA: Cluster using the barycenter of each class
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
    # print("accuracy = %.2f%%"%(correct_class_num / 10000 * 100))
    # for i in range(len(class_weight_berycenter)):
    #     print('Number:',class_weight_berycenter[i].identity,'   times:',class_weight_berycenter[i].count)


    # SECTION:DB-SCAN
    #Record all the embeddings of test images
    # count_threshold = 2
    # imgs_embeddings = []
    # imgs_classes = []
    # print_times = 0
    # #maplist list record the map relation between image cate and label, cate is the label calculated by our algorithm and the label is the ground-truth label
    # maplist = []
    # for j in shuffle_test_index:
    #     print_times = print_times + 1
    #     # test = []
    #     # test.append(test_img_data[j][0])
    #     # test.append(test_img_data[j][1])
    #
    #     temp_out = net(torch.reshape(test_img_data[j][0],(1,3,128,64)))
    #     img_embedding = ImgEmbedding()
    #     img_embedding.label = test_img_data[j][1]
    #     img_embedding.embedding = temp_out
    #     # test_out = F.normalize(test_out, p=2, dim=1)
    #     imgs_embeddings.append(img_embedding)
    #
    #     #Clustering
    #     if(class_num == 0):
    #         class_num = 1
    #         img_embedding.cate = 0
    #         maplist.append(img_embedding.label)
    #         temp_class = ImgClass()
    #         temp_class.classindex = img_embedding.cate
    #         temp_class.count = 1
    #         imgs_classes.append(temp_class)
    #         correct_class_num = correct_class_num + 1
    #         continue
    #     distance = []
    #     for i in imgs_embeddings:
    #         temp_distance = MyDistance()
    #         temp_dist = LossFunction(i.embedding,temp_out)
    #         temp_distance.distance = temp_dist
    #         temp_distance.identity = i.cate
    #         distance.append(temp_distance)
    #     distance.sort(key=lambda x:x.distance)
    #     #Judge whether the new embedding belongs to a new class
    #     if(distance[1].distance > threshold):
    #         class_num = class_num + 1
    #         temp_class = ImgClass()
    #         img_embedding.cate = class_num - 1
    #         maplist.append(img_embedding.label)
    #         temp_class.classindex = img_embedding.cate
    #         temp_class.count = 1
    #         imgs_classes.append(temp_class)
    #         correct_class_num = correct_class_num + 1
    #     elif(min(imgs_classes,key = lambda x:x.count).count < count_threshold):#TODO：10 there need to be changed
    #         closest_class = distance[1].identity
    #         imgs_classes[closest_class].count = imgs_classes[closest_class].count + 1
    #         img_embedding.cate = closest_class
    #     else:
    #         temp_distance_count = [0 for _ in  range(len(imgs_classes))]
    #         for i in distance[1:]:
    #             temp_distance_count[i.identity] = temp_distance_count[i.identity] + 1
    #             if(temp_distance_count[i.identity] >= count_threshold):
    #                 break
    #         closest_class = temp_distance_count.index(count_threshold)
    #         imgs_classes[closest_class].count = imgs_classes[closest_class].count + 1
    #         img_embedding.cate = closest_class
    #         if(img_embedding.label == maplist[img_embedding.cate]):
    #             correct_class_num = correct_class_num + 1
    #
    # print('Clusterd class num:',class_num)
    # print("accuracy = %.2f%%"%(correct_class_num / len(shuffle_test_index) * 100))
    # for i in range(len(maplist)):
    #     print('class:',maplist[i],'   times:',imgs_classes[i].count)