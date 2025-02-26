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
from sklearn import decomposition
import copy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
MobileNetV3_Small = train.MobileNetV3_Small(2048)
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




if __name__ == '__main__':
    #Parameters:
    #data path
    test_data_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/MNIST/data/mnist_test'
    # test_smalldata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/test_small'
    test_smalldata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/query_medium_modified'
    test_mediumdata_path = '/Users/jason/IdeaProjects/PeopleFlowDetection/PedestrianFlow/data/test_real_copy'
    #whether save model parameters and whether load moder
    net_load_flag = 1
    #different class threshold
    threshold = 0.4
    #whether test or not
    test_index = 1
    #weight path
    weight_index = 52
    pthfile = 'weights/weight_mobile_2048_' + str(weight_index) + '.pth'
    #train image data class
    total_image_class = 15

    #load train and test data
    test_img_data = torchvision.datasets.ImageFolder(test_smalldata_path,transform=transform)
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
    seq_test_index.sort()



    # SECTION:Visualization
    #
    #
    test_data_featurespace = []
    for i in range(len(seq_test_index)):
        temp_feature = net(torch.reshape(test_img_data[seq_test_index[i]][0],(1,3,128,64))).squeeze(0)
        # temp_feature = F.normalize(temp_feature, p=2, dim=0)
        # original 28*28 to 2D
        # temp_feature = torch.reshape(test_img_data[seq_test_index[i]][0],(-1,))
        temp_feature = temp_feature.detach().numpy()
        test_data_featurespace.append(temp_feature)

    test_data_featurespace_np = np.stack(test_data_featurespace)

    color_num = 11
    colors = sns.color_palette("tab10", color_num)
    TSNE_dim = 2
    tsne = TSNE(n_components=TSNE_dim, init='pca', random_state=501)
    X_pca = decomposition.TruncatedSVD(n_components=200).fit_transform(test_data_featurespace_np)
    res = tsne.fit_transform(X_pca)
    color_flag = [0 for _ in range(color_num)]
    # X_tsne = tsne.fit_transform(test_data_featurespace)
    if(TSNE_dim == 2):
        for i in range(len(res)):
            for j in range(color_num):
                if(test_img_data[seq_test_index[i]][1]==j):
                    if(color_flag[j] == 0):
                        plt.scatter(res[i, 0], res[i, 1], c=colors[j], label = f'Class {j}', alpha=0.6)
                        color_flag[j] = 1
                    else:
                        plt.scatter(res[i, 0], res[i, 1], c=colors[j], alpha=0.6)
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.title('t-SNE Visualization of 16D Data in 2D Original MNIST ')

    if(TSNE_dim == 3):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(res)):
            for j in range(10):
                if(test_img_data[seq_test_index[i]][1]==j):
                    if(color_flag[j] == 0):
                        ax.scatter(res[i, 0], res[i, 1], c=colors[j], label = f'Class {j}', alpha=0.6)
                        color_flag[j] = 1
                    else:
                        ax.scatter(res[i, 0], res[i, 1], c=colors[j], alpha=0.6)

        ax.set_xlabel('TSNE Component 1')
        ax.set_ylabel('TSNE Component 2')
        ax.set_zlabel('TSNE Component 3')
        ax.set_title('t-SNE Visualization of 16D Data in 3D')

    plt.legend()
    plt.show()







