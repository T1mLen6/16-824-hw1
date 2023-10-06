from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random 
import torch 
import torchvision.models as models
from train_q2 import *
from sklearn.preprocessing import LabelEncoder
import numpy as np
import utils
from voc_dataset import VOCDataset
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# if __name__ =="__main__": 
#     np.random.seed(16824)

#     resnetCkpt = './checkpoint-model-epoch50.pth'

#     device = torch.device("cuda")

#     resnet = torch.load(resnetCkpt)
    



if __name__ =="__main__": 

    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=25, split='test', inp_size=224)
    
    CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                   'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    

    # Specify the number of images you want to select (1000 in this case)
    num_images_to_select = 1000

    # Randomly select 1000 indices

    selected_indices = np.random.choice(4950, size=num_images_to_select, replace=False)
    print(len(selected_indices))

    # Create a new DataLoader containing the selected 1000 images
    selected_images_loader = torch.utils.data.DataLoader(
                                                        dataset=test_loader.dataset,  # Use the same dataset
                                                        batch_size=test_loader.batch_size,
                                                        sampler=torch.utils.data.SubsetRandomSampler(selected_indices)
                                                        )

    #model = resnet18(pretrained=False)
    model = torch.load('./checkpoint-model-epoch50.pth')

    #model.load_state_dict(checkpoint['model_state_dict'])
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    # Extract features from the test set
    features = []
    targets = []
    i = 0

    for images, target, _ in selected_images_loader:
        i = i+1
        with torch.no_grad():
            features_batch = model(images)
        features.append(features_batch)
        targets.append(target)

    print(i)
    targets1 = torch.cat(targets, dim=0)

    class_labels = []  # Initialize an empty list for class labels

    for target in targets:
        target_np = target.cpu().numpy()  # Convert the target tensor to a NumPy array
        class_indices = np.where(target_np == 1)[0].tolist()  # Get the indices where the value is 1
        class_labels.append(class_indices)
        
    features = torch.cat(features, dim=0)
    # Reshape the features tensor to 2D: (num_samples, num_features)
    num_samples, num_channels, height, width = features.shape
    features_2d = features.view(num_samples, -1)

    # Initialize t-SNE with 2D output dimension
    tsne = TSNE(n_components=2, perplexity=70, early_exaggeration=20.0, learning_rate='auto', n_iter=10000)

    # Compute t-SNE projection
    tsne_features = tsne.fit_transform(features_2d)

    # Create a scatter plot with color-coded points based on class labels
    plt.figure(figsize=(10, 8))

    # Define a colormap for classes
    #colormap = plt.cm.get_cmap('tab20', len(CLASS_NAMES))


    colormap = np.array([(0, 0, 255),     # Blue
                (255, 0, 0),     # Red
                (0, 255, 0),     # Green
                (255, 255, 0),   # Yellow
                (255, 0, 255),   # Magenta
                (0, 255, 255),   # Cyan
                (128, 0, 0),     # Maroon
                (0, 128, 0),     # Green (Dark)
                (0, 0, 128),     # Navy
                (128, 128, 0),   # Olive
                (128, 0, 128),   # Purple
                (0, 128, 128),   # Teal
                (128, 64, 0),    # Brown
                (64, 0, 128),    # Indigo
                (128, 0, 64),    # Crimson
                (192, 192, 192),  # Silver
                (128, 128, 128), # Gray
                (255, 165, 0),   # Orange
                (255, 192, 203), # Pink
                (0, 0, 0)        # Black
            ])/255

    # Flatten the list of class labels
    flat_class_labels = [label for sublist in class_labels for label in sublist]

    # Create a list of unique class labels
    unique_class_labels = []

    # Create a list of unique class labels
    unique_class_labels = list(set(flat_class_labels))


    color_per_image = []
    
    color_per_image = np.matmul(targets1.detach().numpy(), colormap) / np.sum(targets1.detach().numpy(), axis=1).reshape(-1, 1)

    legend_patches = []

    for i in range(len(CLASS_NAMES)):

        # Create a legend patch for the current class
        legend_patch = plt.Line2D([0],[0],marker='o',color=colormap[i],label=CLASS_NAMES[i])
        legend_patches.append(legend_patch)

    # Plot t-SNE points
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=color_per_image)

    # Add a legend
    plt.legend(handles=legend_patches)

    # Show the plot
    plt.title('t-SNE Projection of ImageNet Features')
    plt.show()
    #feature_colors = np.matmul(targets.numpy(), 
                              # colormap) / np.sum(targets.numpy(), axis=1).reshape(-1, 1)