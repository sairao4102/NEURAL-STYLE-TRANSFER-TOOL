"""This code is used for Neural Style Transfe
Extracts feature maps from specific layers in a pre-trained VGG model
Computes the Gram matrix from a feature map to capture the image's style"""

#import necessary libraries
import torch
import torch.nn as nn
import torchvision.models as models

# Extracts feature maps from specific layers in a pre-trained VGG model
def get_features(image, model, layers=None):
    if layers is None:
        # Default layers to extract (content and style layers)
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content representation
            '28': 'conv5_1'
        }

    features = {}
    x = image
    # Forward pass through the model and store outputs from specified layers
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Computes the Gram matrix from a feature map to capture the image's style
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w) 
    gram = torch.mm(tensor, tensor.t())  
    return gram
