""" This script performs style transfer using a pre-trained VGG19 model.
It takes a content image and a style image, applies the style of the style image to the content image"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from utils.image_loader import load_image, save_image
from utils.model_utils import get_features, gram_matrix

def perform_style_transfer(content_path, style_path, output_path='static/generated/output.jpg',
                           steps=400, style_weight=1e5, content_weight=1e0):

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set image size (you can adjust this for faster processing)
    image_size = 512

    # Load and resize images
    content_img = load_image(content_path, image_size=image_size).to(device)
    style_img = load_image(style_path, image_size=image_size).to(device)

    # Ensure images are the same size
    assert content_img.size() == style_img.size(), "Images must be the same size"

    # Load pretrained VGG19 model
    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    # Disable gradient computation for the VGG model to save memory
    for param in vgg.parameters():
        param.requires_grad = False

    # Extract features for both content and style images
    content_features = get_features(content_img, vgg)
    style_features = get_features(style_img, vgg)

    # Compute gram matrices for style image
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize target image with content image
    target = content_img.clone().requires_grad_(True).to(device)

    # Use LBFGS optimizer for style transfer
    optimizer = optim.LBFGS([target])

    # Optimization loop
    run = [0]
    while run[0] <= steps:
        def closure():
            optimizer.zero_grad()
            target_features = get_features(target, vgg)

            # Content loss
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

            # Style loss
            style_loss = 0
            for layer in style_grams:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_loss = torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_loss

            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()

            run[0] += 1
            # Print loss every 25 steps for monitoring
            if run[0] % 25 == 0:
                print(f"Step {run[0]}, Content Loss: {content_loss.item():.2f}, Style Loss: {style_loss.item():.2f}, Total Loss: {total_loss.item():.2f}")

            return total_loss

        optimizer.step(closure)

    # Save the output image
    save_image(output_path, target)

    return output_path
