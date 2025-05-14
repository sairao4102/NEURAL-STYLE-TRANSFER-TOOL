"""This script contains functions to load and save images for a neural network model.
Opens an image, resizes it, converts it into a format suitable for neural networks. and
Takes a processed tensor, reverses the changes, and saves it back as a normal image."""

#import necessary libraries
from PIL import Image
import torchvision.transforms as transforms
import torch

#Opens an image, resizes it, converts it into a format suitable for neural networks.
def load_image(image_path, image_size=256):
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to exact size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Lambda(lambda x: x[:3, :, :]),  # Remove alpha channel if present
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


#Takes a processed tensor, reverses the changes, and saves it back as a normal image
def save_image(path, tensor):
    # Undo normalization
    unloader = transforms.Compose([
        transforms.Lambda(lambda x: x.squeeze(0)),  # Remove batch dimension
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1., 1., 1.]),
        transforms.ToPILImage()
    ])
    image = unloader(tensor.cpu().clone().clamp(0, 1))
    image.save(path)
