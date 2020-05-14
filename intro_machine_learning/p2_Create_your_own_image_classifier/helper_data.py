import numpy as np
from PIL import Image
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    validation_trasforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    validation_image_datasets = datasets.ImageFolder(valid_dir, transform=validation_trasforms)

    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, 
                                                    batch_size= 32, shuffle= True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, 
                                                batch_size = 32, shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_image_datasets, 
                                                        batch_size= 32, shuffle=True)

    return train_dataloaders, test_dataloaders, validation_dataloaders, train_image_datasets, test_image_datasets, validation_image_datasets

def load_checkpoint(model, filepath, gpu_mode):
    checkpoint = torch.load(filepath)
    
    model.load_state_dict = checkpoint['model_state_dict']
    model.classifier = checkpoint['model_classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img_pil = Image.open(image_path)
    
    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    pil_transformed = transform(img_pil)
    np_array_img = np.array(pil_transformed)
    
    return np_array_img