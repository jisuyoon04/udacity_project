# TODO:
#   Train a new network on a data set with train.py

#   Basic usage: python train.py data_directory
#   Prints out training loss, validation loss, and validation accuracy as the network trains
#   Options:
#       Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#       Choose architecture: python train.py data_dir --arch "vgg13"
#       Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units --epochs 20
#       Use GPU for training: python train.py data_dir --gpu

import argparse
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from helper_data import load_data
from helper_neuralnet import customize_classifier, train_model, save_model

parser = argparse.ArgumentParser(description='Train Neural Network of Image Classifier')

parser.add_argument('data_dir', action='store', type=str,
                    help='Enter data directory')
parser.add_argument('--save_dir', action='store', default='checkpoint.pth', 
                    help='Enter directory to save checkpoint')
parser.add_argument('--arch', action='store', default='vgg11', 
                    help='Enter a pretrained model. The default architecture is vgg11')
parser.add_argument('--learning_rate', action='store', type=float, default=0.001, 
                    help='Ente learning rate for training the neural network. The default learning rate is 0.001.')
parser.add_argument('--hidden_units', action='store', type=int, default=512, 
                    help='Enter number of hidden units in classifier. The default number is 512')
parser.add_argument('--epochs', action='store', type=int, default=5, 
                    help='Enter number of epochs for training the neural network. The default epochs is 5.')
parser.add_argument('--p_dropout', action='store', type=int, default=0.2, 
                    help='Enter the probability of dropout for training the neural network. The default dropout rate is 0.2.')
parser.add_argument('--gpu', action='store_true', default=False, 
                    help='Activate GPU mode or deactivate. Type --gpu to activate. The default is false.')

args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
p_dropout = args.p_dropout
num_hidden = args.hidden_units
epochs = args.epochs
gpu_mode = args.gpu

# Load data
train_dataloaders, test_dataloaders, validation_dataloaders, train_image_datasets, test_image_datasets, validation_image_datasets = load_data(data_dir)

# Load pre-trained model
model = getattr(models, arch)(pretrained=True)

# Update the classifier
num_in_feature = model.classifier[0].in_features
model = customize_classifier(model, num_in_feature, num_hidden, p_dropout)

# Define criterion, optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# Train model
model, optimizer, losses_train, losses_validation, accuracy = train_model(model, epochs, criterion, optimizer, gpu_mode, train_dataloaders, validation_dataloaders)

# Save trained model
save_model(save_dir, model, train_image_datasets, optimizer, epochs)