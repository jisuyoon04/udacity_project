# TODO:
#   Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

#   Basic usage: python predict.py /path/to/image checkpoint
#    Options:
#       Return top KK most likely classes: python predict.py input checkpoint --top_k 3
#       Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#       Use GPU for inference: python predict.py input checkpoint --gpu

import argparse
import json
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from helper_data import load_data, load_checkpoint, process_image
from helper_neuralnet import customize_classifier, train_model, save_model, predict_model

parser = argparse.ArgumentParser(description='Make prediction on image with trained neural network')

parser.add_argument('path_image', action='store', type=str,
                    help='Enter path to image file')
parser.add_argument('--save_dir', action='store', default='checkpoint.pth', 
                    help='Enter directory to save checkpoint')
parser.add_argument('--arch', action='store', type=str, default='vgg11', 
                    help='Enter a pretrained model. The default architecture is vgg11')
parser.add_argument('--learning_rate', action='store', type=int, default=0.001, 
                    help='Enter learning rate for training the neural network. The default learning rate is 0.001.')
parser.add_argument('--top_k', action='store', type=int, default=5, 
                    help='Enter number of top k most likely classes. The default number is 5')
parser.add_argument('--category_names', action='store', type=str, default='cat_to_name.json', 
                    help='Enter the name of category file. The default epochs is cat_to_name.json.')
parser.add_argument('--gpu', action='store_true', default=False, 
                    help='Activate GPU mode or deactivate. Type --gpu to activate. The default is false.')

args = parser.parse_args()

save_dir = args.save_dir
arch = args.arch
image = args.path_image
top_k = args.top_k
gpu_mode = args.gpu
category_names = args.category_names

# Load category file
with open(category_names, 'r') as f:
    category_names = json.load(f)

# Load pre-trained model
model = getattr(models, arch)(pretrained=True)

# Load checkpoint
model = load_checkpoint(model, save_dir, gpu_mode)

# Image preprocessing
img = process_image(image)

# Make prediction
ps_numpy, classes = predict_model(img, model, gpu_mode, topk=top_k)
names = [category_names[i] for i in classes]

# Print probabilities and predicted classes
print(f"The flower named '{names[0]}' is most likely with a probability of {round(ps_numpy[0]*100,4)} %")
print(f"The probabilities of top '{top_k}': '{ps_numpy}'")
print(f"The predicted classes of top '{top_k}': '{classes}'")