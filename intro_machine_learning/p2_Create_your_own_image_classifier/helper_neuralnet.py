import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time

# Edit the classifier by given inputs
def customize_classifier(model, num_in_feature, num_hidden, p_dropout):
    # Freeze parameters in order to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(num_in_feature, num_hidden)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=p_dropout)),
            ('fc2', nn.Linear(num_hidden, 102)),
            ('output', nn.LogSoftmax(dim=1))
            ]))

    # Replace the pretrained classifier
    model.classifier = classifier # Architecture such as ResNet has the last layer "fc".
    return model

def train_model(model, epochs, criterion, optimizer, gpu_mode,
             train_dataloaders, validation_dataloaders):
    
    device = torch.device("cuda" if gpu_mode else "cpu")
    model.to(device)
    
    steps = 0
    print_every = 5
    running_loss = 0

    losses_train, losses_validation = [], []
    for e in range(epochs):
        start = time.time()
        
        for inputs, labels in train_dataloaders:
            steps += 1
            # Move inputs and labels tensor to GPU / CPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # clear parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.forward(inputs)
            # Calculate loss
            loss_train = criterion(outputs, labels)
            # Backward pass
            loss_train.backward()
            # Update the parameters
            optimizer.step()
            
            running_loss += loss_train.item()
            
            # Validation
            if steps % print_every == 0:
                loss_validation = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    model.to(device)
                    for inputs, labels in validation_dataloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        outputs = model.forward(inputs)
                        loss_batch = criterion(outputs, labels)
                        loss_validation += loss_batch.item()
                        
                        # Calculate probability
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
            
                        # Calculate accuracy
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
                losses_train.append(running_loss/len(train_dataloaders))
                losses_validation.append(loss_validation/len(validation_dataloaders))
            
                print(f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {loss_validation/len(validation_dataloaders):.3f}.. "
                    f"Accuracy: {accuracy/len(validation_dataloaders):.3f}")
                
                running_loss = 0
                model.train()
        calc_time = time.time() - start
        print(f"Time per epoch: {calc_time} seconds")
    
    return model, optimizer, losses_train, losses_validation, accuracy

def save_model(save_dir, model, train_dataset, optimizer, epochs):
    checkpoint = {'model_state_dict': model.state_dict(),
              'model_classifier': model.classifier,
              'class_to_idx': train_dataset.class_to_idx,
              'optimizier_state_dict': optimizer.state_dict(),
              'epochs': epochs}

    return torch.save(checkpoint, save_dir)

def predict_model(image, model, gpu_mode, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if gpu_mode else "cpu")

    model = model.to(device)
    # Convert image to torch tensor
    if device.type == 'cuda':
        img = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        img = torch.from_numpy(image).type(torch.FloatTensor)

    # Adding dimension to image (B x C x W x H) input of model
    img_unsqueezed = img.unsqueeze_(0)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(img_unsqueezed)
    
    # Calculate probabilities
    ps = torch.exp(output)
    ps_topk, indices = ps.topk(topk)
    
    # Convert to cpu() due to the computational problem
    ps_topk = ps_topk.cpu()
    indices = indices.cpu()

    ps_numpy = ps_topk.numpy()[0]
    # Converting probability to list
    ps_list = np.array(ps_topk)[0]
    indices = np.array(indices)[0]
    
    # Load class
    idx_class = model.class_to_idx
    index_class = {val: key for key, val in idx_class.items()}
    classes = [index_class[i] for i in indices]
    
    return ps_numpy, classes