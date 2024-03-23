import torch.nn as nn
import torch
from torchvision import models

# TODO: to fix the following code
class CustomCNN(nn.Module):
  def __init__(self, num_classes):
    super(CustomCNN, self).__init__()
    self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
    self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
    self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
    self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
    self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
    self.relu = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, 1)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.cnn1_bn = nn.BatchNorm2d(8)
    self.cnn2_bn = nn.BatchNorm2d(16)
    self.cnn3_bn = nn.BatchNorm2d(32)
    self.cnn4_bn = nn.BatchNorm2d(64)
    self.cnn5_bn = nn.BatchNorm2d(128)
    self.cnn6_bn = nn.BatchNorm2d(256)
    self.cnn7_bn = nn.BatchNorm2d(256)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, num_classes)
    self.dropout = nn.Dropout(0.3)
    self.log_softmax = nn.LogSoftmax(dim=1)
    
  def forward(self, x):
    x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
    x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
    x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
    x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
    x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
    x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
    x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))
    
    x = x.view(x.size(0), -1)
    
    x = self.relu(self.dropout(self.fc1(x)))
    x = self.relu(self.dropout(self.fc2(x)))
    x = self.log_softmax(self.fc3(x))
    return x

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
# ------------------------------

# DATALOADER
# import cv2
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import torch.utils.data as utils
# from torchvision import transforms
# from PIL import Image


# emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
#                 4: 'sad', 5: 'surprise', 6: 'neutral'}


# def load_fer2013(path_to_fer_csv):
#     data = pd.read_csv(path_to_fer_csv)
#     pixels = data['pixels'].tolist()
#     width, height = 48, 48
#     faces = []
#     for pixel_sequence in pixels:
#         face = [int(pixel) for pixel in pixel_sequence.split(' ')]
#         face = np.asarray(face).reshape(width, height)
#         face = cv2.resize(face.astype('uint8'), (48,48))
#         faces.append(face.astype('float32'))
#     faces = np.asarray(faces)
#     faces = np.expand_dims(faces, -1)
#     emotions = data['emotion'].values
#     return faces, emotions


# def show_random_data(faces, emotions):
#   idx = np.random.randint(len(faces))
#   print(emotion_dict[emotions[idx]])
#   plt.imshow(faces[idx].reshape(48,48), cmap='gray')
#   plt.show()

# class EmotionDataset(utils.Dataset):
#     def __init__(self, X, y, transform=None):
#         self.X = X
#         self.y = y
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, index):
#         x = self.X[index].reshape(48,48)
#         x = Image.fromarray((x))
#         if self.transform is not None:
#             x = self.transform(x)
#         y = self.y[index]
#         return x, y


# def get_dataloaders(path_to_fer_csv='data/fer2013/fer2013.csv', tr_batch_sz=3000, val_batch_sz=500):
#     print('Loading data ...'	)
#     faces, emotions = load_fer2013(path_to_fer_csv)
#     print('Data loaded')
#     print('Splitting data into train and validation sets ...')
#     train_X, val_X, train_y, val_y = train_test_split(faces, emotions, test_size=0.2,
#                                                 random_state = 1, shuffle=True)
#     print('Data splitted into train and validation sets')
#     print('Data Augmentation ...')
#     train_transform = transforms.Compose([
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomRotation(30),
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.507395516207, ),(0.255128989415, )) 
#                         ])
#     print('Data Augmentation done for training set')
#     val_transform = transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.507395516207, ),(0.255128989415, ))
#                     ])  
#     print('Data Augmentation done for validation set')

#     train_dataset = EmotionDataset(train_X, train_y, train_transform)
#     val_dataset = EmotionDataset(val_X, val_y, val_transform)

#     trainloader = utils.DataLoader(train_dataset, tr_batch_sz)
#     validloader = utils.DataLoader(val_dataset, val_batch_sz)

#     return trainloader, validloader

# TRAIN
# import torch
# import torch.nn as nn
# from model import *
# from dataloader import *
# import matplotlib.pyplot as plt
# import datetime
# from tqdm import tqdm
# import os
# import json

# # Parameters
# path_to_fer_csv = 'data/fer2013/fer2013.csv'
# tr_batch_sz = 1024
# val_batch_sz = 128

# model_choosen = Face_Emotion_CNN() # Default: Face_Emotion_CNN()
# criterion_loss = nn.CrossEntropyLoss() # Default: nn.NLLLoss()
# epochs = 50
# lr = 1e-5
# weight_decay = 1e-5
# visualize_learning_curve = True 

# # Create directory to save models and plots with os module
# now = datetime.datetime.now()
# now = now.strftime("%Y-%m-%d_%H-%M-%S")
# os.makedirs('checkpoints/'+now, exist_ok=True)

# os.makedirs('checkpoints/'+now+'/models', exist_ok=True)
# os.makedirs('checkpoints/'+now+'/plots', exist_ok=True)

# # Create config file with parameters and save it as .json
# config = {
#     'tr_batch_sz': tr_batch_sz,
#     'val_batch_sz': val_batch_sz,
#     'criterion_loss': str(criterion_loss),
#     'epochs': epochs,
#     'lr': lr,
#     'weight_decay': weight_decay,
# } 
# with open('checkpoints/'+now+'/config.json', 'w') as f:
#     json.dump(config, f)

# def train_model(model, trainloader, validloader, epochs, visualize_learning_curve):
#     criterion = criterion_loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     valid_loss_min = np.Inf
#     train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []
#     for e in range(epochs):
#         model.train()
#         running_loss = 0
#         tr_accuracy = 0
#         for images, labels in tqdm(trainloader):
#             images = images.cuda()
#             labels = labels.long().cuda()
#             optimizer.zero_grad()
            
#             log_ps  = model(images)
#             loss = criterion(log_ps, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
            
#             ps = torch.exp(log_ps)
#             top_p, top_class = ps.topk(1, dim=1)
#             equals = top_class == labels.view(*top_class.shape)
#             tr_accuracy += torch.mean(equals.type(torch.FloatTensor))
#         # else:
#         val_loss = 0
#         accuracy = 0
#         with torch.no_grad():
#             model.eval()
#             for images, labels in validloader:
#                 images = images.cuda()
#                 labels = labels.long().cuda()
#                 log_ps = model(images)
#                 val_loss += criterion(log_ps, labels)
                
#                 ps = torch.exp(log_ps)
#                 top_p, top_class = ps.topk(1, dim=1)
#                 equals = top_class == labels.view(*top_class.shape)
#                 accuracy += torch.mean(equals.type(torch.FloatTensor))
        
#         train_losses.append(running_loss/len(trainloader))
#         val_losses.append(val_loss/len(validloader))
#         train_accuracy.append(tr_accuracy/len(trainloader))
#         val_accuracy.append(accuracy/len(validloader))


#         print("Epoch: {}/{} ".format(e+1, epochs),
#             "Training Loss: {:.3f} ".format(train_losses[-1]),
#             "Training Acc: {:.3f} ".format(tr_accuracy/len(trainloader)),
#             "Val Loss: {:.3f} ".format(val_losses[-1]),
#             "Val Acc: {:.3f}".format(accuracy/len(validloader)))
        
#         if val_loss/len(validloader) <= valid_loss_min:
#             print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#             valid_loss_min,
#             val_loss/len(validloader)))
#             torch.save(model.state_dict(), 'checkpoints/'+now+'/models/best_model.pt')
#             valid_loss_min = val_loss/len(validloader)

#         # Save the model every epoch
#         if visualize_learning_curve and e > 1:
#             # Convert the lists into tensors
#             train_losses = [torch.tensor(x) for x in train_losses]
#             val_losses = [torch.tensor(x) for x in val_losses]

#             # Convert the tensors into numpy arrays
#             train_losses = [x.cpu().clone().detach().numpy() for x in train_losses]
#             val_losses = [x.cpu().clone().detach().numpy() for x in val_losses]

#             # Generate a plot that shows the training and validation loss over time and save it as .png
#             plt.plot(train_losses, label='Training loss')
#             plt.plot(val_losses, label='Validation loss')
#             plt.legend(frameon=False)
#             plt.savefig('checkpoints/'+now+'/plots/learning_curve.png')
#             plt.close()

#             # Generate a plot that shows the train and validation accuracy over time and save it as .png
#             plt.plot(train_accuracy, label='Training accuracy')
#             plt.plot(val_accuracy, label='Validation accuracy')
#             plt.legend(frameon=False)
#             plt.savefig('checkpoints/'+now+'/plots/accuracy_curve.png')
#             plt.close()


#     return model



# def main():
#     print ('Parameters:')
#     print ('Training batch size: ', tr_batch_sz)
#     print ('Validation batch size: ', val_batch_sz)
#     print('Criterion Loss: ', criterion_loss)
#     print ('Epochs: ', epochs)
#     print ('Learning rate: ', lr)
#     print ('Weight decay: ', weight_decay)
#     print ('Visualize Learning Curve: ', visualize_learning_curve)

#     print('\nPreprocess Data and get DataLoaders...')
#     trainloader, validloader = get_dataloaders(path_to_fer_csv, tr_batch_sz, val_batch_sz)
#     print('Data Preprocessed and got DataLoaders\n')
#     print ('Training data size: ', len(trainloader.dataset))
#     print ('Validation data size: ', len(validloader.dataset))

#     model = model_choosen
#     if torch.cuda.is_available():
#         model.cuda()
#         print('GPU Found, Moving Model to CUDA')
#     else:
#         print('GPU not found, using model with CPU')

#     print('Starting Training loop...\n')
#     model = train_model(model, trainloader, validloader, epochs, visualize_learning_curve)    
#     print('Training Done')

# if __name__ == '__main__':
#     main()