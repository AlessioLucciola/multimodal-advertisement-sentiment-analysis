import torch
import torch.nn as nn
from model import *
from dataloader import *
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import os
import json

# Parameters
path_to_fer_csv = 'data/fer2013/fer2013.csv'
tr_batch_sz = 512 # Default: 3000
val_batch_sz = 32 # Default: 500

criterion_loss = nn.NLLLoss() # Default: nn.NLLLoss()
epochs = 200 # Default: 200
lr = 1e-5 # Default: 1e-5
weight_decay = 1e-5 # Default: 1e-5
visualize_learning_curve = True # Default: True

# Create directory to save models and plots with os module
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs('checkpoints/'+now, exist_ok=True)

os.makedirs('checkpoints/'+now+'/models', exist_ok=True)
os.makedirs('checkpoints/'+now+'/plots', exist_ok=True)

# Create config file with parameters and save it as .json
config = {
    'tr_batch_sz': tr_batch_sz,
    'val_batch_sz': val_batch_sz,
    'criterion_loss': str(criterion_loss),
    'epochs': epochs,
    'lr': lr,
    'weight_decay': weight_decay,
} 
with open('checkpoints/'+now+'/config.json', 'w') as f:
    json.dump(config, f)

def train_model(model, trainloader, validloader, epochs, visualize_learning_curve):
    criterion = criterion_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    valid_loss_min = np.Inf
    train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []
    for e in range(epochs):
        model.train()
        running_loss = 0
        tr_accuracy = 0
        for images, labels in tqdm(trainloader):
            images = images.cuda()
            labels = labels.long().cuda()
            optimizer.zero_grad()
            
            log_ps  = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            tr_accuracy += torch.mean(equals.type(torch.FloatTensor))
        # else:
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in validloader:
                images = images.cuda()
                labels = labels.long().cuda()
                log_ps = model(images)
                val_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        train_losses.append(running_loss/len(trainloader))
        val_losses.append(val_loss/len(validloader))
        train_accuracy.append(tr_accuracy/len(trainloader))
        val_accuracy.append(accuracy/len(validloader))


        print("Epoch: {}/{} ".format(e+1, epochs),
            "Training Loss: {:.3f} ".format(train_losses[-1]),
            "Training Acc: {:.3f} ".format(tr_accuracy/len(trainloader)),
            "Val Loss: {:.3f} ".format(val_losses[-1]),
            "Val Acc: {:.3f}".format(accuracy/len(validloader)))
        
        if val_loss/len(validloader) <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            val_loss/len(validloader)))
            torch.save(model.state_dict(), 'checkpoints/'+now+'/models/best_model.pt')
            valid_loss_min = val_loss/len(validloader)

        # Save the model every epoch
        if visualize_learning_curve and e > 1:
            # Convert the lists into tensors
            train_losses = [torch.tensor(x) for x in train_losses]
            val_losses = [torch.tensor(x) for x in val_losses]

            # Convert the tensors into numpy arrays
            train_losses = [x.cpu().clone().detach().numpy() for x in train_losses]
            val_losses = [x.cpu().clone().detach().numpy() for x in val_losses]

            # Generate a plot that shows the training and validation loss over time and save it as .png
            plt.plot(train_losses, label='Training loss')
            plt.plot(val_losses, label='Validation loss')
            plt.legend(frameon=False)
            plt.savefig('checkpoints/'+now+'/plots/learning_curve.png')
            plt.close()

            # Generate a plot that shows the train and validation accuracy over time and save it as .png
            plt.plot(train_accuracy, label='Training accuracy')
            plt.plot(val_accuracy, label='Validation accuracy')
            plt.legend(frameon=False)
            plt.savefig('checkpoints/'+now+'/plots/accuracy_curve.png')
            plt.close()


    return model



def main():
    print ('Parameters:')
    print ('Training batch size: ', tr_batch_sz)
    print ('Validation batch size: ', val_batch_sz)
    print('Criterion Loss: ', criterion_loss)
    print ('Epochs: ', epochs)
    print ('Learning rate: ', lr)
    print ('Weight decay: ', weight_decay)
    print ('Visualize Learning Curve: ', visualize_learning_curve)

    print('\nPreprocess Data and get DataLoaders...')
    trainloader, validloader = get_dataloaders(path_to_fer_csv, tr_batch_sz, val_batch_sz)
    print('Data Preprocessed and got DataLoaders\n')
    print ('Training data size: ', len(trainloader.dataset))
    print ('Validation data size: ', len(validloader.dataset))

    print('\nLoad Model...')
    model = Face_Emotion_CNN()
    print('Model Loaded')
    if torch.cuda.is_available():
        model.cuda()
        print('GPU Found, Moving Model to CUDA')
    else:
        print('GPU not found, using model with CPU')

    print('Starting Training loop...\n')
    model = train_model(model, trainloader, validloader, epochs, visualize_learning_curve)    
    print('Training Done')

if __name__ == '__main__':
    main()