'''
file: main.py
author: @vincit0re
brief: This file contains the main code flow for the project.
date: 20230-05-05
'''

from utils import *
from hyperparameters import *

argparse = argparse.ArgumentParser("Facial Emotion Detection")
argparse.add_argument('--lr', type=float, default=1e-3)
argparse.add_argument('--weight_decay', type=float, default=1e-5)
argparse.add_argument('--num_epochs', type=int, default=2)
argparse.add_argument('--num_classes', type=int, default=7)
argparse.add_argument('--print_every', type=int, default=100)
argparse.add_argument('--batch_size', type=int, default=32)
argparse.add_argument('--shuffle', type=bool, default=True)
argparse.add_argument('--debug', type=bool, default=True)
argparse.add_argument('--save_path', type=str, default='model.pt')
argparse.add_argument('--num_workers', type=int, default=1)
argparse.add_argument('--model_name', type=str, default='resnet34')
args = argparse.parse_args()

# get the dataloaders
train_loader, val_loader, test_loader = get_dataloaders(
    data_dir=Hyperparameters._DATA_DIR, batch_size=args.batch_size, debug=args.debug)

show_images(train_loader, 'Train Images')
show_images(test_loader, 'Test Images')


params = {
    'lr': args.lr,
    'weight_decay': args.weight_decay,
    'num_epochs': args.num_epochs,
    'num_classes': args.num_classes,
    'print_every': args.print_every,
    'batch_size': args.batch_size,
    'shuffle': args.shuffle,
    'debug': args.debug,
    'save_path': args.save_path,
    'num_workers': args.num_workers
}

model = get_model(params['num_classes'], device, model_name= args.model_name)

#################### Training ####################
trained_model, history = train_model(
    model, train_loader, val_loader, device=device, params=params, debug=Hyperparameters._DEBUG)
##################################################

# plot the results
plot_results(history)

#################### Testing ######################
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = evaluate_model(
    trained_model, test_loader, criterion, device)
###################################################

################ Testing Results ##################
print("-"*100)
print(f"| Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f} |")
print("-"*100)
###################################################
