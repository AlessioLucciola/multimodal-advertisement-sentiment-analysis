import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, hidden_layers, num_classes, dropout=0.5):
        super(CNN, self).__init__()
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_layers[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_layers[1], out_channels=hidden_layers[2], kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(hidden_layers[2] * 4 * 4, hidden_layers[3])
        self.fc2 = nn.Linear(hidden_layers[3], num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x