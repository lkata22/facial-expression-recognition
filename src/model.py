import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 7)
    
    def forward(self, x):
        x = self.model(x)
        return x



class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DeeperCNN2(nn.Module):
  def __init__(self):
      super(DeeperCNN2, self).__init__()
      self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
      self.bn1 = nn.BatchNorm2d(64)

      self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
      self.bn2 = nn.BatchNorm2d(128)

      self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
      self.bn3 = nn.BatchNorm2d(256)

      self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
      self.bn4 = nn.BatchNorm2d(512)

      self.conv5 = nn.Conv2d(512, 512, 3, padding=1)
      self.bn5 = nn.BatchNorm2d(512)

      self.pool = nn.MaxPool2d(2, 2)
      self.dropout = nn.Dropout(0.5)

      self.fc1 = nn.Linear(512 * 1 * 1, 512)
      self.fc2 = nn.Linear(512, 7)

  def forward(self, x):
      x = self.pool(F.relu(self.bn1(self.conv1(x))))
      x = self.pool(F.relu(self.bn2(self.conv2(x))))
      x = self.pool(F.relu(self.bn3(self.conv3(x))))
      x = self.pool(F.relu(self.bn4(self.conv4(x))))
      x = self.pool(F.relu(self.bn5(self.conv5(x))))

      x = x.view(-1, 512 * 1 * 1)
      x = self.dropout(F.relu(self.fc1(x)))
      x = self.fc2(x)
      return x

