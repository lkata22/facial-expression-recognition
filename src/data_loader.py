import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FERDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = np.fromstring(self.data.iloc[idx]['pixels'], sep=' ', dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(image).convert("L")
        label = int(self.data.iloc[idx]['emotion'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

