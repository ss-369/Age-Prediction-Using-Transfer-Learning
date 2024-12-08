import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
import torch.optim as optim

class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()
        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train
        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
            self.transform = self._transform(224, augment=True) 
        else:
            self.transform = self._transform(224, augment=False)

    @staticmethod
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px, augment=True):
        transforms = [Resize(n_px), self._convert_image_to_rgb, ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        if augment:
            transforms.insert(1, RandomHorizontalFlip())  
        return Compose(transforms)

    def read_img(self, file_name):
        im_path = join(self.data_path, file_name)
        img = Image.open(im_path)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img

    def __len__(self):
        return len(self.files)

train_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)

test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(model_name="resnet50", pretrained=True):
    if model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
    else: 
        raise ValueError("Invalid model name. Choose 'resnet50'")
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) 
    return model.to(device)

model = create_model()

def train(model, train_loader, optimizer, criterion, epochs=10):
    best_mae = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), labels.float()) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_mae = evaluate(model, train_loader)  
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Val MAE: {val_mae}')
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), 'best_model.pth')
    print('Finished Training')

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = torch.nn.L1Loss()(outputs.flatten(), labels.float())
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

@torch.no_grad()
def predict(loader, model):
    model.eval()
    predictions = []
    for img in tqdm(loader):
        img = img.to(device)
        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())
    return predictions

criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  
train(model, train_loader, optimizer, criterion, epochs=15)

model.load_state_dict(torch.load('best_model.pth'))

preds = predict(test_loader, model)

submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
submit['age'] = preds
submit.head()
submit.to_csv('submission.csv', index=False)

# Print MAE loss on the test set
test_mae = evaluate(model, test_loader)
print(f"Test MAE: {test_mae}")