import load
import model
import numpy as np
import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader


debug = False


parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, required=False, help="path to where FSC_P4_Streams is located")
args = parser.parse_args()

datadir_path = args.datadir or ""
train_path = os.path.join(datadir_path, "FSC_P4_Streams", "Audio", "Streams", "Train")
train_labels = os.path.join(datadir_path, "FSC_P4_Streams", "Transcripts", "SAD", "Train")
dev_path = os.path.join(datadir_path, "FSC_P4_Streams", "Audio", "Streams", "Dev")
dev_labels = os.path.join(datadir_path, "FSC_P4_Streams", "Transcripts", "SAD", "Dev")

data_loader = load.LoadAudio(debug=debug)
X, audio_info_list, Y = data_loader.load_all(train_path, train_labels)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA current device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

class SADDataset(Dataset):
    def __init__(self, X, Y, min_len=None):
        self.X = X  # List of feature matrices
        self.Y = Y  # List of labels (0/1)
        self.min_len = min_len or min([len(x) for x in X])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        # x_padded = F.pad(x, (0, 0, 0, self.max_len - len(x)))
        x_cut = x[:self.min_len]
        y_cut = y[:self.min_len]
        return x_cut, y_cut


dataset = SADDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

input_size = 40  # MFCC features
hidden_size = [64, 32, 16] if debug else [512, 256, 128]
#num_layers = 3 

sad_model = model.SADModel(input_size, hidden_size).to(device)
epochs = 2 if debug else 20
criterion = torch.nn.BCELoss()  # Binary Cross-Entropy for binary output
optimizer = torch.optim.Adam(sad_model.parameters(), lr=0.001)

print("training model")
sad_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = sad_model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
print("finished training model")
#print("last batch: ", output.shape, "mean: ", output.mean())

