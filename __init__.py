import load
import model
import numpy as np
import os
import argparse
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, required=False, help="path to where FSC_P4_Streams is located")
parser.add_argument("--debug", required=False, action="store_true", help="do a test run")
args = parser.parse_args()

debug = args.debug
datadir_path = args.datadir or ""
train_path = os.path.join(datadir_path, "FSC_P4_Streams", "Audio", "Streams", "Train")
train_labels = os.path.join(datadir_path, "FSC_P4_Streams", "Transcripts", "SAD", "Train")
dev_path = os.path.join(datadir_path, "FSC_P4_Streams", "Audio", "Streams", "Dev")
dev_labels = os.path.join(datadir_path, "FSC_P4_Streams", "Transcripts", "SAD", "Dev")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

try:
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
except Exception as e:
    print("Failed to get CUDA device info. Error:", e)

try:
    print("PyTorch version:", torch.__version__)
    print("CUDA version compiled with PyTorch:", torch.version.cuda)    
    print("CUDA runtime version:", torch._C._cuda_getDriverVersion() if torch.cuda.is_available() else "No CUDA runtime available")
except Exception as e:
    print("Failed to get PyTorch info. Error:", e)

# # cuda test
# try:
#     x = torch.rand(3, 3).to("cuda")
#     print("Tensor on CUDA:", x)
# except Exception as e:
#     print("Failed to move tensor to CUDA. Error:", e)

# hyperparameters
input_size = 40
hidden_size = [64, 32, 16] if debug else [1024, 512, 256]
epochs = 2 if debug else 20
batch_size = 5
criteria = 0.5
learning_rate = 0.001
frame_length = 0.01
#num_layers = 3 


class SADDataset(Dataset):
    def __init__(self, X, Y, min_len=None):
        self.X = X  # List of feature matrices
        self.Y = Y  # List of labels (0/1)
        #self.min_len = min_len or min([len(x) for x in X])
        self.max_len = max([len(x) for x in X])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        x_padded = F.pad(x, (0, 0, 0, self.max_len - len(x)))
        y_padded = F.pad(y, (0, self.max_len - len(y)))
        return x_padded, y_padded
        # x_cut = x[:self.min_len]
        # y_cut = y[:self.min_len]
        # return x_cut, y_cut

data_loader = load.LoadAudio(debug=debug, input_size=input_size, frame_length=frame_length)

# train data
X, audio_info_list, Y = data_loader.load_all(train_path, train_labels)
dataset = SADDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# dev data
X_dev, _, Y_dev = data_loader.load_all(dev_path, dev_labels)
dataset_dev = SADDataset(X_dev, Y_dev)
dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True)


# model
sad_model = model.SADModel(input_size, hidden_size).to(device)
criterion = torch.nn.BCELoss()  # Binary Cross-Entropy for binary output
optimizer = torch.optim.Adam(sad_model.parameters(), lr=learning_rate)

# training
load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} seconds")

print("training model")
sad_model.train()
for epoch in range(epochs):
    accuracies = []
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    fp_time = 0
    fn_time = 0
    y_speech_time = 0
    y_nonspeech_time = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = sad_model(batch_x)
        loss = criterion(outputs, batch_y)
        
        #acc = ((outputs>=criteria) == batch_y).sum().item() / batch_y.shape[1]
        preds = (outputs >= criteria).float()
        correct_predictions += (preds == batch_y).sum().item()
        total_predictions += batch_y.numel()
        
        # Backward
        loss.backward()
        optimizer.step()
        
        fp_time += ((preds == 1) & (batch_y == 0)).sum().item()
        fn_time += ((preds == 0) & (batch_y == 1)).sum().item()
        y_speech_time += batch_y.sum().item()
        y_nonspeech_time += batch_y.numel() - batch_y.sum().item()
        
        running_loss += loss.item()
    train_accuracy = correct_predictions / total_predictions
    pfp = fp_time / y_nonspeech_time # false alarm
    pfn = fn_time / y_speech_time # miss
    dcf = 0.75 * pfn + 0.25 * pfp
    
    # eval
    sad_model.eval()    
    correct_predictions = 0
    total_predictions = 0
    fp_time = 0
    fn_time = 0
    y_speech_time = 0
    y_nonspeech_time = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader_dev:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = sad_model(batch_x)
            preds = (outputs >= criteria).float()
            correct_predictions += (preds == batch_y).sum().item()
            total_predictions += batch_y.numel()
            fp_time += ((preds == 1) & (batch_y == 0)).sum().item()
            fn_time += ((preds == 0) & (batch_y == 1)).sum().item()
            y_speech_time += batch_y.sum().item()
            y_nonspeech_time += batch_y.numel() - batch_y.sum().item()
    dev_accuracy = correct_predictions / total_predictions
    pfp = fp_time / y_nonspeech_time # false alarm
    pfn = fn_time / y_speech_time # miss
    dev_dcf = 0.75 * pfn + 0.25 * pfp
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {train_accuracy:.4f}, DCF: {dcf:.4f}")
    print(f'Validation Accuracy: {dev_accuracy:.4f}, Validation DCF: {dev_dcf:.4f}')
    
print("finished training model")
training_time = time.time() - start_time - load_time
print(f"Training completed in {training_time:.2f} seconds")
#print("last batch: ", output.shape, "mean: ", output.mean())

