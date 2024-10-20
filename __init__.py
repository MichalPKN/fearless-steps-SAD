import load
import model
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

# train_path = "FSC_P4_Streams\Audio\Streams\Train"
# train_labels = "FSC_P4_Streams\Transcripts\SAD\Train"
# dev_path = "FSC_P4_Streams\Audio\Streams\Dev"
# dev_labels = "FSC_P4_Streams\Transcripts\SAD\Dev"

current_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_path, "FSC_P4_Streams\Audio\Streams\Train")
train_labels = os.path.join(current_path, "FSC_P4_Streams\Transcripts\SAD\Train")
dev_path = os.path.join(current_path, "FSC_P4_Streams\Audio\Streams\Dev")
dev_labels = os.path.join(current_path, "FSC_P4_Streams\Transcripts\SAD\Dev")


data_loader = load.LoadAudio(debug=True)
X, audio_info_list, Y = data_loader.load_all(train_path, train_labels)


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

input_size = 40  # feature size from MFCC
hidden_size = 256
num_layers = 3  # LSTM layers

sad_model = model.SADModel(input_size, hidden_size, num_layers)

print("training model")
for batch_x, batch_y in dataloader:
    output = sad_model(batch_x) 
print("finished training model")
print("last batch: ", output.shape, "mean: ", output.mean())

