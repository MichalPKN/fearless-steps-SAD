#import plotly.graph_objects as go
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import os
import librosa


def plot_result(y_actual, y_pred, processed_predictions=None, additional=None, path="", file_name="sad_prediction_comparison.png", debug=False, title="actual vs predictions"):    
    # Plotting in subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # # Plot smoothed predictions
    # #axs[0].plot(y_pred, label="Predikce", color="blue", alpha=0.5)
    # if processed_predictions is not None:
    #     axs[0].plot(processed_predictions, label="výstup modelu", color="red", linestyle='--')
    # #axs[0].set_ylabel("P")
    # #axs[0].legend(loc="lower right")
    
    
    # # Plot actual labels
    # axs[0].plot(y_actual, label="pravdivé výsledky", color="green", alpha=0.5)
    # #axs[0].set_ylabel("výstup modelu a pravdivé výsledky")
    # axs[0].set_title("výstup modelu a pravdivé výsledky")
    # # axs[0].legend(loc="lower center")
    
    x_vals = np.arange(0, len(y_actual)) / 100
    
    axs[0].plot(x_vals, y_actual, label="Predikce modelu", color="green", alpha=0.8)
    axs[0].plot(x_vals, processed_predictions, label="Výstup modelu", color="blue", linestyle='--', alpha=0.5)
    axs[0].set_title("Výstup modelu a pravdivé výsledky")
    axs[0].set_xlabel("Čas [s]")
    axs[0].legend()

    axs[1].imshow(x_vals, additional.T, aspect='auto', cmap='jet', origin='lower')
    axs[1].set_ylabel("Koeficient")
    axs[1].set_xlabel("Čas [s]")
    axs[1].set_title("MFCC")
    
    x_lim = axs[1].get_xlim()
    axs[0].set_xlim(x_lim)

    #plt.subplots_adjust(hspace=2)
    plt.suptitle(title)
    plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(path, file_name))
    if debug:
        plt.show()

class SADDataset(Dataset):
    def __init__(self, X, Y, masks, max_len=None, overlap=0):
        self.X = X  # List of feature matrices
        self.Y = Y  # List of labels (0/1)
        #self.min_len = min_len or min([len(x) for x in X])
        #self.max_len = max_len or max([len(x) for x in X])
        #self.overlap = overlap // 2
        self.masks = masks
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(self.X[idx]):
            x = self.X[idx]
        else:
            x = torch.tensor(self.X[idx], dtype=torch.float32)
        if torch.is_tensor(self.Y[idx]):
            y = self.Y[idx]
        else:
            y = torch.tensor(self.Y[idx], dtype=torch.float32)
        if torch.is_tensor(self.masks[idx]):
            mask = self.masks[idx]
        else:
            mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        # x_padded = F.pad(x, (0, 0, 0, self.max_len - len(x)))
        # y_padded = F.pad(y, (0, 0, 0, self.max_len - len(y)))
        
        # mask = torch.zeros_like(y_padded)
        # mask[:len(x)] = 1
        # if idx > 0:
        #     mask[:self.overlap] = 0
        # if idx < len(self.X) - 1:
        #     mask[-self.overlap:] = 0
        return x, y, mask
        # x_cut = x[:self.min_len]
        # y_cut = y[:self.min_len]
        # return x_cut, y_cut


def split_file(X, y, seq_size=1000, overlap=200, shuffle=False):
    X_sequences = []
    y_sequences = []
    masks = []
    step_size = seq_size - overlap
    for j in range(len(X)):
        if not isinstance(X[j], torch.Tensor):
            X[j] = torch.tensor(X[j], dtype=torch.float32)
        if not isinstance(y[j], torch.Tensor):
            y[j] = torch.tensor(y[j], dtype=torch.float32)
        start = 0
        while start + step_size < len(X[j]):
            end = min(start + seq_size, len(X[j]))
            x_padded = F.pad(X[j][start:end], (0, 0, 0, seq_size - (end - start)))
            y_padded = F.pad(y[j][start:end], (0, 0, 0, seq_size - (end - start)))
            mask = torch.zeros_like(y_padded)
            mask[:end-start] = 1
            if overlap > 0 and start > 0:
                mask[:overlap//2] = 0
            if overlap > 0 and end < len(X[j]):
                mask[-(overlap//2):] = 0
            X_sequences.append(x_padded)
            y_sequences.append(y_padded)
            masks.append(mask)
            start += step_size
    print("splitted")
        # # remainder batch
        # remainder_length = len(X[j]) % (seq_size - overlap)
        # if remainder_length > 1:
        #     X_sequences.append(X[j][len(X[j]) - len(X[j]) % (step_size):])
        #     y_sequences.append(y[j][len(y[j]) - len(y[j]) % (step_size):])
            
    if shuffle:
        # Stack sequences into tensors for shuffling
        X_sequences = torch.stack(X_sequences)
        y_sequences = torch.stack(y_sequences)
        masks = torch.stack(masks)
        
        # Generate shuffled indices
        perm = torch.randperm(X_sequences.size(0))
        X_sequences = X_sequences[perm]
        y_sequences = y_sequences[perm]
        masks = masks[perm]

    print("Number of sequences: ", len(X_sequences))
    print("y_sequences length: ", len(y_sequences))
    print("last sequence length: ", len(X_sequences[len(X_sequences) - 1]))
    return X_sequences, y_sequences, masks

def check_gradients(asd_model):
    """Check for exploding/vanishing gradients."""
    for name, param in asd_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Gradient norm for {name}: {grad_norm:.4f}")
            if grad_norm > 1e4:  # Threshold for exploding gradients
                print(f"Warning: Exploding gradient detected in {name}")
            elif grad_norm < 1e-6:  # Threshold for vanishing gradients
                print(f"Warning: Vanishing gradient detected in {name}")

def smooth_outputs_rnn(smooth_preds, avg_frames=5, criteria=None):
    
    smooth_preds = smooth_preds.transpose(1, 2)
    kernel = torch.ones(avg_frames) / avg_frames
    kernel = kernel.to(smooth_preds.device)
    smoothed = F.conv1d(smooth_preds, kernel.view(1, 1, -1), padding=avg_frames // 2)
    if smoothed.size(2) > smooth_preds.size(2):
        smoothed = smoothed[:, :, :-1]
    smoothed = (smoothed >= criteria).float()
    return smoothed.transpose(1, 2)
    
    # unfolded = smooth_preds.unfold(dimension=1, size=avg_frames, step=1).mean(dim=-1)
    # smooth_preds[:, :-(avg_frames-1)] = unfolded
    # smooth_preds[:, -(avg_frames-1):] = smooth_preds[:, -avg_frames:].mean(dim=1, keepdim=True)
    # if criteria is not None:
    #     smooth_preds = (smooth_preds >= criteria).float()
    # return smooth_preds
