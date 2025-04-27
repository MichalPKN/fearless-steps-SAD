print("starting code")
import load
#import model_architectures.model_transformer as model_sad
import model_architectures.model_conformer as model_sad
import numpy as np
import os
import argparse
import time
import gc
from utils import plot_result, SADDataset, split_file, check_gradients, smooth_outputs_rnn
from train_dev_eval import train_model, validate_model, evaluate_model

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

#turn to True later
# torch.backends.cudnn.enabled = False

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
print("CUDA device count:", torch.cuda.device_count())

# hyperparameters
input_size = 12
hidden_size = 256
epochs = 3 if debug else 20
# batch_size = 1
criteria = 0.5
learning_rate = 0.0005
frame_length = 0.01
num_layers = 2
shuffle_batches = True
audio_size = 250
num_heads = 4
overlap = 50
context_size = 0.025


# for context_size in [0.01, 0.025, 0.05]:
#     for input_size in [30]:


data_loader = load.LoadAudio(debug=debug, input_size=input_size, frame_length=frame_length, context_size=context_size)



# eval data
X_val_loaded, val_info, Y_val_loaded = data_loader.load_all(dev_path, dev_labels)
if debug:
    X_val_loaded = [x[:20134] for x in X_val_loaded]
    Y_val_loaded = [y[:20134] for y in Y_val_loaded]
print(f"num of eval data: {len(X_val_loaded)}")

# evaluation
print("\nEVALUTAION")

best_model = torch.load(os.path.join("models", "best_model_ctf_10-250_0.0005_128_6.pt"), map_location=device)
                    
X_val, Y_val, masks = split_file(X_val_loaded, Y_val_loaded, seq_size=audio_size, overlap=overlap, shuffle=False)
dataset_val = SADDataset(X_val, Y_val, masks)
print(f"X_val length: {len(X_val)}")
print(f"X_val[0] shape: {X_val[0].shape}")
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

# eval
eval_accuracy, eval_dcf, eval_dcf_smooth, toshow_y, toshow_preds, toshow_outputs, toshow_additional = evaluate_model(
    best_model=best_model, dataloader_val=dataloader_val, criteria=criteria, device=device, best_smooth_window=20)                
    
print(f"eval\teval_smoothed")
print(f"{eval_dcf*100:.4f}\t{eval_dcf_smooth*100:.4f}")

if debug:
    path = os.path.join(datadir_path, "plots_rnn")
else:
    path = "/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD/plots_rnn"
    

plot_result(toshow_y.numpy(), toshow_preds.cpu().numpy(), toshow_outputs.cpu().detach().numpy(), toshow_additional.cpu().detach().numpy(), \
                path=path, file_name="sad_prediction_comparison_best_model1.png", debug=True, \
                title=f"conformer")

print("\n----------------------------------------\n\n\n")

        
