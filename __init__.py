import load
import model
import numpy as np
import os
import argparse
import time
from helper_functions import plot_result, SADDataset

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

test_num = 1
for batch_size in [1]:
    for learning_rate in [0.001, 0.0001]:
        for input_size in [13, 40]:
            print(f"\n\nbatch_size: {batch_size}, learning_rate: {learning_rate}, input_size: {input_size}")

            # hyperparameters
            # input_size = 40
            hidden_size = [64, 32, 16] if debug else [1024, 512, 256]
            epochs = 4 if debug else 8
            # batch_size = 1
            criteria = 0.5
            # learning_rate = 0.001
            frame_length = 0.01
            #num_layers = 3 

            data_loader = load.LoadAudio(debug=debug, input_size=input_size, frame_length=frame_length)

            # train data
            X, audio_info, Y = data_loader.load_all(train_path, train_labels)
            dataset = SADDataset(X, Y)
            #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # maybe shuffle True

            # dev data
            X_dev, _, Y_dev = data_loader.load_all(dev_path, dev_labels)
            dataset_dev = SADDataset(X_dev, Y_dev, max_len=dataset.max_len)
            #dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=False)


            # model
            sad_model = model.SADModel(input_size, hidden_size).to(device)
            # weight
            one_ratio = audio_info[0] / audio_info[2]
            zero_ratio = audio_info[1] / audio_info[2]
            print(f"one_ratio: {one_ratio}, zero_ratio: {zero_ratio}")
            pos_weight = torch.tensor(audio_info[1] / audio_info[0]).to(device)
            print(pos_weight)
            criterion = torch.nn.BCELoss(weight=pos_weight).to(device)
            
            optimizer = torch.optim.Adam(sad_model.parameters(), lr=learning_rate)

            # training
            load_time = time.time() - start_time
            print(f"Data loaded in {load_time:.2f} seconds")

            print("training model")
            # i = 1
            for epoch in range(epochs):
                # train
                sad_model.train()
                accuracies = []
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                fp_time = 0
                fn_time = 0
                y_speech_time = 0
                y_nonspeech_time = 0
                for i in range(len(X)):
                    batch_x, batch_y = torch.tensor(X[i], dtype=torch.float32), torch.tensor(Y[i], dtype=torch.float32)
                    print("batch_x: ", batch_x.shape, "batch_y: ", batch_y.shape)
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward
                    outputs = sad_model(batch_x)
                    #print(outputs.mean())
                    loss = criterion(outputs, batch_y)
                    
                    preds = (outputs >= criteria).float()
                    correct_predictions += ((preds == batch_y).float()).sum().item()
                    total_predictions += len(batch_y)
                    
                    # plot_result(batch_y[0].cpu().numpy(), preds[0].cpu().numpy(), outputs[0].cpu().detach().numpy(), path=datadir_path, file_name="sad_prediction_comparison" + str(i) + ".png", debug=False)
                    # i += 1
                    
                    #print("predsum: ", preds.sum(), "batch_y sum: ", batch_y.sum())
                    
                    #print(preds.shape)
                    # Backward
                    loss.backward()
                    optimizer.step()
                    fp_time += ((preds == 1) & (batch_y == 0)).sum().item()
                    fn_time += ((preds == 0) & (batch_y == 1)).sum().item()
                    y_speech_time += (batch_y).sum().item()
                    y_nonspeech_time += ((batch_y == 0)).sum().item()
                    
                    #print(fp_time, fn_time, y_speech_time, y_nonspeech_time)
                    running_loss += loss.item()
                train_accuracy = correct_predictions / total_predictions
                pfp = fp_time / y_nonspeech_time # false alarm
                pfn = fn_time / y_speech_time # miss
                dcf = 0.75 * pfn + 0.25 * pfp
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(X):.4f}, Accuracy: {train_accuracy*100:.2f}, DCF: {dcf*100:.2f}")
                
                # eval
                sad_model.eval()    
                with torch.no_grad():
                    correct_predictions = 0
                    total_predictions = 0
                    fp_time = 0
                    fn_time = 0
                    y_speech_time = 0
                    y_nonspeech_time = 0
                    for i in range(len(X_dev)):
                        batch_x, batch_y = torch.tensor(X_dev[i], dtype=torch.float32), torch.tensor(Y_dev[i], dtype=torch.float32)
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = sad_model(batch_x)
                        preds = (outputs >= criteria).float()
                        correct_predictions += ((preds == batch_y).float()).sum().item()
                        total_predictions += len(batch_y)
                        fp_time += ((preds == 1) & (batch_y == 0)).sum().item()
                        fn_time += ((preds == 0) & (batch_y == 1)).sum().item()
                        y_speech_time += (batch_y).sum().item()
                        y_nonspeech_time += ((batch_y == 0)).sum().item()
                    # for batch_x, batch_y, mask in dataloader_dev:
                    #     batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device),    mask.to(device)
                    #     outputs = sad_model(batch_x)
                    #     preds = (outputs >= criteria).float()
                    #     correct_predictions += ((preds == batch_y).float() * mask).sum().item()
                    #     total_predictions += mask.sum().item()
                    #     fp_time += (((preds == 1) & (batch_y == 0)) * mask).sum().item()
                    #     fn_time += (((preds == 0) & (batch_y == 1)) * mask).sum().item()
                    #     y_speech_time += (batch_y * mask).sum().item()
                    #     y_nonspeech_time += ((batch_y == 0) * mask).sum().item()
                    dev_accuracy = correct_predictions / total_predictions
                    pfp = fp_time / y_nonspeech_time # false alarm
                    pfn = fn_time / y_speech_time # miss
                    dev_dcf = 0.75 * pfn + 0.25 * pfp
                
                    print(f'Validation Accuracy: {dev_accuracy*100:.2f}, Validation DCF: {dev_dcf*100:.4f}')
                
                torch.cuda.empty_cache()
                
            print("finished training model")
            training_time = time.time() - start_time - load_time
            print(f"Training completed in {training_time:.2f} seconds")

            plot_result(batch_y.cpu().numpy(), preds.cpu().numpy(), outputs.cpu().detach().numpy(), path=datadir_path, file_name="sad_prediction_comparison_hp_" + str(test_num) + ".png", debug=False)
            test_num += 1
            
print(f"Total time: {time.time() - start_time:.2f} seconds")
            
