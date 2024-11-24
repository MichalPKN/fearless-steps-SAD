import load
import model
import model2l
import numpy as np
import os
import argparse
import time
from helper_functions import plot_result, SADDataset, split_file, check_gradients, smooth_outputs

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
print("CUDA device count:", torch.cuda.device_count())

# hyperparameters
input_size = 30
hidden_size = [1024, 512]
epochs = 4 if debug else 25
# batch_size = 1
criteria = 0.5
# learning_rate = 0.001
frame_length = 0.01
#num_layers = 3 
shuffle_batches = True

data_loader = load.LoadAudio(debug=debug, input_size=input_size, frame_length=frame_length)

# train data
X_loaded, audio_info, Y_loaded = data_loader.load_all(train_path, train_labels)
dataset = SADDataset(X_loaded, Y_loaded)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # maybe shuffle True

# dev data
X_dev_loded, dev_info, Y_dev_loaded = data_loader.load_all(dev_path, dev_labels)
dataset_dev = SADDataset(X_dev_loded, Y_dev_loaded, max_len=dataset.max_len)
#dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=False)

test_num = 1
X_loaded = X_loaded[:1000] if debug else X_loaded
Y_loaded = Y_loaded[:1000] if debug else Y_loaded
for f_test in range(2, 4):
    if f_test == 3:
        print("-----------------")
        print("No mfcc norm test")
        print("-----------------")
        data_loader = load.LoadAudio(debug=debug, input_size=input_size, frame_length=frame_length, norm=False)
        # train data
        X_loaded, audio_info, Y_loaded = data_loader.load_all(train_path, train_labels)
        dataset = SADDataset(X_loaded, Y_loaded)
        # dev data
        X_dev_loded, dev_info, Y_dev_loaded = data_loader.load_all(dev_path, dev_labels)
        dataset_dev = SADDataset(X_dev_loded, Y_dev_loaded, max_len=dataset.max_len)
    for batch_size in [1000]:
        if f_test == 2:
            print("-----------------")
            print("No shuffle test")
            print("-----------------")
            X, Y = split_file(X_loaded, Y_loaded, batch_size=batch_size, shuffle=False)
        else:
            X, Y = split_file(X_loaded, Y_loaded, batch_size=batch_size, shuffle=shuffle_batches)
        # X_dev, Y_dev = split_file(X_dev_loded, Y_dev_loaded, batch_size=30000, shuffle=shuffle_batches)
        #X_dev, Y_dev = split_file(X_dev_loded, Y_dev_loaded, batch_size=30000)
        X_dev, Y_dev = X_dev_loded, Y_dev_loaded
        for hidden_size in [[1024, 512]]:
            for learning_rate in [0.0005]:
                print(f"\n\nbatch_size: {batch_size}, learning_rate: {learning_rate}, hidden_size: {hidden_size}")
                print(f"X batch size: {len(X)}, X_dev batch size {len(X_dev)}")
                # model
                sad_model = model2l.SADModel(input_size, hidden_size).to(device)
                # weight
                one_ratio = audio_info[0] / audio_info[2]
                zero_ratio = audio_info[1] / audio_info[2]
                print(f"one_ratio: {one_ratio}, zero_ratio: {zero_ratio}")
                if f_test == 1:
                    print("-----------------")
                    print("No weight test")
                    print("-----------------")
                    criterion = torch.nn.BCEWithLogitsLoss()
                else:
                    pos_weight = torch.tensor(audio_info[1] / audio_info[0]).to(device)
                    print("pos_weight: ", pos_weight)
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                optimizer = torch.optim.Adam(sad_model.parameters(), lr=learning_rate)

                # training
                load_time = time.time() - start_time
                print(f"Data loaded in {load_time:.2f} seconds")

                print("training model")
                # i = 1
                for epoch in range(epochs):
                    # train
                    sad_model.train()
                    losses = np.zeros(epochs)
                    running_loss = 0.0
                    correct_predictions = 0
                    total_predictions = 0
                    fp_time = 0
                    fn_time = 0
                    y_speech_time = 0
                    y_nonspeech_time = 0
                    for i in range(len(X)):   #len(X)):
                        batch_x, batch_y = torch.tensor(X[i], dtype=torch.float32), torch.tensor(Y[i], dtype=torch.float32)
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        
                        
                        optimizer.zero_grad()
                        
                        # Forward
                        outputs = sad_model(batch_x)
                        #print(outputs.mean())
                        loss = criterion(outputs, batch_y)
                        
                        loss.backward()
                        optimizer.step()
                        
                        outputs = torch.sigmoid(outputs)
                        
                        preds = (outputs >= criteria).float()
                        correct_predictions += ((preds == batch_y).float()).sum().item()
                        total_predictions += len(batch_y)
                        
                        
                        # plot_result(batch_y[0].cpu().numpy(), preds[0].cpu().numpy(), outputs[0].cpu().detach().numpy(), path=datadir_path, file_name="sad_prediction_comparison" + str(i) + ".png", debug=False)
                        # i += 1
                        
                        #print("predsum: ", preds.sum(), "batch_y sum: ", batch_y.sum())
                        
                        #print(preds.shape)
                        # Backward
                        
                        fp_time += ((preds == 1) & (batch_y == 0)).sum().item()
                        fn_time += ((preds == 0) & (batch_y == 1)).sum().item()
                        y_speech_time += (batch_y).sum().item()
                        y_nonspeech_time += ((batch_y == 0)).sum().item()
                        
                        ## checking explosing gradients
                        # if epoch % 2 == 0 and i == 0:  # Check gradients for the first batch every 2 epochs
                        #     print(f"Epoch {epoch+1}, Batch {i}")
                        #     check_gradients(sad_model)
                        
                        #print(fp_time, fn_time, y_speech_time, y_nonspeech_time)
                        running_loss += loss.item()
                        ## debug:
                        # if epoch == 0 and (i < 20 or (i > 150 and i < 170)):
                            # print(f"i: {i}, Loss: {running_loss/(i+1):.4f}, running_loss: {running_loss:.4f}")
                        if epoch == 0 and i % (len(X) // 10) == 0:
                            train_accuracy = correct_predictions / total_predictions
                            pfp = fp_time / (y_nonspeech_time + 0.0001) # false alarm
                            pfn = fn_time / (y_speech_time + 0.0001) # miss
                            dcf = 0.75 * pfn + 0.25 * pfp
                            print(f'first epoch, Loss: {loss:.4f}, Accuracy: {train_accuracy*100:.2f}, DCF: {dcf*100:.2f}')
                            print("size:", len(preds), "fp_time:", preds.sum(), "ones actual:", batch_y.sum(), "mean:", outputs.mean())
                            print("-----------------------------")
                    train_accuracy = correct_predictions / total_predictions
                    pfp = fp_time / y_nonspeech_time # false alarm
                    pfn = fn_time / y_speech_time # miss
                    dcf = 0.75 * pfn + 0.25 * pfp
                    losses[epoch] = running_loss/len(X)
                    print()
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
                        fp_time_smooth = 0
                        fn_time_smooth = 0
                        for i in range(len(X_dev)):
                            batch_x, batch_y = torch.tensor(X_dev[i], dtype=torch.float32), torch.tensor(Y_dev[i], dtype=torch.float32)
                            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                            outputs = sad_model(batch_x)
                            outputs = torch.sigmoid(outputs)
                            preds = (outputs >= criteria).float()
                            correct_predictions += ((preds == batch_y).float()).sum().item()
                            total_predictions += len(batch_y)
                            fp_time += ((preds == 1) & (batch_y == 0)).sum().item()
                            fn_time += ((preds == 0) & (batch_y == 1)).sum().item()
                            y_speech_time += (batch_y).sum().item()
                            y_nonspeech_time += ((batch_y == 0)).sum().item()
                            if i == 0:
                                toshow_y = batch_y
                                toshow_preds = preds
                                toshow_outputs = outputs
                                toshow_additional = dev_info[3][i]
                                
                            # smoothing:
                            smooth_preds = smooth_outputs(preds, avg_frames=5)
                            fp_time_smooth += ((smooth_preds == 1) & (batch_y == 0)).sum().item()
                            fn_time_smooth += ((smooth_preds == 0) & (batch_y == 1)).sum().item()
                            
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

                        pfp_smooth = fp_time_smooth / y_nonspeech_time # false alarm
                        pfn_smooth = fn_time_smooth / y_speech_time # miss
                        dev_dcf_smooth = 0.75 * pfn_smooth + 0.25 * pfp_smooth
                        
                        print(f'Validation Accuracy: {dev_accuracy*100:.2f}, Validation DCF: {dev_dcf*100:.4f}, Validation DCF smooth: {dev_dcf_smooth*100:.4f}')
                    
                    torch.cuda.empty_cache()
                    
                print("finished training model")
                training_time = time.time() - start_time - load_time
                print(f"Training completed in {training_time:.2f} seconds, {training_time/60:.2f} minutes")
                print(f"losses: {losses}")
                if debug:
                    path = os.path.join(datadir_path, "plots")
                else:
                    path = "/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD/plots"
                
                #TODO: [30000:60000] remove/alter
                plot_result(toshow_y.cpu().numpy()[30000:60000], toshow_preds.cpu().numpy()[30000:60000], toshow_outputs.cpu().detach().numpy()[30000:60000], toshow_additional, \
                            path=path, file_name="sad_prediction_comparison_hp_" + str(test_num) + ".png", debug=False, \
                            title=f"batch_size: {batch_size}, learning_rate: {learning_rate}, hidden_size: {hidden_size}")
                test_num += 1
            
print(f"Total time: {time.time() - start_time:.2f} seconds")
            
