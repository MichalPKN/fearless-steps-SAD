print("starting code")
import load
import model
import model2l
import model_rnn
import numpy as np
import os
import argparse
import time
from helper_functions import plot_result, SADDataset, split_file, check_gradients, smooth_outputs_rnn

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
epochs = 3 if debug else 28
# batch_size = 1
criteria = 0.5
# learning_rate = 0.001
frame_length = 0.01
num_layers = 2
shuffle_batches = True
audio_size = 10000

data_loader = load.LoadAudio(debug=debug, input_size=input_size, frame_length=frame_length)

# train data
X_loaded, audio_info, Y_loaded = data_loader.load_all(train_path, train_labels)
X_loaded = X_loaded[:20000] if debug else X_loaded
Y_loaded = Y_loaded[:20000] if debug else Y_loaded

# train test split
print(f"num of data: {len(X_loaded)}")
#dev_idxs = [1] if debug else [5, 18, 27, 43, 68, 91, 112, 129]
dev_idxs = [1] if debug else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X_dev_loaded = [X_loaded[i] for i in dev_idxs]
Y_dev_loaded = [Y_loaded[i] for i in dev_idxs]
X_loaded = [X_loaded[i] for i in range(len(X_loaded)) if i not in dev_idxs]
Y_loaded = [Y_loaded[i] for i in range(len(Y_loaded)) if i not in dev_idxs]
print(f"num of trainig data: {len(X_loaded)}, num of dev data: {len(X_dev_loaded)}")

# eval data
X_val_loaded, val_info, Y_val_loaded = data_loader.load_all(dev_path, dev_labels)
X_val_loaded = X_val_loaded[:20000] if debug else X_val_loaded
Y_val_loaded = Y_val_loaded[:20000] if debug else Y_val_loaded
print(f"num of eval data: {len(X_val_loaded)}")


# training
test_num = 1
for f_test in range(1):
    for batch_size, audio_size in [[1, 10000000], [2, 10000000], [2, 100000]]: # [[40, 100], [1, 10000000], [2, 10000000], [2, 100000]]
        X, Y = split_file(X_loaded, Y_loaded, batch_size=audio_size, shuffle=False)
        dataset = SADDataset(X, Y) 
        print(f"max size: {dataset.max_len}")
        print(f"X length: {len(X)}")
        print(f"X[0] shape: {X[0].shape}")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_batches) # maybe shuffle True

        #X_dev, Y_dev = split_file(X_dev_loaded, Y_dev_loaded, batch_size=30000, shuffle=shuffle_batches)
        #X_dev, Y_dev = split_file(X_dev_loaded, Y_dev_loaded, batch_size=30000)
        
        X_dev, Y_dev = split_file(X_dev_loaded, Y_dev_loaded, batch_size=audio_size, shuffle=False)
        dataset_dev = SADDataset(X_dev, Y_dev, max_len=dataset.max_len)
        dataloader_dev = DataLoader(dataset_dev, batch_size=1, shuffle=False)
        
        X_val, Y_val = split_file(X_val_loaded, Y_val_loaded, batch_size=audio_size, shuffle=False)
        dataset_val = SADDataset(X_val, Y_val, max_len=dataset.max_len)
        dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
        
        for hidden_size in [512]:
            for learning_rate in [0.0001]:
                print(f"\n\nbatch_size: {batch_size}, learning_rate: {learning_rate}, hidden_size: {hidden_size}")
                print(f"X length: {len(X)}, X_dev length {len(X_dev)}")
                # model
                sad_model = model_rnn.SADModel(input_size, hidden_size, num_layers).to(device)
                # weight
                one_ratio = audio_info[0] / audio_info[2]
                zero_ratio = audio_info[1] / audio_info[2]
                print(f"one_ratio: {one_ratio}, zero_ratio: {zero_ratio}")
                if f_test == 1:
                    print("-----------------")
                    print("No weight test")
                    print("-----------------")
                    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
                else:
                    pos_weight = torch.tensor(audio_info[1] / audio_info[0]).to(device)
                    print("pos_weight: ", pos_weight)
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
                
                optimizer = torch.optim.Adam(sad_model.parameters(), lr=learning_rate)

                # training
                load_time = time.time() - start_time
                print(f"Data loaded in {load_time:.2f} seconds, {load_time/60:.2f} minutes")

                print("training model")
                # i = 1
                losses = np.zeros(epochs)
                for epoch in range(epochs):
                    # train
                    sad_model.train()
                    running_loss = 0.0
                    correct_predictions = 0
                    total_predictions = 0
                    fp_time = 0
                    fn_time = 0
                    y_speech_time = 0
                    y_nonspeech_time = 0
                    i = 0
                    for batch_x, batch_y, mask in dataloader:
                        batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
                        
                        if not batch_x.is_contiguous():
                            print("not contiguous")
                            batch_x = batch_x.contiguous()
                        
                        optimizer.zero_grad()
                        
                        # Forward
                        outputs = sad_model(batch_x)
                        #print(outputs.mean())
                        raw_loss = criterion(outputs, batch_y)
                        loss = (raw_loss * mask).mean()
                        
                        loss.backward()
                        optimizer.step()
                        
                        outputs = torch.sigmoid(outputs)
                        preds = (outputs >= criteria).float()
                        correct_predictions += ((preds == batch_y).float() * mask).sum().item()
                        total_predictions += mask.sum().item()
                        fp_time += (((preds == 1) & (batch_y == 0)) * mask).sum().item()
                        fn_time += (((preds == 0) & (batch_y == 1)) * mask).sum().item()
                        y_speech_time += (batch_y * mask).sum().item()
                        y_nonspeech_time += ((batch_y == 0) * mask).sum().item()
                        
                        ## checking explosing gradients
                        # if epoch % 2 == 0 and i == 0:  # Check gradients for the first batch every 2 epochs
                        #     print(f"Epoch {epoch+1}, Batch {i}")
                        #     check_gradients(sad_model)
                        
                        #print(fp_time, fn_time, y_speech_time, y_nonspeech_time)
                        running_loss += loss.item()
                        ## debug:
                        # if epoch == 0 and (i < 20 or (i > 150 and i < 170)):
                            # print(f"i: {i}, Loss: {running_loss/(i+1):.4f}, running_loss: {running_loss:.4f}")
                        # if epoch == 0 and i % (len(X) // 10) == 0:
                        #     train_accuracy = correct_predictions / total_predictions
                        #     pfp = fp_time / (y_nonspeech_time + 0.0001) # false alarm
                        #     pfn = fn_time / (y_speech_time + 0.0001) # miss
                        #     dcf = 0.75 * pfn + 0.25 * pfp
                        #     print(f'first epoch, Loss: {loss:.4f}, Accuracy: {train_accuracy*100:.2f}, DCF: {dcf*100:.2f}')
                        #     print("size:", len(preds), "fp_time:", preds.sum(), "ones actual:", batch_y.sum(), "mean:", outputs.mean())
                        #     print("-----------------------------")
                        i += 1
                    train_accuracy = correct_predictions / total_predictions
                    pfp = fp_time / y_nonspeech_time # false alarm
                    pfn = fn_time / y_speech_time # miss
                    dcf = 0.75 * pfn + 0.25 * pfp
                    losses[epoch] = running_loss/len(X)
                    print()
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(X):.4f}, Accuracy: {train_accuracy*100:.2f}, DCF: {dcf*100:.2f}")
                    
                    # dev
                    sad_model.eval()
                    with torch.no_grad():
                        smooth_window = [3, 5, 8, 10]
                        correct_predictions = 0
                        total_predictions = 0
                        fp_time = 0
                        fn_time = 0
                        y_speech_time = 0
                        y_nonspeech_time = 0
                        fp_time_smooth = [0 for asd in range(len(smooth_window))]
                        fn_time_smooth = [0 for asd in range(len(smooth_window))]
                        i = 0
                        for batch_x, batch_y, mask in dataloader_dev:
                            batch_x, batch_y, mask  = batch_x.to(device), batch_y.to(device), mask.to(device)
                            outputs = sad_model(batch_x)
                            outputs = torch.sigmoid(outputs)
                            preds = (outputs >= criteria).float()
                            correct_predictions += ((preds == batch_y).float() * mask).sum().item()
                            total_predictions += mask.sum().item()
                            fp_time += (((preds == 1) & (batch_y == 0)) * mask).sum().item()
                            fn_time += (((preds == 0) & (batch_y == 1)) * mask).sum().item()
                            y_speech_time += (batch_y * mask).sum().item()
                            y_nonspeech_time += ((batch_y == 0) * mask).sum().item()
                                                            
                            
                            # smoothing:
                            for window_idx, window in enumerate(smooth_window):
                                smooth_preds = smooth_outputs_rnn(preds, avg_frames=window, criteria=criteria)
                                fp_time_smooth[window_idx] += ((smooth_preds == 1) & (batch_y == 0)).sum().item()
                                fn_time_smooth[window_idx] += ((smooth_preds == 0) & (batch_y == 1)).sum().item()
                            
                            # if i == 0:
                            #     toshow_y = batch_y[0]
                            #     toshow_preds = preds[0]
                            #     toshow_outputs = outputs[0]
                            #     toshow_additional = smooth_preds[0]
                            i += 1
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
                        
                        print(f'Dev Accuracy: {dev_accuracy*100:.2f}, Dev DCF: {dev_dcf*100:.4f}')
                        best_smooth_window_dcf = 101
                        for window_idx, window in enumerate(smooth_window):
                            pfp_smooth = fp_time_smooth[window_idx] / y_nonspeech_time
                            pfn_smooth = fn_time_smooth[window_idx] / y_speech_time
                            dev_dcf_smooth = 0.75 * pfn_smooth + 0.25 * pfp_smooth
                            print(f'Dev DCF smooth {window}: {dev_dcf_smooth*100:.4f}', end=", ")
                            if dev_dcf_smooth < best_smooth_window_dcf:
                                best_smooth_window_dcf = dev_dcf_smooth
                                best_smooth_window = window
                        print()
                    
                torch.cuda.empty_cache()
                
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
                    i = 0
                    for batch_x, batch_y, mask in dataloader_val:
                        batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
                        outputs = sad_model(batch_x)
                        outputs = torch.sigmoid(outputs)
                        preds = (outputs >= criteria).float()
                        correct_predictions += ((preds == batch_y).float() * mask).sum().item()
                        total_predictions += mask.sum().item()
                        fp_time += (((preds == 1) & (batch_y == 0)) * mask).sum().item()
                        fn_time += (((preds == 0) & (batch_y == 1)) * mask).sum().item()
                        y_speech_time += (batch_y * mask).sum().item()
                        y_nonspeech_time += ((batch_y == 0) * mask).sum().item()
                        
                        # smoothing:
                        smooth_preds = smooth_outputs_rnn(preds, avg_frames=best_smooth_window, criteria=criteria)
                        fp_time_smooth += ((smooth_preds == 1) & (batch_y == 0)).sum().item()
                        fn_time_smooth += ((smooth_preds == 0) & (batch_y == 1)).sum().item()
                        
                        if i == 0:
                            toshow_y = batch_y[0]
                            toshow_preds = preds[0]
                            toshow_outputs = outputs[0]
                            toshow_additional = smooth_preds[0]
                        i += 1
                    # for batch_x, batch_y, mask in dataloader_val:
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
                    
                    print(f'Validation Accuracy: {dev_accuracy*100:.2f}, Validation DCF: {dcf*100:.4f}, Validation DCF smooth {best_smooth_window}: {dev_dcf_smooth*100:.4f}')
                    print()
                    
                    
                print("finished training model")
                training_time = time.time() - start_time - load_time
                print(f"Training completed in {training_time:.2f} seconds, {training_time/60:.2f} minutes")
                print(f"losses: {losses}")
                if debug:
                    path = os.path.join(datadir_path, "plots_rnn")
                else:
                    path = "/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD/plots_rnn"
                
                plot_result(toshow_y.cpu().numpy(), toshow_preds.cpu().numpy(), toshow_outputs.cpu().detach().numpy(), toshow_additional.cpu().detach().numpy(), \
                            path=path, file_name="sad_prediction_comparison_hp_" + str(test_num) + ".png", debug=False, \
                            title=f"batch_size: {batch_size}, learning_rate: {learning_rate}, hidden_size: {hidden_size}")
                test_num += 1
            
print(f"Total time: {time.time() - start_time:.2f} seconds, {training_time/60:.2f} minutes, {training_time/3600:.2f} hours")
print("\n----------------------------------------\n\n\n")
            
