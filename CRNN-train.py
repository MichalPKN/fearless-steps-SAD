print("starting code")
import load
import model_architectures.model_crnn as model_sad
import numpy as np
import os
import argparse
import time
import gc
from utils import plot_result, SADDataset, split_file, check_gradients, smooth_outputs_rnn

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
input_size = 30
hidden_size = 256
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
X_loaded_all, audio_info, Y_loaded_all = data_loader.load_all(train_path, train_labels)
if debug:
    X_loaded_all = [x[:20830] for x in X_loaded_all]
    Y_loaded_all = [y[:20789] for y in Y_loaded_all]

# train test split
print(f"num of data before train dev split: {len(X_loaded_all)}")
#dev_idxs = [1] if debug else [5, 18, 27, 43, 68, 91, 112, 129]
dev_idxs = [1] if debug else [5, 12, 18, 27, 43, 56, 68, 91, 112, 129]
train_idxs = [i for i in range(len(X_loaded_all)) if i not in dev_idxs]
X_dev_loaded = [X_loaded_all[i] for i in dev_idxs]
Y_dev_loaded = [Y_loaded_all[i] for i in dev_idxs]
dev_files_info = [[audio_info[3][i] for i in dev_idxs], [audio_info[4][i] for i in dev_idxs]]
X_loaded = [X_loaded_all[i] for i in train_idxs]
Y_loaded = [Y_loaded_all[i] for i in train_idxs]
train_files_info = [[audio_info[j][i] for i in train_idxs] for j in [3, 4]]
print(f"dev data: {dev_idxs}")
print(f"num of trainig data: {len(X_loaded)}, num of dev data: {len(X_dev_loaded)}")
print("dev files:")
for i in range(len(dev_idxs)):
    print(f"{dev_files_info[0][i]}, {dev_files_info[1][i]}")
# print("train files:")
# for i in range(len(X_loaded)):
#     print(f"{train_files_info[0][i]}, {train_files_info[1][i]}")



# eval data
X_val_loaded, val_info, Y_val_loaded = data_loader.load_all(dev_path, dev_labels)
if debug:
    X_val_loaded = [x[:20534] for x in X_val_loaded]
    Y_val_loaded = [y[:20567] for y in Y_val_loaded]
print(f"num of eval data: {len(X_val_loaded)}")

del X_loaded_all, Y_loaded_all
gc.collect()

# training
test_num = 1
for f_test in range(1):
    for batch_size, audio_size in [[10, 1000], [30, 10000]]:
        print(f"\nsplitting, padding, etc. all data to batch size {batch_size}, audio size {audio_size}")
        X, Y = split_file(X_loaded, Y_loaded, batch_size=audio_size, shuffle=False)
        dataset = SADDataset(X, Y) 
        print(f"max size: {dataset.max_len}")
        print(f"X length: {len(X)}")
        print(f"X[0] shape: {X[0].shape}")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_batches) # maybe shuffle True        
                    
        X_dev, Y_dev = split_file(X_dev_loaded, Y_dev_loaded, batch_size=audio_size, shuffle=False)
        dataset_dev = SADDataset(X_dev, Y_dev, max_len=dataset.max_len)
        dataloader_dev = DataLoader(dataset_dev, batch_size=1, shuffle=False)
        print(f"X_dev length: {len(X_dev)}")
        print(f"X_dev[0] shape: {X_dev[0].shape}")
        
        for num_layers in [2]:#[2, 4]:
            for filter_num in [16, 32]:
                for learning_rate in [0.0001, 0.00001]: #[0.001, 0.0001, 0.00001]:
                    print(f"\n\nbatch_size: {batch_size}, sequence_size: {audio_size}, learning_rate: {learning_rate}, hidden_size: {hidden_size}, num_layers: {num_layers}")
                    #print(f"X length: {len(X)}, X_dev length {len(X_dev)}")
                    
                    # model
                    sad_model = model_sad.SADModel(input_size).to(device)
                    if torch.cuda.device_count() > 1:
                        print(f"Using {torch.cuda.device_count()} GPUs")
                        sad_model = torch.nn.DataParallel(sad_model)
                    print("Model's state_dict:")
                    for param_tensor in sad_model.state_dict():
                        print(param_tensor, "\t", sad_model.state_dict()[param_tensor].size())
                        
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
                    
                    best_val = 100
                    model_path = os.path.join(datadir_path, "models", f"model_sad_{batch_size}-{audio_size}_{learning_rate}_{hidden_size}_{num_layers}.pt")

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
                            
                            running_loss += loss.item()
                            # debug:
                            if epoch == 0 and (i < 20 or (i > 150 and i < 170)):
                                print(f"i: {i}, Loss: {running_loss/(i+1):.4f}, running_loss: {running_loss:.4f}")
                            if epoch == 0 and i % (len(X) // 10) == 0:
                                train_accuracy = correct_predictions / total_predictions
                                pfp = fp_time / (y_nonspeech_time + 0.0001) # false alarm
                                pfn = fn_time / (y_speech_time + 0.0001) # miss
                                dcf = 0.75 * pfn + 0.25 * pfp
                                print(f'first epoch, Loss: {loss:.4f}, Accuracy: {train_accuracy*100:.2f}, DCF: {dcf*100:.2f}')
                                print("size:", len(preds), "fp_time:", preds.sum(), "ones actual:", batch_y.sum(), "mean:", outputs.mean())
                                print("-----------------------------")
                            i += 1
                        train_accuracy = correct_predictions / total_predictions
                        pfp = fp_time / y_nonspeech_time # false alarm
                        pfn = fn_time / y_speech_time # miss
                        dcf = 0.75 * pfn + 0.25 * pfp
                        losses[epoch] = running_loss/len(X)
                        print()
                        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(X):.4f}, Accuracy: {train_accuracy*100:.2f}, DCF: {dcf*100:.4f}")
                        
                        # dev
                        sad_model.eval()
                        with torch.no_grad():
                            smooth_window = [5, 10, 20, 40]
                            correct_predictions = 0
                            total_predictions = 0
                            fp_time = 0
                            fn_time = 0
                            y_speech_time = 0
                            y_nonspeech_time = 0
                            fp_time_smooth = [0 for asd in range(len(smooth_window))]
                            fn_time_smooth = [0 for asd in range(len(smooth_window))]
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
                                    fp_time_smooth[window_idx] += (((smooth_preds == 1) & (batch_y == 0)) * mask).sum().item()
                                    fn_time_smooth[window_idx] += (((smooth_preds == 0) & (batch_y == 1)) * mask).sum().item()
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
                                    top_smooth_window = window
                            print()
                    
                            if dev_dcf < best_val:
                                best_val = dev_dcf
                                dcf_train = dcf
                                dcf_dev = dev_dcf
                                dcf_dev_smooth = best_smooth_window_dcf
                                best_smooth_window = top_smooth_window
                                torch.save(sad_model, model_path)
                                
                                
                    print("\nVALIDATION")
                    
                    best_model = torch.load(model_path)
                    
                    X_val, Y_val = split_file(X_val_loaded, Y_val_loaded, batch_size=audio_size, shuffle=False)
                    dataset_val = SADDataset(X_val, Y_val, max_len=dataset.max_len)
                    print(f"X_val length: {len(X_val)}")
                    print(f"X_val[0] shape: {X_val[0].shape}")
                    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
                    
                    # eval
                    best_model.eval()
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
                            outputs = best_model(batch_x)
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
                            fp_time_smooth += (((smooth_preds == 1) & (batch_y == 0)) * mask).sum().item()
                            fn_time_smooth += (((smooth_preds == 0) & (batch_y == 1)) * mask).sum().item()
                            
                            if i == 5:
                                toshow_y = batch_y[0]
                                toshow_preds = preds[0]
                                toshow_outputs = outputs[0]
                                toshow_additional = smooth_preds[0]
                            i += 1
                        dev_accuracy = correct_predictions / total_predictions
                        pfp = fp_time / y_nonspeech_time # false alarm
                        pfn = fn_time / y_speech_time # miss
                        dev_dcf = 0.75 * pfn + 0.25 * pfp

                        pfp_smooth = fp_time_smooth / y_nonspeech_time # false alarm
                        pfn_smooth = fn_time_smooth / y_speech_time # miss
                        dev_dcf_smooth = 0.75 * pfn_smooth + 0.25 * pfp_smooth
                        
                        print(f'Validation Accuracy: {dev_accuracy*100:.2f}, Validation DCF: {dev_dcf*100:.4f}, Validation DCF smooth {best_smooth_window}: {dev_dcf_smooth*100:.4f}')
                        print()                        
                        
                    print("finished training model")
                    training_time = time.time() - start_time - load_time
                    print(f"Training completed in {training_time:.2f} seconds, {training_time/60:.2f} minutes, {training_time/3600:.2f} hours")
                    print(f"losses: {losses}")
                    
                    torch.cuda.empty_cache()
                    
                    if debug:
                        path = os.path.join(datadir_path, "plots_rnn")
                    else:
                        path = "/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD/plots_rnn"
                    
                    plot_result(toshow_y.cpu().numpy(), toshow_preds.cpu().numpy(), toshow_outputs.cpu().detach().numpy(), toshow_additional.cpu().detach().numpy(), \
                                path=path, file_name="sad_prediction_comparison_hp_" + str(test_num) + ".png", debug=False, \
                                title=f"batch_size: {batch_size}, learning_rate: {learning_rate}, hidden_size: {hidden_size}")
                    test_num += 1
                    print("\n----------------------------------------\n\n\n")
                    
        del sad_model, best_model, dataset, dataset_dev, dataset_val, dataloader, dataloader_dev, dataloader_val
        del X, Y, X_dev, Y_dev, X_val, Y_val
        gc.collect()
        
print(f"Total time: {time.time() - start_time:.2f} seconds, {training_time/60:.2f} minutes, {training_time/3600:.2f} hours")
print("\n----------------------------------------\n\n\n")
            
