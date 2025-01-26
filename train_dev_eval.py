import torch
from utils import smooth_outputs_rnn


def train_model(sad_model, optimizer, criterion, X_size, criteria, epochs, device, dataloader, losses, epoch):
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
    print()
    print(f"total features predicted: {total_predictions}, num of batches: {i}, mask sum in base seq: {mask[0].sum()}, mask sum in last sequence: {mask[len(mask)-1].sum()}")
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/X_size:.4f}, Accuracy: {train_accuracy*100:.2f}, DCF: {dcf*100:.4f}")

    return losses, dcf

def validate_model(sad_model, dataloader_dev, criteria, device):
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
        
        return dev_accuracy, dev_dcf, best_smooth_window_dcf, top_smooth_window

def evaluate_model(best_model, dataloader_val, criteria, device, best_smooth_window):
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
        eval_accuracy = correct_predictions / total_predictions
        pfp = fp_time / y_nonspeech_time # false alarm
        pfn = fn_time / y_speech_time # miss
        eval_dcf = 0.75 * pfn + 0.25 * pfp

        pfp_smooth = fp_time_smooth / y_nonspeech_time # false alarm
        pfn_smooth = fn_time_smooth / y_speech_time # miss
        eval_dcf_smooth = 0.75 * pfn_smooth + 0.25 * pfp_smooth
        
        print(f'Validation Accuracy: {eval_accuracy*100:.2f}, Validation DCF: {eval_dcf*100:.4f}, Validation DCF smooth {best_smooth_window}: {eval_dcf_smooth*100:.4f}')
        print() 
        
        return eval_accuracy, eval_dcf, eval_dcf_smooth, toshow_y, toshow_preds, toshow_outputs, toshow_additional

        
#         # dev
#         sad_model.eval()
        
#         with torch.no_grad():
#             best_val = 100
#             smooth_window = [5, 10, 20, 40]
#             correct_predictions = 0
#             total_predictions = 0
#             fp_time = 0
#             fn_time = 0
#             y_speech_time = 0
#             y_nonspeech_time = 0
#             fp_time_smooth = [0 for asd in range(len(smooth_window))]
#             fn_time_smooth = [0 for asd in range(len(smooth_window))]
#             for batch_x, batch_y, mask in dataloader_dev:
#                 batch_x, batch_y, mask  = batch_x.to(device), batch_y.to(device), mask.to(device)
#                 outputs = sad_model(batch_x)
#                 outputs = torch.sigmoid(outputs)
#                 preds = (outputs >= criteria).float()
#                 correct_predictions += ((preds == batch_y).float() * mask).sum().item()
#                 total_predictions += mask.sum().item()
#                 fp_time += (((preds == 1) & (batch_y == 0)) * mask).sum().item()
#                 fn_time += (((preds == 0) & (batch_y == 1)) * mask).sum().item()
#                 y_speech_time += (batch_y * mask).sum().item()
#                 y_nonspeech_time += ((batch_y == 0) * mask).sum().item()                                                            
                
#                 # smoothing:
#                 for window_idx, window in enumerate(smooth_window):
#                     smooth_preds = smooth_outputs_rnn(preds, avg_frames=window, criteria=criteria)
#                     fp_time_smooth[window_idx] += (((smooth_preds == 1) & (batch_y == 0)) * mask).sum().item()
#                     fn_time_smooth[window_idx] += (((smooth_preds == 0) & (batch_y == 1)) * mask).sum().item()
#             dev_accuracy = correct_predictions / total_predictions
#             pfp = fp_time / y_nonspeech_time # false alarm
#             pfn = fn_time / y_speech_time # miss
#             dev_dcf = 0.75 * pfn + 0.25 * pfp
            
#             print(f'Dev Accuracy: {dev_accuracy*100:.2f}, Dev DCF: {dev_dcf*100:.4f}')
#             best_smooth_window_dcf = 101
#             for window_idx, window in enumerate(smooth_window):
#                 pfp_smooth = fp_time_smooth[window_idx] / y_nonspeech_time
#                 pfn_smooth = fn_time_smooth[window_idx] / y_speech_time
#                 dev_dcf_smooth = 0.75 * pfn_smooth + 0.25 * pfp_smooth
#                 print(f'Dev DCF smooth {window}: {dev_dcf_smooth*100:.4f}', end=", ")
#                 if dev_dcf_smooth < best_smooth_window_dcf:
#                     best_smooth_window_dcf = dev_dcf_smooth
#                     top_smooth_window = window
#             print()
    
#             if dev_dcf < best_val:
#                 best_val = dev_dcf
#                 dcf_train = dcf
#                 dcf_dev = dev_dcf
#                 dcf_dev_smooth = best_smooth_window_dcf
#                 best_smooth_window = top_smooth_window
#                 torch.save(sad_model, model_path)
                
                
#     print("\nVALIDATION")
    
#     best_model = torch.load(model_path)
    
#     X_val, Y_val = split_file(X_val_loaded, Y_val_loaded, batch_size=audio_size, shuffle=False)
#     dataset_val = SADDataset(X_val, Y_val, max_len=dataset.max_len, overlap=overlap)
#     print(f"X_val length: {len(X_val)}")
#     print(f"X_val[0] shape: {X_val[0].shape}")
#     dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
    
#     # eval
#     best_model.eval()
#     with torch.no_grad():
#         correct_predictions = 0
#         total_predictions = 0
#         fp_time = 0
#         fn_time = 0
#         y_speech_time = 0
#         y_nonspeech_time = 0
#         fp_time_smooth = 0
#         fn_time_smooth = 0
#         i = 0
#         for batch_x, batch_y, mask in dataloader_val:
#             batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
#             outputs = best_model(batch_x)
#             outputs = torch.sigmoid(outputs)
#             preds = (outputs >= criteria).float()
#             correct_predictions += ((preds == batch_y).float() * mask).sum().item()
#             total_predictions += mask.sum().item()
#             fp_time += (((preds == 1) & (batch_y == 0)) * mask).sum().item()
#             fn_time += (((preds == 0) & (batch_y == 1)) * mask).sum().item()
#             y_speech_time += (batch_y * mask).sum().item()
#             y_nonspeech_time += ((batch_y == 0) * mask).sum().item()
            
#             # smoothing:
#             smooth_preds = smooth_outputs_rnn(preds, avg_frames=best_smooth_window, criteria=criteria)
#             fp_time_smooth += (((smooth_preds == 1) & (batch_y == 0)) * mask).sum().item()
#             fn_time_smooth += (((smooth_preds == 0) & (batch_y == 1)) * mask).sum().item()
            
#             if i == 5:
#                 toshow_y = batch_y[0]
#                 toshow_preds = preds[0]
#                 toshow_outputs = outputs[0]
#                 toshow_additional = smooth_preds[0]
#             i += 1
#         eval_accuracy = correct_predictions / total_predictions
#         pfp = fp_time / y_nonspeech_time # false alarm
#         pfn = fn_time / y_speech_time # miss
#         eval_dcf = 0.75 * pfn + 0.25 * pfp

#         pfp_smooth = fp_time_smooth / y_nonspeech_time # false alarm
#         pfn_smooth = fn_time_smooth / y_speech_time # miss
#         eval_dcf_smooth = 0.75 * pfn_smooth + 0.25 * pfp_smooth
        
#         print(f'Validation Accuracy: {eval_accuracy*100:.2f}, Validation DCF: {eval_dcf*100:.4f}, Validation DCF smooth {best_smooth_window}: {eval_dcf_smooth*100:.4f}')
#         print()                        
        
#     print("finished training model")
#     training_time = time.time() - start_time - load_time
#     print(f"Training completed in {training_time:.2f} seconds, {training_time/60:.2f} minutes, {training_time/3600:.2f} hours")
#     print(f"losses: {losses}")
    
#     torch.cuda.empty_cache()
    
#     if debug:
#         path = os.path.join(datadir_path, "plots_rnn")
#     else:
#         path = "/storage/brno2/home/miapp/fearless-steps-SAD/fearless-steps-SAD/plots_rnn"
    
#     plot_result(toshow_y.cpu().numpy(), toshow_preds.cpu().numpy(), toshow_outputs.cpu().detach().numpy(), toshow_additional.cpu().detach().numpy(), \
#                 path=path, file_name="sad_prediction_comparison_hp_" + str(test_num) + ".png", debug=False, \
#                 title=f"batch_size: {batch_size}, learning_rate: {learning_rate}, hidden_size: {hidden_size}")
#     test_num += 1
#     print("\n----------------------------------------\n\n\n")

# print("results:")
# print(f"parameters: batch_size: {batch_size}, audio_size: {audio_size}, overlap: {overlap}, hidden_size: {hidden_size}, num_layers: {num_layers}, learning_rate: {learning_rate}")
# print("dcf_train\tdcf_dev\tdcf_dev_smooth\tdcf_eval\tdcf_eval_smooth\tbest_smooth_window")
# print(f"{dcf_train:.4f}\t{dcf_dev:.4f}\t{dcf_dev_smooth:.4f}\t{eval_dcf:.4f}\t{eval_dcf_smooth:.4f}\t{best_smooth_window:.4f}")