#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import os


def plot_result(y_actual, y_pred, processed_predictions=None, path="", file_name="sad_prediction_comparison.png", debug=False):    
    # Plotting in subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    # Plot the actual labels
    axs[0].plot(y_actual, label="Actual", color="green")
    axs[0].set_ylabel("Actual")
    axs[0].legend(loc="upper right")

    # Plot the raw and smoothed predictions
    axs[1].plot(y_pred, label="Predictions", color="blue", alpha=0.5)
    if processed_predictions is not None:
        axs[1].plot(processed_predictions, label="Outputs", color="red", linestyle='--')
    axs[1].set_ylabel("Predictions")
    axs[1].legend(loc="upper right")

    # Plot the difference (absolute error)
    difference = np.abs(y_actual - y_pred)
    axs[2].plot(difference, label="Absolute Difference", color="purple")
    axs[2].set_ylabel("Difference")
    axs[2].set_xlabel("Time Steps")
    axs[2].legend(loc="upper right")

    plt.suptitle("SAD Model Prediction vs Ground Truth (Zoomed in)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(path, file_name))
    if debug:
        plt.show()

class SADDataset(Dataset):
    def __init__(self, X, Y, max_len=None):
        self.X = X  # List of feature matrices
        self.Y = Y  # List of labels (0/1)
        #self.min_len = min_len or min([len(x) for x in X])
        self.max_len = max_len or max([len(x) for x in X])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        x_padded = F.pad(x, (0, 0, 0, self.max_len - len(x)))
        y_padded = F.pad(y, (0, 0, 0, self.max_len - len(y)))
        
        mask = torch.zeros_like(y_padded)
        mask[:len(x)] = 1
        return x_padded, y_padded, mask
        # x_cut = x[:self.min_len]
        # y_cut = y[:self.min_len]
        # return x_cut, y_cut


# def plot_result_plotly(y_actual, y_pred, processed_predictions=None, path="", file_name="sad_prediction_comparison.png", debug=False):    
#     # Plotting in subplots
#     fig = go.Figure()

#     # Add ground truth as a heatmap-like trace
#     fig.add_trace(go.Heatmap(
#         z=[y_actual.squeeze()],
#         colorscale=[[0, 'white'], [1, 'green']],
#         showscale=False,
#         name='Actual'
#     ))

#     # Add predictions as a second heatmap-like trace
#     fig.add_trace(go.Heatmap(
#         z=[processed_predictions.squeeze()],
#         colorscale=[[0, 'white'], [1, 'blue']],
#         showscale=False,
#         name='Predictions'
#     ))

#     # Adjust layout for clarity
#     fig.update_layout(
#         title="SAD Model Prediction vs Ground Truth",
#         xaxis_title="Time Steps",
#         yaxis_title="",
#         yaxis=dict(
#             showticklabels=False  # Hide y-axis labels to emphasize the binary coloring
#         ),
#         height=400
#     )

#     # Show and save the figure
#     fig.show()
#     fig.write_image(path + file_name)






# batch_x, batch_y, mask = batch_x.to(device), batch_y.to(device), mask.to(device)
                    
#                     optimizer.zero_grad()
                    
#                     # Forward
#                     outputs = sad_model(batch_x)
#                     #print(outputs.mean())
#                     loss = criterion(outputs, batch_y)
                    
#                     preds = (outputs >= criteria).float()
#                     correct_predictions += ((preds == batch_y).float() * mask).sum().item()
#                     total_predictions += mask.sum().item()
                    
#                     # plot_result(batch_y[0].cpu().numpy(), preds[0].cpu().numpy(), outputs[0].cpu().detach().numpy(), path=datadir_path, file_name="sad_prediction_comparison" + str(i) + ".png", debug=False)
#                     # i += 1
                    
#                     #print("predsum: ", preds.sum(), "batch_y sum: ", batch_y.sum())
                    
#                     #print(preds.shape)
#                     # Backward
#                     loss.backward()
#                     optimizer.step()
#                     fp_time += (((preds == 1) & (batch_y == 0)) * mask).sum().item()
#                     fn_time += (((preds == 0) & (batch_y == 1)) * mask).sum().item()
#                     y_speech_time += (batch_y * mask).sum().item()
#                     y_nonspeech_time += ((batch_y == 0) * mask).sum().item()
                    
#                     #print(fp_time, fn_time, y_speech_time, y_nonspeech_time)
#                     running_loss += loss.item()