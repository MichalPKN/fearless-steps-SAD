#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


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
    plt.savefig(path + file_name)
    if debug:
        plt.show()


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