import torch
import torchaudio
import load
import numpy as np
import os
from utils import smooth_outputs_rnn

# model
vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=True)
(get_speech_timestamps, _, read_audio, _, _) = utils

labels_path = "FSC_P4_Streams\\Transcripts\\SAD\\Dev"
wav_path_o = "FSC_P4_Streams\\Audio\\Streams\\Dev"

labels = []
X = []
fp_time = 0
fn_time = 0
y_speech_time = 0
y_nonspeech_time = 0
correct_count = 0
y_time = 0
loader = load.LoadAudio()
print(f"Loading labels from {labels_path}")
for i, filename in enumerate(sorted(os.listdir(labels_path))):

    wav_path = os.path.join(wav_path_o, filename.replace(".txt", ".wav"))
    sample_rate = 16000
    waveform = read_audio(wav_path, sampling_rate=sample_rate)
    print(sample_rate)  

    # Apply VAD
    speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=sample_rate)

    # Convert to 10ms frame-level labels
    frame_duration = 0.01  # 10ms
    num_frames = int(waveform.shape[0] / sample_rate / frame_duration)
    vad_labels = np.zeros(num_frames, dtype=int)  # Initialize all as non-speech

    for segment in speech_timestamps:
        start_idx = int(segment['start'] / (sample_rate * frame_duration))
        end_idx = min(int(segment['end'] / (sample_rate * frame_duration)), num_frames)
        vad_labels[start_idx:end_idx] = 1
    #vad labels to float
    vad_labels = vad_labels.astype(float)
    print("finished classification")

    print(f"shape: {vad_labels.shape}")
    print(f"speech: {np.count_nonzero(vad_labels)}")

    label_path = os.path.join(labels_path, filename)
    labels, num_of_1s, num_of_0s = loader.add_labels(label_path, vad_labels)
    labels = labels.squeeze()
    print("added labels")
    print(f"shape: {labels.shape}")
    print(labels[90100:90400])
    print(vad_labels[90100:90400])
    fp_time += np.count_nonzero((labels == 0) & (vad_labels == 1))
    fn_time += np.count_nonzero((labels == 1) & (vad_labels == 0))
    y_speech_time += (labels == 1).sum()
    y_nonspeech_time += (labels == 0).sum()
    correct_count += np.count_nonzero(labels == vad_labels)
    y_time += len(labels)
    break

pfp = fp_time / y_nonspeech_time # false alarm
pfn = fn_time / y_speech_time # miss
dcf = 0.75 * pfn + 0.25 * pfp
accuracy = correct_count / y_time

print(f"False Positives (FP): {fp_time}")
print(f"False Negatives (FN): {fn_time}")
print(f"Total Speech Frames: {y_speech_time}")
print(f"Total Non-Speech Frames: {y_nonspeech_time}")
print(f"PFP: {pfp}, PFN: {pfn}")

print(f"DCF: {dcf*100:.4f}%")
print(f"Accuracy: {accuracy*100:.4f}%")

import matplotlib.pyplot as plt


time_axis = np.arange(len(labels)) * 0.01

plt.figure(figsize=(12, 6))

plt.plot(time_axis, labels, label="Actual", color="black", linestyle="dotted", alpha=0.7)

plt.plot(time_axis, vad_labels, label="Predicted", color="blue", alpha=0.7)

#plt.plot(time_axis, speech_probs[:, 0], label="Speech Probability", color="red", alpha=0.5)

plt.xlabel("Time (seconds)")
plt.ylabel("Speech Activity")
plt.title("Actual vs Predicted Speech Activity")
plt.legend()
plt.grid()

plt.savefig("vad_results.png", dpi=300, bbox_inches="tight")

# import torch
# import torchaudio
# import numpy as np
# from typing import List, Dict
# import load

# def improve_vad_detection(wav_path: str, 
#                          threshold: float = 0.5,  # Increased threshold
#                          min_speech_duration_ms: int = 400,  # Increased minimum speech duration
#                          min_silence_duration_ms: int = 300,  # Increased minimum silence duration
#                          window_size_samples: int = 1536):
#     """
#     Enhanced VAD detection with more conservative speech detection parameters
#     """
#     # Load model
#     vad_model, utils = torch.hub.load('snakers4/silero-vad', 
#                                     'silero_vad', 
#                                     force_reload=True)
#     (get_speech_timestamps, _, _, _, _) = utils
    
#     # Load audio
#     waveform, sample_rate = torchaudio.load(wav_path)
    
#     # Get speech timestamps with more conservative parameters
#     speech_timestamps = get_speech_timestamps(
#         waveform,
#         vad_model,
#         sampling_rate=sample_rate,
#         threshold=threshold,
#         min_speech_duration_ms=min_speech_duration_ms,
#         min_silence_duration_ms=min_silence_duration_ms,
#         window_size_samples=window_size_samples
#     )
    
#     # Convert to frame-level labels
#     frame_duration = 0.01  # 10ms
#     num_frames = int(waveform.shape[1] / sample_rate / frame_duration)
#     vad_labels = np.zeros(num_frames, dtype=int)
    
#     # Apply labels with minimal smoothing
#     smooth_window = int(20 / frame_duration)  # Reduced smoothing window to 20ms
    
#     for segment in speech_timestamps:
#         start_idx = int(segment['start'] / (sample_rate * frame_duration))
#         end_idx = int(segment['end'] / (sample_rate * frame_duration))
        
#         # Add smaller context window
#         start_idx = max(0, start_idx - smooth_window)
#         end_idx = min(num_frames, end_idx + smooth_window)
        
#         vad_labels[start_idx:end_idx] = 1
    
#     # Add post-processing to remove isolated speech segments
#     min_segment_frames = int(200 / frame_duration)  # 200ms minimum segment
#     labels_processed = remove_short_segments(vad_labels, min_segment_frames)
    
#     return labels_processed, sample_rate

# def remove_short_segments(labels: np.ndarray, min_length: int) -> np.ndarray:
#     """
#     Remove speech segments that are too short
#     """
#     processed = labels.copy()
    
#     # Find contiguous regions
#     changes = np.diff(np.concatenate(([0], processed, [0])))
#     starts = np.where(changes == 1)[0]
#     ends = np.where(changes == -1)[0]
    
#     # Remove short segments
#     for start, end in zip(starts, ends):
#         if end - start < min_length:
#             processed[start:end] = 0
            
#     return processed

# def evaluate_vad(vad_labels: np.ndarray, 
#                 ground_truth: np.ndarray,
#                 verbose: bool = True) -> Dict[str, float]:
#     """
#     Evaluate VAD performance with detailed metrics
#     """
#     fp_time = np.count_nonzero((ground_truth == 0) & (vad_labels == 1))
#     fn_time = np.count_nonzero((ground_truth == 1) & (vad_labels == 0))
#     y_speech_time = (ground_truth == 1).sum()
#     y_nonspeech_time = (ground_truth == 0).sum()
    
#     pfp = fp_time / y_nonspeech_time if y_nonspeech_time > 0 else 0
#     pfn = fn_time / y_speech_time if y_speech_time > 0 else 0
#     dcf = 0.75 * pfn + 0.25 * pfp
    
#     metrics = {
#         'dcf': dcf * 100,
#         'pfp': pfp * 100,
#         'pfn': pfn * 100,
#         'false_positives': fp_time,
#         'false_negatives': fn_time,
#         'speech_frames': y_speech_time,
#         'nonspeech_frames': y_nonspeech_time
#     }
    
#     if verbose:
#         print(f"\nVAD Evaluation Metrics:")
#         print(f"DCF: {metrics['dcf']:.2f}%")
#         print(f"False Alarm Rate (PFP): {metrics['pfp']:.2f}%")
#         print(f"Miss Rate (PFN): {metrics['pfn']:.2f}%")
#         print(f"False Positives: {metrics['false_positives']}")
#         print(f"False Negatives: {metrics['false_negatives']}")
#         print(f"Total Speech Frames: {metrics['speech_frames']}")
#         print(f"Total Non-Speech Frames: {metrics['nonspeech_frames']}")
    
#     return metrics

# # Example usage
# if __name__ == "__main__":
#     wav_path = "FSC_P4_Streams/Audio/Streams/Dev/fsc_p4_dev_001.wav"
    
#     # Try different parameter combinations with more conservative values
#     parameter_sets = [
#         {'threshold': 0.5, 'min_speech_duration_ms': 40, 'min_silence_duration_ms': 30},
#         {'threshold': 0.5, 'min_speech_duration_ms': 20, 'min_silence_duration_ms': 10},
#         {'threshold': 0.5, 'min_speech_duration_ms': 200, 'min_silence_duration_ms': 100}
#     ]
    
#     best_dcf = float('inf')
#     best_params = None
    
#     for params in parameter_sets:
#         print(f"\nTrying parameters: {params}")
#         vad_labels, sample_rate = improve_vad_detection(wav_path, **params)
        
#         # Load ground truth
#         loader = load.LoadAudio()
#         ground_truth, _, _ = loader.add_labels(
#             "FSC_P4_Streams/Transcripts/SAD/Dev/fsc_p4_dev_001.txt", 
#             vad_labels
#         )
#         ground_truth = ground_truth.squeeze()
        
#         # Evaluate
#         metrics = evaluate_vad(vad_labels, ground_truth)
        
#         if metrics['dcf'] < best_dcf:
#             best_dcf = metrics['dcf']
#             best_params = params
    
#     print(f"\nBest parameters found: {best_params}")
#     print(f"Best DCF: {best_dcf:.2f}%")