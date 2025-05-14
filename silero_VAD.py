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

    speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=sample_rate)

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
