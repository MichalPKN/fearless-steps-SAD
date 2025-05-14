import torch
import nemo.collections.asr as nemo_asr
import librosa
import numpy as np
import os
import load
from utils import smooth_outputs_rnn

loader = load.LoadAudio()

device = "cuda" if torch.cuda.is_available() else "cpu"

vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="vad_marblenet")
vad_model.spec_augmentation = None  # Disable augmentation
vad_model.eval()
vad_model = vad_model.to(device)

def classify_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    frame_len = 400  # 25ms (400 samples)
    frame_step = 160  # 10ms (160 samples)

    frames = [audio[i: i + frame_len] for i in range(0, len(audio) - frame_len, frame_step)]
    frames = torch.tensor(np.array(frames), dtype=torch.float32)
    print(f"frames: {frames.shape}")
    frames = frames[:60000]
    print(f"frames: {frames.shape}")
    frames = frames.to(device)
    print(f"frames: {frames.shape}")

    with torch.no_grad():
        predictions = vad_model.forward(input_signal=frames, input_signal_length=torch.tensor([frame_len] * len(frames)))
        #predictions = vad_model(frames)
    
    print(f"predictions: {predictions.shape}")

    speech_probabilities = torch.sigmoid(predictions).squeeze().cpu().numpy()
    speech_detected = speech_probabilities[:, 0] < 0.6  # Threshold
    speech_detected = speech_detected.reshape(1, -1, 1)
    speech_detected = torch.tensor(speech_detected, dtype=torch.float32)
    speech_detected = smooth_outputs_rnn(speech_detected, avg_frames=20, criteria=0.5)
    speech_detected = speech_detected.squeeze()
    
    print(f"speech_detected: {speech_detected.shape}")
    print(f"num of ones: {np.count_nonzero(speech_detected)}")
    print(f"num of zeros: {np.count_nonzero(speech_detected == 0)}")
    
    return speech_detected, speech_probabilities

audio_path = "FSC_P4_Streams/Audio/Streams/Dev/fsc_p4_dev_001.wav"
vad_results, speech_probs = classify_audio(audio_path)
print(f"speech_probs: {speech_probs.shape}")
print(f"speech_probs: {speech_probs[:10]}")

# Print results
print(f"vad_results: {vad_results}")
print(f"classified: {vad_results.shape}")

label_path = "FSC_P4_Streams/Transcripts/SAD/Dev/fsc_p4_dev_001.txt"

#label_path = os.path.join(labels_path, labels_path)
labels, num_of_1s, num_of_0s = loader.add_labels(label_path, vad_results)
labels = torch.from_numpy(labels.squeeze()[:60000])
print("added labels")
print(f"shape: {labels.shape}")
fp_time = np.count_nonzero((labels == 0) & (vad_results == 1))
fn_time = np.count_nonzero((labels == 1) & (vad_results == 0))
y_speech_time = (labels == 1).sum()
y_nonspeech_time = (labels == 0).sum()
accuracy = (labels == vad_results).sum() / len(labels)
print(f"accuracy: {accuracy}")

pfp = fp_time / y_nonspeech_time # false alarm
pfn = fn_time / y_speech_time # miss
dcf = 0.75 * pfn + 0.25 * pfp

print(f"dcf: {dcf}")
print(f"False Positives (FP): {fp_time}")
print(f"False Negatives (FN): {fn_time}")

import matplotlib.pyplot as plt


time_axis = np.arange(len(labels)) * 0.01

plt.figure(figsize=(12, 6))

plt.plot(time_axis, labels, label="Actual", color="black", linestyle="dotted", alpha=0.7)

plt.plot(time_axis, vad_results, label="Predicted", color="blue", alpha=0.7)

plt.plot(time_axis, speech_probs[:, 0], label="Speech Probability", color="red", alpha=0.5)

plt.xlabel("Time (seconds)")
plt.ylabel("Speech Activity")
plt.title("Actual vs Predicted Speech Activity")
plt.legend()
plt.grid()

plt.savefig("vad_results.png", dpi=300, bbox_inches="tight")