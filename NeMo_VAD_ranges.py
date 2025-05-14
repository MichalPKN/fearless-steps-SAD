import torch
import nemo.collections.asr as nemo_asr
import librosa
import numpy as np
import os
import load

loader = load.LoadAudio()

device = "cuda" if torch.cuda.is_available() else "cpu"

vad_model = nemo_asr.models.VoiceActivityDetectionModel.from_pretrained(model_name="vad_marblenet")
# vad_model.spec_augmentation = None  # Disable augmentation
# vad_model.eval()
# vad_model = vad_model.to(device)

def classify_audio(audio_path):
    wav_file = audio_path
    speech_timestamps = vad_model.detect_speech(wav_file)

    audio, sr = librosa.load(wav_file, sr=16000)
    total_duration = len(audio) / sr

    frame_times = np.arange(0, total_duration, 0.01)  # Time steps: 0.01s (10ms)
    frame_labels = np.zeros_like(frame_times, dtype=int)  # Default: Silence (0)

    for segment in speech_timestamps:
        start, end = segment["start"], segment["end"]
        frame_labels[(frame_times >= start) & (frame_times < end)] = 1  # speech
    print(end)
    print(f"frame_labels: {frame_labels.shape}")
    print(f"num of ones: {np.count_nonzero(frame_labels)}")
    print(f"num of zeros: {np.count_nonzero(frame_labels == 0)}")

    for i in range(1000, 1050):
        print(f"Time {frame_times[i]:.2f}s: {'Speech' if frame_labels[i] == 1 else 'Silence'}")

    return frame_labels[:60000]

audio_path = "FSC_P4_Streams/Audio/Streams/Dev/fsc_p4_dev_001.wav"
vad_results = classify_audio(audio_path)

print(f"vad_results: {vad_results}")
print(f"classified: {vad_results.shape}")

label_path = "FSC_P4_Streams/Transcripts/SAD/Dev/fsc_p4_dev_001.txt"

labels, num_of_1s, num_of_0s = loader.add_labels(label_path, vad_results)
labels = labels.squeeze()[:60000]
print("added labels")
print(f"shape: {labels.shape}")
fp_time = np.count_nonzero((labels == 0) & (vad_results == 1))
fn_time = np.count_nonzero((labels == 1) & (vad_results == 0))
y_speech_time = (labels == 1).sum()
y_nonspeech_time = (labels == 0).sum()

pfp = fp_time / y_nonspeech_time # false alarm
pfn = fn_time / y_speech_time # miss
dcf = 0.75 * pfn + 0.25 * pfp

print(f"dcf: {dcf}")
print(f"False Positives (FP): {fp_time}")
print(f"False Negatives (FN): {fn_time}")