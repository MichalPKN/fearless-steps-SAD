print("starting code")
import load
import model_architectures.model_nn as model_sad
import numpy as np
import os
import argparse
import time
import gc
from utils import plot_result, SADDataset, split_file, check_gradients, smooth_outputs_rnn
from train_dev_eval import train_model, validate_model, evaluate_model
import nemo.collections.asr as nemo_asr
import librosa
from utils import smooth_outputs_rnn

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

loader = load.LoadAudio()

vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="vad_marblenet")
vad_model.spec_augmentation = None  # Disable augmentation
vad_model.eval()


def classify_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    frame_len = 400  # 25ms (400 samples)
    frame_step = 160  # 10ms (160 samples)

    frames = [audio[i: i + frame_len] for i in range(0, len(audio) - frame_len, frame_step)]
    frames = torch.tensor(np.array(frames), dtype=torch.float32)
    print(f"frames: {frames.shape}")
    frames = frames
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

fp_time = 0
fn_time = 0
y_speech_time = 0
y_nonspeech_time = 0
accuracy = 0
total_time = 0
dev_files = os.listdir(dev_path).sort()
for audio_file in dev_files:
    audio_path = os.join(dev_path, audio_file)
    
    vad_results, _ = classify_audio(audio_path)

    # Print results
    print(f"vad_results: {vad_results[:+100]}")
    print(f"classified: {vad_results.shape}")  # List of True/False for each 10ms segment

    label_file = audio_file.split(".")[0] + ".txt"
    label_path = os.join(dev_labels, label_file)

    #label_path = os.path.join(labels_path, labels_path)
    labels, num_of_1s, num_of_0s = loader.add_labels(label_path, vad_results)
    labels = labels.squeeze()
    
    if len(labels) < len(vad_results):
        vad_results = vad_results[:len(labels)]
    elif len(labels) > len(vad_results):
        labels = labels[:len(vad_results)]
        
    print("added labels")
    print(f"shape: {labels.shape}")
    fp_time += np.count_nonzero((labels == 0) & (vad_results == 1))
    fn_time += np.count_nonzero((labels == 1) & (vad_results == 0))
    y_speech_time += np.count_nonzero((labels == 1))
    y_nonspeech_time += np.count_nonzero((labels == 0))
    
    accuracy += (labels == vad_results).sum()
    total_time += len(labels)

pfp = fp_time / y_nonspeech_time # false alarm
pfn = fn_time / y_speech_time # miss
dcf = 0.75 * pfn + 0.25 * pfp

print(f"dcf: {dcf}")
print(f"False Positives (FP): {fp_time}")
print(f"False Negatives (FN): {fn_time}")

accuracy = accuracy / total_time
print(f"accuracy: {accuracy}")