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

# Function to process and classify a wav file
def classify_audio(audio_path):
    # # Load audio (librosa loads as float32 by default)
    # audio, sr = librosa.load(audio_path, sr=16000)  # Ensure 16kHz sample rate

    # # Convert to tensor format expected by NeMo
    # audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Add batch dim

    # # Run VAD model
    # logits = vad_model.forward(input_signal=audio_tensor, input_signal_length=torch.tensor([audio.shape[0]]))

    # # Convert logits to probabilities
    # probs = torch.sigmoid(logits)

    # # Classification: 1 means speech, 0 means silence/no speech
    # vad_prediction = (probs > 0.5).int()

    # return vad_prediction, probs
    
    audio, sr = librosa.load(audio_path, sr=16000)

    frame_len = int(0.25 * 16000)  # 25ms (400 samples)
    frame_step = int(0.010 * 16000)  # 10ms (160 samples)

    frames = [audio[i: i + frame_len] for i in range(0, len(audio) - frame_len, frame_step)]
    frames = torch.tensor(np.array(frames))
    print(f"frames: {frames.shape}")
    # frames = frames.unsqueeze(1)
    # print(f"frames: {frames.shape}")

    # Run VAD model
    with torch.no_grad():
        predictions = vad_model.forward(input_signal=frames, input_signal_length=torch.tensor([frame_len] * len(frames)))

    # Convert to binary speech/non-speech
    speech_probabilities = torch.sigmoid(predictions).squeeze().cpu().numpy()
    speech_detected = speech_probabilities > 0.5  # Threshold
    
    return speech_detected, speech_probabilities

# train
audio_file = "fsc_p4_dev_001.wav"
audio_path = os.join(dev_path, audio_file)
vad_results, _ = classify_audio(audio_path)

# Print results
print(f"vad_results: {vad_results[:400]}")
print(f"classified: {vad_results.shape}")  # List of True/False for each 10ms segment

label_file = "fsc_p4_dev_001.txt"
label_path = os.join(dev_labels, label_file)

#label_path = os.path.join(labels_path, labels_path)
labels, num_of_1s, num_of_0s = loader.add_labels(label_path, vad_results)
labels = labels.squeeze()
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