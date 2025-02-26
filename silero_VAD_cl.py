import torch
import torchaudio
import numpy as np
import os
import load

labels_path = "FSC_P4_Streams\\Transcripts\\SAD\\Dev"
wav_path_o = "FSC_P4_Streams\\Audio\\Streams\\Dev"

labels = []
X = []
fp_time = 0
fn_time = 0
y_speech_time = 0
y_nonspeech_time = 0
loader = load.LoadAudio()

def preprocess_audio(waveform, original_sr, target_sr=16000):
    """Preprocess audio for Silero VAD."""
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        waveform = resampler(waveform)
    
    # Normalize audio
    waveform = waveform / torch.max(torch.abs(waveform))
    
    return waveform

def evaluate_vad(wav_path, label_path, vad_model, frame_duration=0.01):
    """Evaluate VAD performance with detailed logging."""
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = preprocess_audio(waveform, sample_rate)
    
    # Get VAD predictions
    speech_timestamps = get_speech_timestamps(
        waveform, 
        vad_model,
        sampling_rate=16000,
        threshold=0.5,  # You might want to tune this
        min_speech_duration_ms=250,  # And this
        min_silence_duration_ms=100  # And this
    )
    
    # Convert to frame-level predictions
    num_frames = int(waveform.shape[1] / 16000 / frame_duration)
    vad_labels = np.zeros(num_frames, dtype=int)
    
    for segment in speech_timestamps:
        start_idx = int(segment['start'] / (16000 * frame_duration))
        end_idx = min(int(segment['end'] / (16000 * frame_duration)), num_frames)
        vad_labels[start_idx:end_idx] = 1
    
    # Load ground truth labels
    with open(label_path, 'r') as f:
        # Implement your label loading logic here
        # Make sure labels and vad_labels have the same length
        pass
    
    # Compute metrics with sanity checks
    if len(labels) != len(vad_labels):
        print(f"WARNING: Length mismatch - Labels: {len(labels)}, VAD: {len(vad_labels)}")
        # Truncate to shorter length
        min_len = min(len(labels), len(vad_labels))
        labels = labels[:min_len]
        vad_labels = vad_labels[:min_len]
    
    # Print debug info
    print(f"File: {os.path.basename(wav_path)}")
    print(f"Total frames: {len(vad_labels)}")
    print(f"Speech frames (VAD): {np.sum(vad_labels)}")
    print(f"Speech frames (Ground truth): {np.sum(labels)}")
    
    return labels, vad_labels

# Usage example
for filename in sorted(os.listdir(labels_path)):
    wav_path = os.path.join(wav_path_o, filename.replace(".txt", ".wav"))
    label_path = os.path.join(labels_path, filename)
    
    labels, vad_labels = evaluate_vad(wav_path, label_path, vad_model)
    
    # Update metrics
    fp_time += np.count_nonzero((labels == 0) & (vad_labels == 1))
    fn_time += np.count_nonzero((labels == 1) & (vad_labels == 0))
    y_speech_time += (labels == 1).sum()
    y_nonspeech_time += (labels == 0).sum()