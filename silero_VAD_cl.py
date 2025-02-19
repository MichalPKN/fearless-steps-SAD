import torch
import torchaudio
import numpy as np
import os
from typing import Tuple, List, Dict

def load_labels(label_path: str, num_frames: int) -> np.ndarray:
    """
    Load labels from a text file and convert to frame-level labels.
    
    Args:
        label_path: Path to label file
        num_frames: Total number of frames expected
        
    Returns:
        numpy array of frame-level labels (0 for non-speech, 1 for speech)
    """
    labels = np.zeros(num_frames, dtype=int)
    
    with open(label_path, 'r') as f:
        for line in f:
            # Assuming label format is: "start_time end_time"
            start_time, end_time = map(float, line.strip().split())
            # Convert time to frame indices (assuming 10ms frames)
            start_frame = int(start_time * 100)  # 100 frames per second
            end_frame = int(end_time * 100)
            
            # Ensure we don't exceed array bounds
            end_frame = min(end_frame, num_frames)
            if start_frame < num_frames:
                labels[start_frame:end_frame] = 1
                
    return labels

def preprocess_audio(waveform: torch.Tensor, 
                    original_sr: int, 
                    target_sr: int = 16000) -> torch.Tensor:
    """
    Preprocess audio for Silero VAD.
    
    Args:
        waveform: Input audio waveform
        original_sr: Original sample rate
        target_sr: Target sample rate (default 16kHz for Silero VAD)
        
    Returns:
        Preprocessed waveform
    """
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        waveform = resampler(waveform)
    
    # Normalize audio
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    
    return waveform

def evaluate_vad(wav_path: str, 
                label_path: str, 
                vad_model: torch.nn.Module, 
                get_speech_timestamps: callable,
                frame_duration: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate VAD performance with detailed logging.
    
    Args:
        wav_path: Path to audio file
        label_path: Path to label file
        vad_model: Silero VAD model
        get_speech_timestamps: Timestamp extraction function from Silero
        frame_duration: Frame duration in seconds (default 0.01s = 10ms)
        
    Returns:
        Tuple of (ground truth labels, VAD predictions)
    """
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = preprocess_audio(waveform, sample_rate)
    
    # Get VAD predictions
    speech_timestamps = get_speech_timestamps(
        waveform, 
        vad_model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        return_seconds=False  # Return timestamps in samples
    )
    
    # Convert to frame-level predictions
    num_frames = int(waveform.shape[1] / 16000 / frame_duration)
    vad_labels = np.zeros(num_frames, dtype=int)
    
    for segment in speech_timestamps:
        start_idx = int(segment['start'] / (16000 * frame_duration))
        end_idx = min(int(segment['end'] / (16000 * frame_duration)), num_frames)
        vad_labels[start_idx:end_idx] = 1
    
    # Load ground truth labels
    labels = load_labels(label_path, num_frames)
    
    # Compute metrics with sanity checks
    if len(labels) != len(vad_labels):
        print(f"WARNING: Length mismatch - Labels: {len(labels)}, VAD: {len(vad_labels)}")
        # Truncate to shorter length
        min_len = min(len(labels), len(vad_labels))
        labels = labels[:min_len]
        vad_labels = vad_labels[:min_len]
    
    # Print debug info
    print(f"\nFile: {os.path.basename(wav_path)}")
    print(f"Total frames: {len(vad_labels)}")
    print(f"Speech frames (VAD): {np.sum(vad_labels)}")
    print(f"Speech frames (Ground truth): {np.sum(labels)}")
    
    return labels, vad_labels

def main():
    # Load model
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=True
    )
    (get_speech_timestamps, _, _, _, _) = utils
    
    # Paths
    labels_path = "FSC_P4_Streams/Transcripts/SAD/Dev"
    wav_path_o = "FSC_P4_Streams/Audio/Streams/Dev"
    
    # Initialize metrics
    fp_time = fn_time = y_speech_time = y_nonspeech_time = 0
    
    # Process all files
    for filename in sorted(os.listdir(labels_path)):
        if not filename.endswith('.txt'):
            continue
            
        wav_path = os.path.join(wav_path_o, filename.replace(".txt", ".wav"))
        label_path = os.path.join(labels_path, filename)
        
        try:
            labels, vad_labels = evaluate_vad(
                wav_path, 
                label_path, 
                vad_model, 
                get_speech_timestamps
            )
            
            # Update metrics
            fp_time += np.count_nonzero((labels == 0) & (vad_labels == 1))
            fn_time += np.count_nonzero((labels == 1) & (vad_labels == 0))
            y_speech_time += np.count_nonzero(labels == 1)
            y_nonspeech_time += np.count_nonzero(labels == 0)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Calculate final metrics
    pfp = fp_time / y_nonspeech_time if y_nonspeech_time > 0 else 0  # false alarm
    pfn = fn_time / y_speech_time if y_speech_time > 0 else 0  # miss
    dcf = 0.75 * pfn + 0.25 * pfp
    
    print("\nFinal Metrics:")
    print(f"False Positives (FP): {fp_time}")
    print(f"False Negatives (FN): {fn_time}")
    print(f"Total Speech Frames: {y_speech_time}")
    print(f"Total Non-Speech Frames: {y_nonspeech_time}")
    print(f"PFP: {pfp:.4f}, PFN: {pfn:.4f}")
    print(f"DCF: {dcf*100:.4f}%")

if __name__ == "__main__":
    main()