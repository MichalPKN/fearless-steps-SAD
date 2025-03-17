from speechbrain.pretrained import VAD
import torchaudio
from speechbrain.dataio.dataio import read_audio
import numpy as np
import load
import torch
import os

# Load the pre-trained VAD model
vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="tmpdir")

labels_path = "FSC_P4_Streams\\Transcripts\\SAD\\Dev"
wav_path_o = "FSC_P4_Streams\\Audio\\Streams\\Dev"

labels = []
X = []
fp_time = 0
fn_time = 0
y_speech_time = 0
y_nonspeech_time = 0
y_time = 0
correct_count = 0
loader = load.LoadAudio()

def classify(audio_path):

    # 8 kHz
    waveform = read_audio("FSC_P4_Streams/Audio/Streams/Dev/fsc_p4_dev_001.wav")
    waveform = waveform.unsqueeze(0)  # Add batch dimension

    # 16 kHz
    resampler = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    resampled_waveform = resampler(waveform)

    resampled_waveform = resampled_waveform.squeeze(0)
    speech_probs = vad.get_speech_prob_chunk(resampled_waveform)

    threshold = 0.5
    speech_segments = (speech_probs > threshold).float()

    speech_segments = speech_segments.squeeze()
    speech_probs = speech_probs.squeeze()

    # sample_rate = 16000
    # Print out speech segments (start and end times)
    # for segment in speech_segments:
    #     print(f"Speech segment from {segment[0] / sample_rate}s to {segment[1] / sample_rate}s")

    print(f"speech_probs: {speech_probs.shape}") #torch.Size([180001, 1])
    print(f"speecch_segments: {speech_segments.shape}") #torch.Size([180001, 1]
    
    return speech_probs, speech_segments

brakee = False
print(f"Loading labels from {labels_path}")
for i, filename in enumerate(sorted(os.listdir(labels_path))):
    
    wav_path = os.path.join(wav_path_o, filename.replace(".txt", ".wav"))


    speech_probs, speech_segments = classify(wav_path)
    vad_results = speech_segments.numpy()
    loader = load.LoadAudio()

    label_path = os.path.join(labels_path, filename)
    labels, num_of_1s, num_of_0s = loader.add_labels(label_path, vad_results)
    labels = torch.from_numpy(labels.squeeze())
    print("added labels")
    print(f"shape: {labels.shape}")
    labels = labels[:vad_results.shape[0]]
    vad_results = vad_results[:labels.shape[0]]
    print(f"shape: {labels.shape}")
    fp_time += np.count_nonzero((labels == 0) & (vad_results == 1))
    fn_time += np.count_nonzero((labels == 1) & (vad_results == 0))
    y_speech_time += np.count_nonzero(labels == 1)
    y_nonspeech_time += np.count_nonzero(labels == 0)
    y_time += len(labels)
    correct_count += np.count_nonzero(labels == vad_results)
    if brakee:
        break
    #brakee = True

pfp = fp_time / y_nonspeech_time # false alarm
pfn = fn_time / y_speech_time # miss
dcf = 0.75 * pfn + 0.25 * pfp
accuracy = correct_count / y_time

print(f"dcf: {dcf}")
print(f"False Positives (FP): {fp_time}")
print(f"False Negatives (FN): {fn_time}")
print(f"accuracy: {accuracy}")

    
import matplotlib.pyplot as plt
import numpy as np

# Plot the speech probabilities
plt.figure(figsize=(12, 6))

# Plot the speech probabilities
plt.subplot(3, 1, 1)
plt.plot(speech_probs.numpy(), label='Speech Probabilities', color='blue')
plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
plt.title('Speech Probabilities Over Time')
plt.xlabel('Time (frames)')
plt.ylabel('Probability')
plt.legend()

# Plot the detected speech segments
plt.subplot(3, 1, 2)
plt.plot(speech_segments.numpy(), label='Speech Segments', color='green', drawstyle='steps-post')
plt.title('Detected Speech Segments')
plt.xlabel('Time (frames)')
plt.ylabel('Speech (1) / No Speech (0)')
plt.legend()

# Plot the acutal
plt.subplot(3, 1, 3)
plt.plot(labels.numpy(), label='actual labels', color='red', drawstyle='steps-post')
plt.title('Actual labels')
plt.xlabel('Time (frames)')
plt.ylabel('Speech (1) / No Speech (0)')
plt.legend()

plt.tight_layout()
plt.show()