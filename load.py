import os
import librosa
import numpy as np

class LoadAudio:
    def __init__(self, input_size=40, frame_length=0.01, debug=False):
        self.input_size = input_size
        self.frame_length = frame_length
        self.debug = debug

    def load_all(self, audio_dir, labels_path=None):
        if not os.path.exists(audio_dir):
            raise ValueError(f"Audio directory {audio_dir} does not exist.")
        
        labels = []
        audio_info = [0, 0, 0, []]
        X = []
        print(f"Loading audio from {audio_dir}")
        for i, filename in enumerate(os.listdir(audio_dir)):
            if filename.endswith(".wav"):
                file_path = os.path.join(audio_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                audio_info[3].append(audio_data)
                features = self.extract_features(file_path)
                #audio_info = self.extract_audio_info(file_path)
                if features is not None:
                    X.append(features)
                    #audio_info_list.append(audio_info)
            if self.debug and i >= 1:
                break
        print(f"Loaded {len(X)} audio files")
        if labels_path is not None:
            print(f"Loading labels from {labels_path}")
            for i, filename in enumerate(os.listdir(labels_path)):
                label_path = os.path.join(labels_path, filename)
                labels_f, num_of_1s, num_of_0s = self.add_labels(label_path, X[i])
                audio_info[0] += num_of_1s
                audio_info[1] += num_of_0s
                audio_info[2] += len(labels_f)
                if num_of_0s + num_of_1s != len(labels_f):
                    print(f"Error: num_of_0s + num_of_1s != len(labels_f) for {label_path}")
                    print(f"num_of_0s: {num_of_0s}, num_of_1s: {num_of_1s}, len(labels_f): {len(labels_f)}")
                labels.append(labels_f)
                if self.debug and i >= 1:
                    break
            print(f"Loaded {len(labels)} labels")
            
        if labels:
            # for i in range(len(labels)):
            #     print(f"audio length: {len(X[i])}, labels length: {len(labels[i])}")
            return X, audio_info, labels
        return X, None, None

    def extract_features(self, file_path):
        audio_data, sr = librosa.load(file_path, sr=None)
        frame_size = int(self.frame_length * sr)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=self.input_size, hop_length=frame_size)
        return mfcc.T
    
    
    def extract_audio_info(self, file_path):
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=None)  # sr=None to preserve original sample rate
            
            # Get duration in seconds
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Other features you can extract
            rms = librosa.feature.rms(y=y)  # Root Mean Square Energy
            zero_crossings = librosa.zero_crossings(y).sum()  # Zero crossing rate
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Tempo estimation
            
            # Return gathered information
            return {
                "file_path": file_path,
                "sample_rate": sr,
                "duration": duration,
                "rms": rms.mean(),  # Mean RMS value
                "zero_crossings": zero_crossings,
                "tempo": tempo
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
        
    def add_labels(self, labels_path, features):
        if not os.path.exists(labels_path):
            raise ValueError(f"Labels file {labels_path} does not exist.")
        
        with open(labels_path, "r") as f:
            label_file = f.readlines()
        
        labels = np.zeros((features.shape[0], 1))
        num_of_1s = 0
        num_of_0s = 0
        for line in label_file:
            line_data = line.split()
            start_time = line_data[2]
            end_time = line_data[3]
            label = 1 if line_data[4] == "S" else 0
            start = int(float(start_time) / self.frame_length)
            end = int(float(end_time) / self.frame_length)
            labels[start:end, 0] = int(label)
            if label == 1:
                num_of_1s += end - start
            else:
                num_of_0s += end - start
        labels[end:, 0] = label
        if label == 1:
            num_of_1s += len(labels) - end
        else:
            num_of_0s += len(labels) - end
        # print(end, start, end-start, labels.shape[0])
        # print(np.count_nonzero(labels), num_of_1s)
        # print(np.count_nonzero(labels == 0), num_of_0s)
        return labels, num_of_1s, num_of_0s

if __name__ == "__main__":
    audio_dir = "FSC_P4_Streams\Audio\Streams\Dev"

    loader = LoadAudio(audio_dir, debug=True)
    X, audio_info_list = loader.load_all()

    total_time = sum(info["duration"] for info in audio_info_list)
    print(f"{len(audio_info_list)} audio files - {total_time / 3600:.2f} hours")
    print(X.shape)
    print(X.dtype)