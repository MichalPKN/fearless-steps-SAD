import io
import os
from google.cloud import speech_v1p1beta1 as speech
import pyaudio
import wave

class VoiceActivityDetector:
    def __init__(self, language_code='en-US', sample_rate=16000):
        """
        Initialize Voice Activity Detector
        
        Args:
            language_code (str): Language for speech recognition
            sample_rate (int): Audio sampling rate
        """
        # Initialize Speech Client
        self.client = speech.SpeechClient()
        
        # Streaming recognition configuration
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=language_code,
                # VAD-specific configurations
                enable_speaker_diarization=True,
                diarization_speaker_count=2,
                use_enhanced=True,
                model='video',  # Enhanced VAD model
                speech_contexts=[
                    speech.SpeechContext(
                        phrases=["hello", "yes", "no"],  # Optional: improve recognition
                        boost=1.0
                    )
                ]
            ),
            interim_results=True  # Get interim results
        )
        
        # PyAudio setup for microphone streaming
        self.pyaudio_instance = pyaudio.PyAudio()
        self.sample_rate = sample_rate
        self.chunk_size = 1600  # 100ms of audio at 16000 Hz
    
    def stream_audio_to_speech(self, duration=10):
        """
        Stream audio from microphone and detect voice activity
        
        Args:
            duration (int): Recording duration in seconds
        """
        # Open microphone stream
        stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Prepare streaming recognize requests generator
        def audio_generator():
            for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                yield speech.StreamingRecognizeRequest(audio_content=data)
        
        # Perform streaming recognition
        responses = self.client.streaming_recognize(
            streaming_config=self.streaming_config, 
            requests=audio_generator()
        )
        
        # Process responses
        for response in responses:
            if not response.results:
                continue
            
            result = response.results[0]
            if result.is_final:
                print("Final transcript:", result.alternatives[0].transcript)
                
                # Speaker diarization details
                if result.speaker_labels:
                    for speaker in result.speaker_labels:
                        print(f"Speaker {speaker.label}: {speaker.confidence}")
            
            # Intermediate results
            elif result.alternatives:
                print("Interim result:", result.alternatives[0].transcript)
        
        # Close the stream
        stream.stop_stream()
        stream.close()
    
    def process_audio_file(self, file_path):
        """
        Process an existing audio file for VAD
        
        Args:
            file_path (str): Path to the audio file
        """
        with io.open(file_path, 'rb') as audio_file:
            content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=content)
        
        # Use streaming config for file processing
        response = self.client.recognize(
            config=self.streaming_config.config, 
            audio=audio
        )
        
        for result in response.results:
            print("Transcript:", result.alternatives[0].transcript)
            
            # Check for speaker labels
            if result.speaker_labels:
                for speaker in result.speaker_labels:
                    print(f"Speaker {speaker.label}: {speaker.confidence}")

def main():
    # Set up Google Cloud credentials
    # Make sure to set the GOOGLE_APPLICATION_CREDENTIALS environment variable
    # to the path of your service account key JSON file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'just-surge-452717-h6-8c46a3c0536e.json'
    
    # Path to your audio file (WAV format recommended)
    audio_file_path = 'FSC_P4_Streams/Audio/Streams/Dev/fsc_p4_dev_001.wav'
    
    # Initialize VAD
    vad = VoiceActivityDetector()

    vad.process_audio_file('FSC_P4_Streams/Audio/Streams/Dev/fsc_p4_dev_001.wav')

if __name__ == '__main__':
    main()