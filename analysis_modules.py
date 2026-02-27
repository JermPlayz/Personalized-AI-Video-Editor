"""
Step 4.2 — Analysis modules for video understanding
Extracts audio, transcribes, detects silences, scene cuts, motion
Outputs JSON for inspection
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass, asdict
import subprocess

import librosa
import scipy.signal as signal
from faster_whisper import WhisperModel
from scenedetect import detect, ContentDetector
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    start: float
    end: float
    duration: float
    rms_energy: float
    is_silence: bool

@dataclass
class TranscriptionWord:
    word: str
    start: float
    end: float
    confidence: float

@dataclass
class SceneChange:
    frame_num: int
    timestamp: float
    confidence: float

class AudioAnalyzer:
    """Extracts audio features: silence, energy, voice detection"""
    
    def __init__(self, silence_threshold: float = 0.02):
        self.silence_threshold = silence_threshold
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        logger.info("Extracting audio...")
        audio_path = "temp_audio.wav"
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Audio extracted: {audio_path}")
        return audio_path
    
    def detect_silences(self, audio_path: str,
                    silence_percentile: float = 20.0,
                    min_silence_duration: float = 0.3) -> List[Dict]:

        logger.info("Detecting silences...")

        audio, sr = librosa.load(audio_path, sr=16000)

        frame_length = 2048
        hop_length = 512

        # RMS directly from waveform (FIXED)
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Smooth energy (prevents jittery cuts)
        energy_smooth = librosa.util.normalize(
            np.convolve(energy, np.ones(5)/5, mode='same')
        )

        # Dynamic threshold
        threshold = np.percentile(energy_smooth, silence_percentile)

        silence_frames = energy_smooth < threshold

        times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=sr,
            hop_length=hop_length
        )

        silences = []
        silence_start = None

        for time, is_silent in zip(times, silence_frames):
            if is_silent and silence_start is None:
                silence_start = time
            elif not is_silent and silence_start is not None:
                duration = time - silence_start
                if duration >= min_silence_duration:
                    silences.append({
                        "start": float(silence_start),
                        "end": float(time),
                        "duration": float(duration)
                    })
                silence_start = None

        if silence_start is not None:
            duration = times[-1] - silence_start
            if duration >= min_silence_duration:
                silences.append({
                    "start": float(silence_start),
                    "end": float(times[-1]),
                    "duration": float(duration)
                })

        logger.info(f"Found {len(silences)} silence segments")
        return silences
    
    def compute_energy(self, audio_path: str, 
                      hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RMS energy over time"""
        logger.info("Computing audio energy...")
        
        audio, sr = librosa.load(audio_path, sr=None)
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
        
        return times, energy

class TranscriptionAnalyzer:
    """Transcribe audio and get word-level timestamps"""
    
    def __init__(self, model_size: str = "base"):
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    def transcribe(self, audio_path: str) -> Tuple[str, List[Dict]]:
        """Transcribe with word-level timestamps"""
        logger.info("Transcribing with Faster-Whisper...")
        
        segments, info = self.model.transcribe(audio_path, word_level=True)
        
        words = []
        full_text = ""
        
        for segment in segments:
            for word in segment.words:
                words.append({
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "confidence": float(word.confidence)
                })
                full_text += word.word + " "
        
        logger.info(f"Transcribed {len(words)} words")
        return full_text.strip(), words

class SceneDetector:
    """Detect scene changes in video"""
    
    def __init__(self):
        pass
    
    def detect_scenes(self, video_path: str) -> List[Dict]:
        """Detect scene cuts using PySceneDetect"""
        logger.info("Detecting scene changes...")
        
        try:
            scenes = detect(video_path, ContentDetector(threshold=27.0))
            
            scene_list = []
            for i, scene in enumerate(scenes):
                timestamp = scene[0].get_seconds()
                scene_list.append({
                    "index": i,
                    "timestamp": float(timestamp),
                    "confidence": 0.9
                })
            
            logger.info(f"Found {len(scene_list)} scene changes")
            return scene_list
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            return []

class VideoFeatureAnalyzer:
    """Extract visual features: motion, face presence, blur"""
    
    def __init__(self, sample_rate: int = 10):
        self.sample_rate = sample_rate  # Analyze every Nth frame
    
    def analyze_motion(self, video_path: str) -> List[Dict]:
        """Detect motion (optical flow magnitude)"""
        logger.info("Analyzing motion...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_frame = None
        motion_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                continue
            
            # Convert to grayscale and resize for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))
            
            if prev_frame is not None:
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = np.mean(magnitude)
                
                timestamp = frame_idx / fps
                motion_data.append({
                    "timestamp": float(timestamp),
                    "motion_score": float(motion_score),
                    "is_high_motion": motion_score > 5.0
                })
            
            prev_frame = gray
            frame_idx += 1
        
        cap.release()
        logger.info(f"Analyzed motion in {len(motion_data)} frames")
        return motion_data
    
    def analyze_faces(self, video_path: str) -> List[Dict]:
        """Detect face presence using cascade classifier"""
        logger.info("Detecting faces...")
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        face_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            timestamp = frame_idx / fps
            face_data.append({
                "timestamp": float(timestamp),
                "face_count": int(len(faces)),
                "has_face": len(faces) > 0
            })
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Detected faces in {len(face_data)} frames")
        return face_data

class VideoAnalyzer:
    """Orchestrates all analysis modules"""
    
    def __init__(self, whisper_model: str = "base"):
        self.audio_analyzer = AudioAnalyzer()
        self.transcription_analyzer = TranscriptionAnalyzer(whisper_model)
        self.scene_detector = SceneDetector()
        self.video_feature_analyzer = VideoFeatureAnalyzer()
    
    def analyze_full(self, video_path: str, output_json: str = "analysis.json") -> Dict:
        """Run complete analysis on video"""
        logger.info(f"Starting full analysis: {video_path}")
        
        try:
            # Extract and analyze audio
            audio_path = self.audio_analyzer.extract_audio(video_path)
            silences = self.audio_analyzer.detect_silences(audio_path)
            times, energy = self.audio_analyzer.compute_energy(audio_path)
            
            # Transcribe
            full_text, words = self.transcription_analyzer.transcribe(audio_path)
            
            # Detect scenes
            scenes = self.scene_detector.detect_scenes(video_path)
            
            # Analyze video features
            motion_data = self.video_feature_analyzer.analyze_motion(video_path)
            face_data = self.video_feature_analyzer.analyze_faces(video_path)
            
            # Compile results
            analysis = {
                "video_path": video_path,
                "audio": {
                    "silences": silences,
                    "energy_data": {
                        "times": times.tolist(),
                        "energy": energy.tolist()
                    }
                },
                "transcription": {
                    "full_text": full_text,
                    "words": words
                },
                "scenes": scenes,
                "motion": motion_data,
                "faces": face_data
            }
            
            # Save to JSON
            with open(output_json, "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Analysis saved: {output_json}")
            return analysis
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise