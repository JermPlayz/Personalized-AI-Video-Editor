"""
Step 4.2 — Analysis modules for video understanding
Uses proven pause detection + repeat removal from working script
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import subprocess
import os
import re

import librosa
from faster_whisper import WhisperModel
from scenedetect import detect, ContentDetector
import cv2

try:
    from rapidfuzz import fuzz
except ImportError:
    print("Installing rapidfuzz...")
    subprocess.run(["pip", "install", "rapidfuzz"], check=True)
    from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# AUDIO ANALYZER
# =========================

class AudioAnalyzer:
    """Extract and analyze audio"""
    
    def __init__(self):
        pass

    def extract_audio(self, video_path: str) -> str:
        logger.info("Extracting audio...")
        audio_path = "temp_audio.mp3"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "mp3",
            audio_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio extracted: {audio_path}")
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio: {e}")
            raise

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(audio_path, sr=16000)
        return audio, sr

    def compute_energy(self, audio: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=sr,
            hop_length=hop_length
        )
        return times, energy


# =========================
# TRANSCRIPTION + PAUSE DETECTION (PROVEN LOGIC)
# =========================

class TranscriptionAnalyzer:
    """
    Transcribe with word timestamps and detect pauses.
    Uses the proven pause-detection logic from your original script.
    """
    
    def __init__(self, model_size: str = "base", pause_threshold: float = 0.5, repeat_threshold: float = 90):
        logger.info(f"Loading Faster-Whisper model: {model_size}")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.pause_threshold = pause_threshold
        self.repeat_threshold = repeat_threshold

    def transcribe(self, audio_path: str) -> Tuple[str, List[Dict]]:
        """Transcribe audio to words with timestamps"""
        logger.info("Transcribing with Faster-Whisper...")

        segments, info = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True
        )

        words = []
        full_text_parts = []

        for segment in segments:
            if segment.words:
                for word in segment.words:
                    confidence = getattr(
                        word,
                        "probability",
                        getattr(word, "confidence", 0.0)
                    )

                    words.append({
                        "word": word.word.strip(),
                        "start": float(word.start),
                        "end": float(word.end),
                        "confidence": float(confidence)
                    })

                    full_text_parts.append(word.word.strip())

        return " ".join(full_text_parts), words

    def build_chunks_from_pauses(self, words: List[Dict]) -> List[List[Dict]]:
        """
        Build speech chunks based on pauses between words.
        This is the PROVEN logic from your original script.
        """
        logger.info(f"Building chunks with pause threshold: {self.pause_threshold}s")

        if not words:
            logger.warning("No words found in transcription")
            return []

        chunks = []
        current_chunk = []

        for i, word in enumerate(words):
            if i == 0:
                current_chunk.append(word)
                continue

            prev_word = words[i - 1]
            pause = word["start"] - prev_word["end"]

            if pause > self.pause_threshold:
                # End current chunk and start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [word]
            else:
                current_chunk.append(word)

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Built {len(chunks)} chunks based on pauses")
        return chunks

    def chunks_to_segments(self, chunks: List[List[Dict]]) -> List[Dict]:
        """Convert chunks to keep segments with timing"""
        keep_segments = []

        for chunk in chunks:
            # Remove leading silence
            while chunk and chunk[0]["word"].strip() in ["", " "]:
                chunk.pop(0)

            # Remove trailing silence
            while chunk and chunk[-1]["word"].strip() in ["", " "]:
                chunk.pop()

            if not chunk:
                continue  # Skip empty chunks

            start_time = chunk[0]["start"]
            end_time = chunk[-1]["end"]
            text = " ".join(w["word"] for w in chunk)

            keep_segments.append({
                "start": float(start_time),
                "end": float(end_time),
                "text": text,
                "duration": float(end_time - start_time)
            })

        logger.info(f"Created {len(keep_segments)} keep segments")
        return keep_segments

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = re.sub(r"\s+", " ", text)  # collapse whitespace
        return text.strip()

    def remove_repeated_phrases(self, segments: List[Dict]) -> List[Dict]:
        """Remove segments with repeated phrases (PROVEN LOGIC)"""
        logger.info(f"Removing repeated phrases (threshold: {self.repeat_threshold}%)")

        clean_segments = []
        seen_texts = []
        removed_count = 0

        for seg in segments:
            normalized = self.normalize_text(seg["text"])

            # Check if similar to any previous segment
            is_repeat = any(
                fuzz.token_set_ratio(normalized, self.normalize_text(seen)) > self.repeat_threshold
                for seen in seen_texts
            )

            if is_repeat:
                logger.debug(f"Removed repeat: {seg['text'][:50]}...")
                removed_count += 1
            else:
                clean_segments.append(seg)
                seen_texts.append(seg["text"])

        logger.info(f"Removed {removed_count} repeated segments")
        return clean_segments

    def merge_close_segments(self, segments: List[Dict], min_gap: float = 0.5) -> List[Dict]:
        """Merge segments that are too close together"""
        logger.info(f"Merging segments closer than {min_gap}s")

        if not segments:
            return []

        merged_segments = [segments[0].copy()]
        merged_count = 0

        for seg in segments[1:]:
            last_seg = merged_segments[-1]
            gap = seg["start"] - last_seg["end"]

            if gap < min_gap:
                # Merge segments
                last_seg["end"] = max(last_seg["end"], seg["end"])
                last_seg["text"] += " " + seg["text"]
                last_seg["duration"] = last_seg["end"] - last_seg["start"]
                merged_count += 1
            else:
                merged_segments.append(seg.copy())

        logger.info(f"Merged {merged_count} segment pairs")
        return merged_segments

    def analyze_transcript_full(self, audio_path: str) -> List[Dict]:
        """Full transcription pipeline: transcribe -> chunk -> remove repeats -> merge"""
        logger.info("Starting full transcription analysis...")

        # Transcribe
        full_text, words = self.transcribe(audio_path)

        # Build chunks based on pauses
        chunks = self.build_chunks_from_pauses(words)

        # Convert to segments with timing
        keep_segments = self.chunks_to_segments(chunks)

        # Remove repeated phrases
        clean_segments = self.remove_repeated_phrases(keep_segments)

        # Merge close segments
        final_segments = self.merge_close_segments(clean_segments)

        logger.info(f"Transcription analysis complete: {len(final_segments)} final segments")
        return final_segments


# =========================
# SCENE DETECTION
# =========================

class SceneDetector:
    """Detect scene changes in video"""
    
    def detect_scenes(self, video_path: str) -> List[Dict]:
        logger.info("Detecting scene changes...")
        try:
            scenes = detect(video_path, ContentDetector(threshold=27.0))

            scene_list = []
            for i, scene in enumerate(scenes):
                timestamp = scene[0].get_seconds()
                scene_list.append({
                    "index": int(i),
                    "timestamp": float(timestamp),
                    "confidence": 0.9
                })

            logger.info(f"Found {len(scene_list)} scene changes")
            return scene_list
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            return []


# =========================
# VISUAL FEATURES
# =========================

class VideoFeatureAnalyzer:
    """Extract visual features: motion, faces"""
    
    def __init__(self, sample_rate: int = 10):
        self.sample_rate = sample_rate

    def analyze_visual_features(self, video_path: str) -> Tuple[List[Dict], List[Dict]]:
        logger.info("Analyzing visual features...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        prev_frame = None
        frame_idx = 0

        motion_data = []
        face_data = []

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (320, 240))

            timestamp = float(frame_idx / fps)

            # Motion
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray_small,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = float(np.mean(magnitude))

                motion_data.append({
                    "timestamp": timestamp,
                    "motion_score": motion_score,
                    "is_high_motion": bool(motion_score > 5.0)
                })

            prev_frame = gray_small

            # Faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face_data.append({
                "timestamp": timestamp,
                "face_count": int(len(faces)),
                "has_face": bool(len(faces) > 0)
            })

            frame_idx += 1

        cap.release()
        logger.info(f"Analyzed {frame_idx} frames")
        return motion_data, face_data


# =========================
# MAIN ORCHESTRATOR
# =========================

class VideoAnalyzer:
    """Orchestrates all analysis modules"""
    
    def __init__(self, whisper_model: str = "base", pause_threshold: float = 0.5, repeat_threshold: float = 90):
        self.audio_analyzer = AudioAnalyzer()
        self.transcription_analyzer = TranscriptionAnalyzer(whisper_model, pause_threshold, repeat_threshold)
        self.scene_detector = SceneDetector()
        self.video_feature_analyzer = VideoFeatureAnalyzer()

    def analyze_full(self, video_path: str, output_json: str = "analysis.json") -> Dict:
        """Run complete analysis on video"""
        logger.info(f"Starting full analysis: {video_path}")

        audio_path = None

        try:
            # Extract audio
            audio_path = self.audio_analyzer.extract_audio(video_path)

            # Transcription + pause detection
            keep_segments = self.transcription_analyzer.analyze_transcript_full(audio_path)

            # Scene detection
            scenes = self.scene_detector.detect_scenes(video_path)

            # Visual features
            motion_data, face_data = self.video_feature_analyzer.analyze_visual_features(video_path)

            analysis = {
                "video_path": str(video_path),
                "keep_segments": keep_segments,  # Key output: segments to keep
                "scenes": scenes,
                "motion": motion_data,
                "faces": face_data
            }

            with open(output_json, "w") as f:
                json.dump(analysis, f, indent=2)

            logger.info(f"Analysis saved: {output_json}")
            return analysis

        finally:
            # Cleanup temp audio
            if audio_path and Path(audio_path).exists():
                try:
                    os.remove(audio_path)
                except:
                    pass