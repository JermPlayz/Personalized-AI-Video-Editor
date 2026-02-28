"""
Advanced Analysis Modules
Optimized + Highlight Scoring Layer Added
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

from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# AUDIO ANALYZER
# =========================

class AudioAnalyzer:
    def extract_audio(self, video_path: str) -> str:
        logger.info("Extracting audio...")
        audio_path = "temp_audio.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            audio_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(audio_path, sr=16000)
        return audio, sr

    def compute_energy(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        hop_length = 512
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=sr,
            hop_length=hop_length
        )
        return times, energy


# =========================
# TRANSCRIPTION ANALYZER
# =========================

class TranscriptionAnalyzer:

    def __init__(self, model_size="base", pause_threshold=0.5, repeat_threshold=90):
        logger.info(f"Loading Faster-Whisper model: {model_size}")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.pause_threshold = pause_threshold
        self.repeat_threshold = repeat_threshold

    def transcribe(self, audio_path: str):
        segments, _ = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True
        )

        words = []
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

        return words

    def build_segments(self, words: List[Dict]) -> List[Dict]:
        chunks = []
        current = []

        for i, word in enumerate(words):
            if i == 0:
                current.append(word)
                continue

            pause = word["start"] - words[i-1]["end"]

            if pause > self.pause_threshold:
                chunks.append(current)
                current = [word]
            else:
                current.append(word)

        if current:
            chunks.append(current)

        segments = []
        for chunk in chunks:
            text = " ".join(w["word"] for w in chunk)
            segments.append({
                "start": float(chunk[0]["start"]),
                "end": float(chunk[-1]["end"]),
                "text": text,
                "duration": float(chunk[-1]["end"] - chunk[0]["start"])
            })

        return segments

    def remove_repeats(self, segments: List[Dict]) -> List[Dict]:
        clean = []
        seen = []

        for seg in segments:
            norm = self._normalize(seg["text"])

            is_repeat = any(
                fuzz.token_set_ratio(norm, self._normalize(s)) > self.repeat_threshold
                for s in seen
            )

            if not is_repeat:
                clean.append(seg)
                seen.append(seg["text"])

        return clean

    def _normalize(self, text: str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# =========================
# VISUAL FEATURES
# =========================

class VideoFeatureAnalyzer:

    def __init__(self, sample_rate=10):
        self.sample_rate = sample_rate

    def analyze(self, video_path: str):
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
            small = cv2.resize(gray, (320, 240))
            timestamp = float(frame_idx / fps)

            # Motion
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, small,
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = float(np.mean(mag))

                motion_data.append({
                    "timestamp": timestamp,
                    "motion_score": motion_score,
                    "is_high_motion": bool(motion_score > 5.0)
                })

            prev_frame = small

            # Faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face_data.append({
                "timestamp": timestamp,
                "face_count": int(len(faces)),
                "has_face": bool(len(faces) > 0)
            })

            frame_idx += 1

        cap.release()
        return motion_data, face_data


# =========================
# MAIN ORCHESTRATOR
# =========================

class VideoAnalyzer:

    def __init__(self):
        self.audio = AudioAnalyzer()
        self.transcriber = TranscriptionAnalyzer()
        self.visual = VideoFeatureAnalyzer()

    def analyze_full(self, video_path: str, output_json="analysis.json"):

        audio_path = self.audio.extract_audio(video_path)

        audio, sr = self.audio.load_audio(audio_path)
        times, energy = self.audio.compute_energy(audio, sr)

        words = self.transcriber.transcribe(audio_path)
        segments = self.transcriber.build_segments(words)
        segments = self.transcriber.remove_repeats(segments)

        motion, faces = self.visual.analyze(video_path)

        # 🔥 HIGHLIGHT SCORING LAYER
        scored_segments = self.score_segments(segments, motion, faces)

        analysis = {
            "video_path": video_path,
            "segments": scored_segments,
            "motion": motion,
            "faces": faces
        }

        with open(output_json, "w") as f:
            json.dump(analysis, f, indent=2)

        os.remove(audio_path)

        return analysis

    def score_segments(self, segments, motion, faces):
        for seg in segments:
            seg_motion = [
                m["motion_score"]
                for m in motion
                if seg["start"] <= m["timestamp"] <= seg["end"]
            ]

            seg_faces = [
                f["has_face"]
                for f in faces
                if seg["start"] <= f["timestamp"] <= seg["end"]
            ]

            motion_score = np.mean(seg_motion) if seg_motion else 0
            face_score = sum(seg_faces) / len(seg_faces) if seg_faces else 0

            seg["highlight_score"] = float(
                0.6 * motion_score +
                0.4 * face_score * 5
            )

        return sorted(segments, key=lambda x: x["highlight_score"], reverse=True)