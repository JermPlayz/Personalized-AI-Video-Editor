"""
Step 4.3 — Edit Decision Language (EDL)
Defines what edits to make in a standardized JSON format
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json

@dataclass
class VideoTransform:
    crop: str = "9:16_center"  # "9:16_center", "9:16_center_face", "16:9", etc
    zoom: float = 1.0
    zoom_duration: Optional[float] = None  # If animating zoom
    rotation: float = 0

@dataclass
class AudioTransform:
    voice_gain_db: float = 0
    denoise: bool = False
    normalize: bool = True

@dataclass
class EditSegment:
    src: str  # Source video file
    in_time: float  # Start time in source (seconds)
    out_time: float  # End time in source (seconds)
    video: VideoTransform
    audio: AudioTransform
    reason: Optional[str] = None  # Why we kept this segment

@dataclass
class CaptionStyle:
    font: str = "Arial"
    size: int = 64
    stroke: int = 3
    color: str = "#FFFFFF"
    position: str = "bottom_center"  # "top_center", "bottom_center", "bottom_left"
    background: bool = False

@dataclass
class MusicTrack:
    src: str
    start: float  # When in timeline to start
    duck_db: float = -12  # How much to lower voice when music plays

@dataclass
class EditPlan:
    meta: Dict  # fps, resolution, etc
    timeline: List[EditSegment]
    captions: Dict  # burn_in, style
    music: List[MusicTrack]
    
    def to_json(self, path: str):
        """Save edit plan as JSON"""
        data = {
            "meta": self.meta,
            "timeline": [asdict(seg) for seg in self.timeline],
            "captions": self.captions,
            "music": [asdict(m) for m in self.music]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def from_json(path: str) -> "EditPlan":
        """Load edit plan from JSON"""
        with open(path, "r") as f:
            data = json.load(f)
        
        segments = [EditSegment(**s) for s in data["timeline"]]
        music = [MusicTrack(**m) for m in data.get("music", [])]
        
        return EditPlan(
            meta=data["meta"],
            timeline=segments,
            captions=data.get("captions", {}),
            music=music
        )

class EditPlanBuilder:
    """Build edit plans from analysis and rules"""
    
    def __init__(self, analysis: Dict, video_path: str):
        self.analysis = analysis
        self.video_path = video_path
        self.plan = EditPlan(
            meta={"fps": 30, "resolution": "1080x1920"},
            timeline=[],
            captions={"burn_in": True, "style": {
                "font": "Montserrat",
                "size": 64,
                "stroke": 6,
                "position": "bottom_center"
            }},
            music=[]
        )
    
    def apply_silence_removal_rule(self, silence_threshold_s: float = 0.5):
        """Remove segments that are pure silence"""
        silences = self.analysis["audio"]["silences"]
        
        # Mark all times that are silence
        silence_ranges = [(s["start"], s["end"]) for s in silences 
                         if s["duration"] > silence_threshold_s]
        
        # Build segments avoiding silences
        words = self.analysis["transcription"]["words"]
        if not words:
            return
        
        current_segment_start = words[0]["start"]
        
        for i, word in enumerate(words):
            in_silence = any(s_start <= word["start"] < s_end 
                           for s_start, s_end in silence_ranges)
            
            if in_silence and current_segment_start is not None:
                # End segment
                segment = EditSegment(
                    src=self.video_path,
                    in_time=current_segment_start,
                    out_time=word["start"],
                    video=VideoTransform(),
                    audio=AudioTransform(),
                    reason="silence_removed"
                )
                self.plan.timeline.append(segment)
                current_segment_start = None
            elif not in_silence and current_segment_start is None:
                current_segment_start = word["start"]
        
        # Add final segment
        if current_segment_start is not None:
            segment = EditSegment(
                src=self.video_path,
                in_time=current_segment_start,
                out_time=words[-1]["end"],
                video=VideoTransform(),
                audio=AudioTransform(),
                reason="silence_removed"
            )
            self.plan.timeline.append(segment)
    
    def apply_scene_cut_rule(self):
        """Use scene cuts as natural breakpoints"""
        scenes = self.analysis["scenes"]
        # Implementation: use scene cuts as segment boundaries
        pass
    
    def apply_motion_emphasis_rule(self, motion_threshold: float = 5.0):
        """Add zoom/punch-in on high-motion segments"""
        motion_data = self.analysis["motion"]
        
        for seg in self.plan.timeline:
            segment_motion = [m for m in motion_data 
                            if seg.in_time <= m["timestamp"] <= seg.out_time]
            
            avg_motion = np.mean([m["motion_score"] for m in segment_motion]) if segment_motion else 0
            
            if avg_motion > motion_threshold:
                seg.video.zoom = 1.08  # Subtle punch-in
                seg.video.zoom_duration = seg.out_time - seg.in_time
    
    def get_plan(self) -> EditPlan:
        """Return the current edit plan"""
        return self.plan