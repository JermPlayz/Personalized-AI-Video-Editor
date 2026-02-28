"""
Step 4.3 — Edit Decision Language (EDL)
Builds edit plans from analysis
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
import numpy as np

@dataclass
class VideoTransform:
    crop: str = "9:16_center"
    zoom: float = 1.0
    zoom_duration: Optional[float] = None
    rotation: float = 0

@dataclass
class AudioTransform:
    voice_gain_db: float = 0
    denoise: bool = False
    normalize: bool = True

@dataclass
class EditSegment:
    src: str
    in_time: float
    out_time: float
    video: VideoTransform
    audio: AudioTransform
    reason: Optional[str] = None

@dataclass
class MusicTrack:
    src: str
    start: float
    duck_db: float = -12

@dataclass
class EditPlan:
    meta: Dict
    timeline: List[EditSegment]
    captions: Dict
    music: List[MusicTrack]
    
    def to_json(self, path: str):
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
    """Build edit plans from analysis"""
    
    def __init__(self, analysis: Dict, video_path: str, padding: float = 0.15):
        self.analysis = analysis
        self.video_path = video_path
        self.padding = padding
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
    
    def apply_keep_segments_rule(self, min_segment_duration: float = 0.5):
        """
        Apply the keep_segments directly from transcription analysis.
        This uses the PROVEN pause detection logic.
        """
        keep_segments = self.analysis.get("keep_segments", [])
        
        if not keep_segments:
            return
        
        for seg in keep_segments:
            # Skip very short segments
            if seg["duration"] < min_segment_duration:
                continue
            
            # Add padding
            start = max(0, seg["start"] - self.padding)
            end = seg["end"] + self.padding
            
            edit_segment = EditSegment(
                src=self.video_path,
                in_time=start,
                out_time=end,
                video=VideoTransform(),
                audio=AudioTransform(),
                reason=f"keep_segment: {seg['text'][:30]}"
            )
            
            self.plan.timeline.append(edit_segment)
    
    def apply_motion_emphasis_rule(self, motion_threshold: float = 5.0):
        """Add zoom on high-motion segments"""
        motion_data = self.analysis.get("motion", [])
        
        for seg in self.plan.timeline:
            segment_motion = [m for m in motion_data 
                            if seg.in_time <= m["timestamp"] <= seg.out_time]
            
            if segment_motion:
                avg_motion = np.mean([m["motion_score"] for m in segment_motion])
                
                if avg_motion > motion_threshold:
                    seg.video.zoom = 1.08
                    seg.video.zoom_duration = seg.out_time - seg.in_time
    
    def get_plan(self) -> EditPlan:
        return self.plan