"""
Step 4.4 — Video Renderer
Takes an EditPlan and produces final MP4 using FFmpeg
"""

import subprocess
import json
import os
import logging
from pathlib import Path
from typing import List, Dict
import tempfile
from edit_plan import EditPlan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FFmpegRenderer:
    """Render EditPlan to final video using FFmpeg"""
    
    def __init__(self, temp_dir: str = "temp_render"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def render(self, plan: EditPlan, output_path: str) -> str:
        """Render edit plan to video"""
        logger.info(f"Rendering to {output_path}")
        
        # Step 1: Extract and process clips
        clips = self._extract_clips(plan.timeline)
        
        # Step 2: Create concat demuxer file
        concat_file = self._create_concat_file(clips)
        
        # Step 3: Concatenate clips
        concatenated = self._concatenate(concat_file)
        
        # Step 4: Add captions
        if plan.captions.get("burn_in"):
            concatenated = self._add_captions(concatenated, plan)
        
        # Step 5: Add music/mix audio
        if plan.music:
            concatenated = self._mix_audio(concatenated, plan)
        
        # Step 6: Final output
        self._finalize(concatenated, output_path)
        
        logger.info(f"✓ Video rendered: {output_path}")
        return output_path
    
    def _extract_clips(self, timeline: List) -> List[str]:
        """Extract clips from source video based on timeline"""
        clips = []
        
        for i, segment in enumerate(timeline):
            output_file = self.temp_dir / f"clip_{i:03d}.mp4"
            
            duration = segment.out_time - segment.in_time
            
            # Build FFmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-i", segment.src,
                "-ss", str(segment.in_time),
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                str(output_file)
            ]
            
            if segment.video.crop:
                # Add crop/scale filter
                cmd = self._add_crop_filter(cmd, segment.video)
            
            logger.info(f"Extracting clip {i}...")
            subprocess.run(cmd, check=True, capture_output=True)
            clips.append(str(output_file))
        
        return clips
    
    def _add_crop_filter(self, cmd: List[str], video_transform) -> List[str]:
        """Add crop/scale filter to FFmpeg command"""
        # Example: crop to 9:16 (vertical)
        if "9:16" in video_transform.crop:
            # Assuming 1080x1920 output
            filter_str = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
            cmd.extend(["-vf", filter_str])
        
        return cmd
    
    def _create_concat_file(self, clips: List[str]) -> str:
        """Create FFmpeg concat demuxer file"""
        concat_file = self.temp_dir / "concat.txt"
        
        with open(concat_file, "w") as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")
        
        return str(concat_file)
    
    def _concatenate(self, concat_file: str) -> str:
        """Concatenate clips using FFmpeg"""
        output = self.temp_dir / "concatenated.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            str(output)
        ]
        
        logger.info("Concatenating clips...")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return str(output)
    
    def _add_captions(self, video_path: str, plan: EditPlan) -> str:
        """Burn captions into video"""
        # Implementation: Generate ASS subtitle file and burn with FFmpeg
        logger.info("Adding captions...")
        # TODO: Implement caption burning
        return video_path
    
    def _mix_audio(self, video_path: str, plan: EditPlan) -> str:
        """Mix voice and music with ducking"""
        logger.info("Mixing audio...")
        # TODO: Implement audio mixing with ducking
        return video_path
    
    def _finalize(self, video_path: str, output_path: str):
        """Final encode pass"""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            output_path
        ]
        
        logger.info("Final encoding...")
        subprocess.run(cmd, check=True, capture_output=True)
    
    def cleanup(self):
        """Remove temp files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)