"""
Test the base editor - simple version
"""

import json
import os
from pathlib import Path
from analysis_modules import VideoAnalyzer
from edit_plan import EditPlanBuilder
from renderer import FFmpegRenderer

def test_editor(video_path: str):
    """Test: Analyze → Build Plan → Render"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {video_path}")
    print(f"{'='*60}")
    
    # Step 1: Analyze
    print("\n[Step 1] Analyzing video...")
    analyzer = VideoAnalyzer(
        whisper_model="base",
        pause_threshold=0.5,      # Cut on pauses > 0.5s
        repeat_threshold=90       # Remove 90%+ similar repeats
    )
    analysis = analyzer.analyze_full(video_path, f"{Path(video_path).stem}_analysis.json")
    
    # Print what was found
    keep_segments = analysis.get("keep_segments", [])
    print(f"\n✓ Analysis complete:")
    print(f"  Keep segments found: {len(keep_segments)}")
    print(f"  Total keep duration: {sum(s['duration'] for s in keep_segments):.1f}s")
    
    if keep_segments:
        print(f"\n  First 3 segments:")
        for i, seg in enumerate(keep_segments[:3]):
            print(f"    [{i+1}] {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s): {seg['text'][:40]}...")
    
    # Step 2: Build Edit Plan
    print("\n[Step 2] Building edit plan...")
    builder = EditPlanBuilder(analysis, video_path, padding=0.1)  # Less padding
    builder.apply_keep_segments_rule(min_segment_duration=0.3)
    plan = builder.get_plan()
    
    print(f"✓ Edit plan created:")
    print(f"  Timeline segments: {len(plan.timeline)}")
    total_edited = sum(seg.out_time - seg.in_time for seg in plan.timeline)
    print(f"  Total edited duration: {total_edited:.1f}s")
    
    # Save plan
    plan_path = f"{Path(video_path).stem}_plan.json"
    plan.to_json(plan_path)
    print(f"  Saved to: {plan_path}")
    
    # Step 3: Render
    print("\n[Step 3] Rendering video...")
    renderer = FFmpegRenderer()
    output_path = f"{Path(video_path).stem}_edited.mp4"
    
    try:
        renderer.render(plan, output_path)
        print(f"✓ Video rendered: {output_path}")
        renderer.cleanup()
    except Exception as e:
        print(f"✗ Render failed: {e}")
        renderer.cleanup()
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test with one video
    video_path = "input_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        print("Place a test video in the current directory and name it 'input_video.mp4'")
    else:
        test_editor(video_path)