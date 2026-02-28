"""
Test the upgraded AI Video Editor
Highlight-Scoring Version
"""

import os
from pathlib import Path
from analysis_modules import VideoAnalyzer
from edit_plan import EditPlanBuilder
from renderer import FFmpegRenderer


def test_editor(video_path: str):
    """Test: Analyze → Score → Build Plan → Render"""

    print(f"\n{'='*60}")
    print(f"Testing: {video_path}")
    print(f"{'='*60}")

    # =============================
    # Step 1: Analyze (New System)
    # =============================
    print("\n[Step 1] Analyzing video with AI highlight scoring...")

    analyzer = VideoAnalyzer()  # New version does not take constructor args
    analysis = analyzer.analyze_full(
        video_path,
        output_json=f"{Path(video_path).stem}_analysis.json"
    )

    segments = analysis.get("segments", [])

    print(f"\n✓ Analysis complete:")
    print(f"  Total segments found: {len(segments)}")

    if segments:
        print(f"\n  Top 3 highlight segments:")
        for i, seg in enumerate(segments[:3]):
            print(
                f"    [{i+1}] "
                f"{seg['start']:.2f}s - {seg['end']:.2f}s "
                f"({seg['duration']:.2f}s) | "
                f"Score: {seg.get('highlight_score', 0):.2f}"
            )
            print(f"         {seg['text'][:60]}...")

    # =============================
    # Step 2: Build Edit Plan
    # =============================
    print("\n[Step 2] Building edit plan from top highlights...")

    builder = EditPlanBuilder(analysis, video_path)

    # NEW RULES
    builder.apply_top_highlights_rule(max_segments=8)
    builder.apply_motion_emphasis_rule()

    plan = builder.get_plan()

    print(f"✓ Edit plan created:")
    print(f"  Timeline segments: {len(plan.timeline)}")

    total_edited = sum(
        seg.out_time - seg.in_time
        for seg in plan.timeline
    )

    print(f"  Total edited duration: {total_edited:.1f}s")

    # Save plan
    plan_path = f"{Path(video_path).stem}_plan.json"
    plan.to_json(plan_path)
    print(f"  Saved to: {plan_path}")

    # =============================
    # Step 3: Render
    # =============================
    print("\n[Step 3] Rendering video...")

    renderer = FFmpegRenderer()
    output_path = f"{Path(video_path).stem}_edited.mp4"

    try:
        renderer.render(plan, output_path)
        print(f"✓ Video rendered successfully: {output_path}")
    except Exception as e:
        print(f"✗ Render failed: {e}")
    finally:
        renderer.cleanup()

    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    video_path = "input_video.mp4"

    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        print("Place a test video in the current directory and name it 'input_video.mp4'")
    else:
        test_editor(video_path)