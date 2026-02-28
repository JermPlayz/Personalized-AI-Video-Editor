"""
Streamlit UI - wraps the editor
"""

import streamlit as st
from analysis_modules import VideoAnalyzer
from edit_plan import EditPlanBuilder
from renderer import FFmpegRenderer
import os

st.title("🎬 AI Video Editor")

uploaded_file = st.file_uploader("Choose a video", type=["mp4"])

if uploaded_file:
    # Save uploaded file
    video_path = f"temp_{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Analyze
    if st.button("Analyze & Edit"):
        analyzer = VideoAnalyzer()
        analysis = analyzer.analyze_full(video_path)
        
        builder = EditPlanBuilder(analysis, video_path)
        builder.apply_top_highlights_rule(max_segments=8)
        builder.apply_motion_emphasis_rule()
        plan = builder.get_plan()
        
        renderer = FFmpegRenderer()
        output_path = "output_edited.mp4"
        renderer.render(plan, output_path)
        renderer.cleanup()
        
        # Download
        with open(output_path, "rb") as f:
            st.download_button("Download", f.read(), "edited.mp4", "video/mp4")