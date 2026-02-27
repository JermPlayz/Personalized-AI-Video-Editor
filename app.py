"""
Streamlit UI for video editing pipeline
"""

import streamlit as st
import json
from pathlib import Path
from analysis_modules import VideoAnalyzer
from edit_plan import EditPlanBuilder
from renderer import FFmpegRenderer

st.set_page_config(page_title="AI Video Editor", layout="wide")

st.title("🎬 AI Video Editor - Base Pipeline")

# Sidebar for configuration
st.sidebar.header("Configuration")
whisper_model = st.sidebar.selectbox("Whisper Model", ["tiny", "base", "small"])
format_type = st.sidebar.selectbox("Output Format", ["9:16 Vertical", "16:9 Horizontal"])
target_length = st.sidebar.slider("Target Length (seconds)", 15, 300, 60)

# Main tab interface
tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Analyze", "Edit Plan", "Render"])

with tab1:
    st.header("Step 1: Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file:
        video_path = f"temp_{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✓ Video saved: {video_path}")
        st.session_state.video_path = video_path

with tab2:
    st.header("Step 2: Analyze Video")
    
    if "video_path" in st.session_state:
        if st.button("Run Analysis"):
            with st.spinner("Analyzing video..."):
                analyzer = VideoAnalyzer(whisper_model=whisper_model)
                analysis = analyzer.analyze_full(st.session_state.video_path)
                
                st.session_state.analysis = analysis
                st.success("✓ Analysis complete!")
                
                # Display analysis results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silences Found", len(analysis["audio"]["silences"]))
                with col2:
                    st.metric("Scene Changes", len(analysis["scenes"]))
                with col3:
                    st.metric("Words", len(analysis["transcription"]["words"]))
                
                st.subheader("Transcription Preview")
                st.text(analysis["transcription"]["full_text"][:500] + "...")

with tab3:
    st.header("Step 3: Build Edit Plan")
    
    if "analysis" in st.session_state:
        st.subheader("Editing Rules")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            remove_silence = st.checkbox("Remove Silences", value=True)
            silence_threshold = st.slider("Silence Threshold (s)", 0.1, 2.0, 0.5)
        
        with col2:
            use_scenes = st.checkbox("Use Scene Cuts", value=True)
            min_segment = st.slider("Min Segment (s)", 0.1, 2.0, 0.5)
        
        with col3:
            add_zoom = st.checkbox("Add Zoom on Motion", value=False)
            motion_threshold = st.slider("Motion Threshold", 1.0, 10.0, 5.0)
        
        if st.button("Build Edit Plan"):
            builder = EditPlanBuilder(st.session_state.analysis, st.session_state.video_path)
            
            if remove_silence:
                builder.apply_silence_removal_rule(silence_threshold)
            if add_zoom:
                builder.apply_motion_emphasis_rule(motion_threshold)
            
            plan = builder.get_plan()
            st.session_state.edit_plan = plan
            
            st.success(f"✓ Edit plan created with {len(plan.timeline)} segments")
            st.json(plan.to_json("temp_plan.json"))

with tab4:
    st.header("Step 4: Render Video")
    
    if "edit_plan" in st.session_state:
        if st.button("Render Final Video"):
            with st.spinner("Rendering... This may take a while"):
                renderer = FFmpegRenderer()
                output_path = "output_edited.mp4"
                renderer.render(st.session_state.edit_plan, output_path)
                renderer.cleanup()
                
                st.success(f"✓ Video rendered: {output_path}")
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Edited Video",
                        data=f.read(),
                        file_name="edited_video.mp4",
                        mime="video/mp4"
                    )