from demo_config import AUDIO_MODEL_EPOCH, VIDEO_MODEL_EPOCH, AUDIO_MODEL_PATH, VIDEO_MODEL_PATH
from demo.demo_utils import create_chart
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from fusion.fusion_main import main as fusion_main

st.title('Offline Emotion Recognition')
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as temp_video:
        temp_video.write(uploaded_file.read())

    processed_windows = fusion_main(audio_model_path=AUDIO_MODEL_PATH,
                video_model_path=VIDEO_MODEL_PATH,
                audio_model_epoch=AUDIO_MODEL_EPOCH,
                video_model_epoch=VIDEO_MODEL_EPOCH,
                video_frames="temp_video.mp4",
                audio_frames=os.path.join("data", "AUDIO", "recorded_audio.wav"),
                live_demo=False,
                get_audio_from_video=True
                )
    
    #print(processed_windows)
    
    os.remove("temp_video.mp4")

    if processed_windows is not None:
        chart, legend = create_chart(processed_windows)

        # Render the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)
        st.write("Legend:")
        st.write(
            """ 
                Neutral: Grey,
                Positive: Blue,
                Negative: Orange
            """)

        st.text("Processed windows debug:")
        st.write(processed_windows)