from demo_config import AUDIO_MODEL_EPOCH, VIDEO_MODEL_EPOCH, AUDIO_MODEL_PATH, VIDEO_MODEL_PATH, AUDIO_IMPORTANCE, PPG_MODEL_PATH, PPG_MODEL_EPOCH
from demo_utils import create_chart
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

    processed_windows, ppg_windows = fusion_main(audio_model_path=AUDIO_MODEL_PATH,
                video_model_path=VIDEO_MODEL_PATH,
                ppg_model_path=PPG_MODEL_PATH,
                audio_model_epoch=AUDIO_MODEL_EPOCH,
                video_model_epoch=VIDEO_MODEL_EPOCH,
                ppg_model_epoch=PPG_MODEL_EPOCH,
                video_frames="temp_video.mp4",
                audio_frames=os.path.join("data", "AUDIO", "recorded_audio.wav"),
                live_demo=False,
                get_audio_from_video=True,
                audio_importance=AUDIO_IMPORTANCE
                )
    
    #print(processed_windows)
    os.remove("temp_video.mp4")

    if processed_windows is not None:
        chart, legend = create_chart(processed_windows, title="Emotion with Audio/Video")

        # Render the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)
        st.altair_chart(legend, use_container_width=True)

    if ppg_windows is not None:
        chart, legend = create_chart(ppg_windows, title="Emotion with PPG signal")
        # Render the chart using Streamlit
        # st.altair_chart(chart, use_container_width=True)
        st.altair_chart(legend, use_container_width=True)

    st.text("Processed windows debug:")
    st.write(processed_windows)
    st.text("Processed windows debug:")
    st.write(ppg_windows)
