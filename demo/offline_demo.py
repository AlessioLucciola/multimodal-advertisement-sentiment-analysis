from demo_config import AUDIO_MODEL_EPOCH, VIDEO_MODEL_EPOCH, AUDIO_MODEL_PATH, VIDEO_MODEL_PATH
import streamlit as st
import altair as alt
import pandas as pd
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
    
    print(processed_windows)
    
    os.remove("temp_video.mp4")

    if processed_windows is not None:
        df = pd.DataFrame(processed_windows)
        df = df.drop(columns=['logits'])
        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y('window_type:N', title='Window Type'),
            x=alt.X('start_time:Q', title='Time'),
            x2='end_time:Q',
            color=alt.Color('emotion_string:N', legend=None),
            tooltip=['start_time', 'end_time', 'emotion_string']
        ).properties(
            width=600,
            height=200
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )

        legend = chart.mark_rect().encode(
            y=alt.Y('emotion_string:N', axis=alt.Axis(orient='right')),
            color=alt.Color('emotion_string:N', scale=alt.Scale(scheme='category20'), legend=None),
        ).properties(
            title='Emotion'
        )

        # Render the chart using Streamlit
        st.altair_chart(chart, use_container_width=True)
        

        st.text("Processed windows debug:")
        st.write(processed_windows)