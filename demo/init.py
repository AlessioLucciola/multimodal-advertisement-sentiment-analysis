from component_initializer import get_audio_stream
from camera_input_live import camera_input_live
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from fusion.audio_processing import main as audio_main

# Session states and other variables
is_components_initialized = False
if 'text' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False
    st.session_state['audio_stream_frames'] = []

# Initialize required components
audio_stream = get_audio_stream()
if audio_stream is not None:
    is_components_initialized = True
video_stream = camera_input_live()
    
# Functions
def start_listening():
    st.session_state['audio_stream_frames'] = [] # Initialize audio stream frames
    st.session_state['video_stream_frames'] = [] # Initialize video stream frames
    st.session_state['run'] = True

def stop_listening():
    st.session_state['run'] = False
    audio_stream_frames = b''.join(st.session_state['audio_stream_frames'])
    video_stream_frames = st.session_state['video_stream_frames']
    print(audio_main(model_path="AudioNetCT_2024-04-08_17-00-51", audio_file=audio_stream_frames, epoch=484, live_demo=True))
    print(len(video_stream_frames))

while st.session_state['run']:
    try:
        data = audio_stream.read(8000)
        st.session_state['audio_stream_frames'].append(data)  # Append data to audio stream frames
        st.session_state['video_stream_frames'].append(video_stream)  # Append data to video stream frames
    except Exception as e:
        st.error('Error recording the audio')
        st.error(e)

# Layout
if (is_components_initialized):
    st.title('📹 Youtube Advertisement Emotion Recognition')
    video_player = st.video(data="https://www.youtube.com/watch?v=EDM0JYfsQwc&ab_channel=AdTV", start_time=0)
    st.header("How to build a successful emotion recognition system")
    st.markdown(
        """
        <div style="background-color: #363637; padding: 20px; border-radius: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <p style="text-align: left;">64k views</p>
                <p style="text-align: right;">5 years ago</p>
            </div>
            <p style="text-align: left;">In this comprehensive video guide, we delve into the intricate process of developing an effective emotion recognition system. Emotion recognition, a crucial component of artificial intelligence and human-computer interaction, holds immense potential in various fields, including healthcare</p>
            <p>...more</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    col1.button('Stop recording', on_click=stop_listening, key='stop_button')
    col2.button('Start recording', on_click=start_listening, key='start_button')


