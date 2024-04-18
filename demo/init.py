from component_initializer import get_audio_stream, get_video_stream
from datetime import datetime
import streamlit as st
import imageio.v3
import sys
import os
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from fusion.fusion_main import main as fusion_main

# Session states and other variables
is_components_initialized = False
if 'run' not in st.session_state:
    st.session_state['text'] = 'Listening...'
    st.session_state['run'] = False
    st.session_state['audio_stream_frames'] = []
    st.session_state['video_stream_frames'] = []
    st.session_state['processed_windows'] = None

# Initialize required components
audio_stream = get_audio_stream()
with st.sidebar:
    st.title('Settings')
    audio_stream = get_audio_stream()
    if audio_stream is not None:
        is_components_initialized = True
    
    video_stream = get_video_stream()
    
# Functions
def start_listening():
    st.session_state['audio_stream_frames'] = [] # Initialize audio stream frames
    st.session_state['video_stream_frames'] = [] # Initialize video stream frames
    st.session_state['processed_windows'] = None
    st.session_state['run'] = True

def stop_listening():
    st.session_state['run'] = False
    audio_stream_frames = b''.join(st.session_state['audio_stream_frames'])
    video_stream_frames = st.session_state['video_stream_frames']
    st.session_state["processed_windows"] = fusion_main(audio_model_path="AudioNetCT_2024-04-18_11-09-07",
                                                        video_model_path=None,
                                                        audio_model_epoch=155,
                                                        video_model_epoch=None,
                                                        audio_frames=audio_stream_frames,
                                                        video_frames=video_stream_frames,
                                                        live_demo=True)

while st.session_state['run']:
    try:
        data = audio_stream.read(12000)
        st.session_state['audio_stream_frames'].append(data)  # Append data to audio stream frames
        # Capture video frame
        # TO DO: previous code was capturing video frames here but it didn't work properly because it conflicted with the audio recorder
        
        st.session_state['video_stream_frames'].append("")  # Append data to video stream frames
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
        <div style="background-color: #363637; padding: 20px; border-radius: 10px; color: white;">
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
    col1, col2 = st.columns([1, 4])
    col1.button('Start recording', on_click=start_listening, key='start_button')
    col2.button('Stop recording', on_click=stop_listening, key='stop_button')

    if "processed_windows" in st.session_state and st.session_state["processed_windows"] is not None:
        st.text("Processed audio windows:")
        st.write(st.session_state["processed_windows"])
