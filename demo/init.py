from component_initializer import get_audio_stream
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
    
# Functions
def start_listening():
    st.session_state['audio_stream_frames'] = []  # Initialize audio stream frames
    st.session_state['run'] = True

def stop_listening():
    st.session_state['run'] = False
    audio_stream_frames = b''.join(st.session_state['audio_stream_frames'])
    print(audio_main(model_path="AudioNetCT_2024-04-08_17-00-51", audio_file=audio_stream_frames, epoch=484, live_demo=True))

while st.session_state['run']:
    try:
        data = audio_stream.read(3200)
        st.session_state['audio_stream_frames'].append(data)  # Append data to audio stream frames
    except Exception as e:
        st.error('Error recording the audio')
        st.error(e)

# Layout
if (is_components_initialized):
    st.title('📹 Youtube Advertisement Emotion Recognition')
    col1, col2 = st.columns(2)

    col1.button('Start', on_click=start_listening)
    col2.button('Stop', on_click=stop_listening)

