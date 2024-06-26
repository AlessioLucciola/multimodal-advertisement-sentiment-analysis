from demo_config import AUDIO_MODEL_EPOCH, VIDEO_MODEL_EPOCH, AUDIO_MODEL_PATH, VIDEO_MODEL_PATH, AUDIO_IMPORTANCE, PPG_MODEL_PATH, PPG_MODEL_EPOCH
from component_initializer import get_audio_stream, get_video_stream
from demo.demo_utils import create_chart
from st_pages import Page, show_pages
from datetime import datetime
import streamlit as st
import sys
import os
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

# Pages definition
show_pages(
    [
        Page("demo/init.py", "Online demo", "📹"),
        Page("demo/offline_demo.py", "Offline demo", "💡"),
    ]
)

# Initialize required components
audio_stream = get_audio_stream()
video_stream = get_video_stream()
if audio_stream is not None and video_stream is not None:
    is_components_initialized = True
    
# Functions
def start_listening():
    print("Started listening!")
    st.session_state['audio_stream_frames'] = [] # Initialize audio stream frames
    st.session_state['video_stream_frames'] = [] # Initialize video stream frames
    st.session_state['processed_windows'] = None
    st.session_state['run'] = True

def stop_listening():
    print("Stopped listening!")
    st.session_state['run'] = False
    audio_stream_frames = b''.join(st.session_state['audio_stream_frames'])
    video_stream_frames = st.session_state['video_stream_frames']
    st.session_state["processed_windows"] = fusion_main(audio_model_path=AUDIO_MODEL_PATH,
                                                        video_model_path=VIDEO_MODEL_PATH,
                                                        ppg_model_path = PPG_MODEL_PATH,
                                                        audio_model_epoch=AUDIO_MODEL_EPOCH,
                                                        video_model_epoch=VIDEO_MODEL_EPOCH,
                                                        ppg_model_epoch=PPG_MODEL_EPOCH,
                                                        audio_frames=audio_stream_frames,
                                                        video_frames=video_stream_frames,
                                                        live_demo=True,
                                                        audio_importance=AUDIO_IMPORTANCE)

while st.session_state['run']:
    try:
        # Audio stream reading
        data = audio_stream.read(12000)
        st.session_state['audio_stream_frames'].append(data)  # Append data to audio stream frames
        
        # Video stream reading
        current_time = datetime.now()
        video_frame = next(video_stream)
        st.session_state['video_stream_frames'].append(tuple((video_frame, current_time)))  # Append data to video stream frames
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
        #print(st.session_state["processed_windows"])

        # Create the chart
        audio_video_windows, ppg_windows = st.session_state["processed_windows"]
        
        if audio_video_windows is not None:
            chart, legend = create_chart(audio_video_windows, title="Emotion with Video/Audio")
            # Render the chart using Streamlit
            st.altair_chart(chart, use_container_width=True)
            st.altair_chart(legend, use_container_width=True)
        if ppg_windows is not None:
            chart, legend = create_chart(ppg_windows, title="Emotion with PPG")
            st.altair_chart(legend, use_container_width=True)

        st.text("Processed windows debug:")
        st.write(audio_video_windows)

