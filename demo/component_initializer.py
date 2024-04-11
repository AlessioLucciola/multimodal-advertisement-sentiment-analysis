import streamlit as st
import pyaudio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import AUDIO_SAMPLE_RATE

def get_audio_stream():
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=AUDIO_SAMPLE_RATE, input=True, frames_per_buffer=3200, input_device_index=1, output_device_index=3)
        return stream
    except (IOError, OSError) as e:
        st.error('Error initializing PyAudio. You might check if the microphone is connected.')
        st.error(e)
        st.title('Debug:')
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            st.write("Device index:", i)
            st.write("Device name:", device_info['name'])
            st.write("Max input channels:", device_info.get('maxInputChannels', 'N/A'))
            st.write("Max output channels:", device_info.get('maxOutputChannels', 'N/A'))
            st.write("Default sample rate:", device_info.get('defaultSampleRate', 'N/A'))
            st.write("__________")
        p.terminate()
        return None