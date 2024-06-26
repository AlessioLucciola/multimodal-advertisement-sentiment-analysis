import streamlit as st
import imageio
import pyaudio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import AUDIO_SAMPLE_RATE
import imageio

# Initialize PyAudio
p = pyaudio.PyAudio()

# Display audio device information in cli
def display_audio_device_info():
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print("Device index:", i)
        print("Device name:", device_info['name'])
        print("Max input channels:", device_info.get('maxInputChannels', 'N/A'))
        print("Max output channels:", device_info.get('maxOutputChannels', 'N/A'))
        print("Default sample rate:", device_info.get('defaultSampleRate', 'N/A'))
        print("__________")
display_audio_device_info()

def get_audio_stream():
    try:
        stream = p.open(format=pyaudio.paFloat32, channels=1, 
                        rate=AUDIO_SAMPLE_RATE, input=True, 
                        frames_per_buffer=2500, 
                        input_device_index=0, # Alessio: 1 | Danilo: 1 | Domiziano: 1
                        output_device_index=1 # Alessio: 3 | Danilo: 3 | Domiziano: 4
                        ) 
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

def get_video_stream():
    try :
        stream = imageio.v3.imiter('<video0>') # Alessio: '<video0>' | Danilo: '<video0>'
        print(f"stream is: {stream}")
        return stream
    except Exception as e:
        st.error('Error initializing the video stream.')
        st.error(e)

        return None
