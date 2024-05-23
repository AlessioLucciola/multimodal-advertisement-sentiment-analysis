# Dataset to MP4 Videos Conversion Script
# 
# This is a simple script that serves as an example to convert 
# an rPPG dataset of interest into a folder of MP4 video files 
# that can ultimately be used with OpenFace for further analysis.
#
# See comments and the motion_analysis folder README For more details.
import os, glob
import cv2
import numpy as np
from scipy import io as scio
import shutil
from typing import Dict, Any
from packages.rppg_toolbox.config import DUMP_FRAMES_PATH

# Functions for reading rPPG media of interest and saving frames
# def read_video(video_file: str,
#                max_frames_split: int = 128, 
#                desired_fr: int = 30) -> Dict[str, Any]:
def read_video(video_file: str) -> Dict[str, Any]:
    """Reads a video file, returns frames(T, H, W, 3) """
    if os.path.exists(DUMP_FRAMES_PATH):
        print(f"temp_frames already found, removing it!")
        shutil.rmtree(DUMP_FRAMES_PATH)
    os.makedirs(DUMP_FRAMES_PATH, exist_ok=True)
    print(f"Creating new temp_frames directory!")
    VidObj = cv2.VideoCapture(video_file)
    fps = VidObj.get(cv2.CAP_PROP_FPS)
    max_frames_split = round(fps)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    frames = None
    success, frame = VidObj.read()
    curr_frame = 0
    curr_split = 0
    splits_paths = []
    splits_timestamps = []
    curr_timestamps = [] 
    while success:
        # i += 1
        # print(f"Frame {i}")
        # print(f"Frames_step {frames_step}")
        # if i % frames_step != 0:
        #     continue
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        if frames is None:
            frames = np.expand_dims(np.zeros_like(frame), 0)
            frames = np.repeat(frames, max_frames_split, axis=0)
            print(f"Frames initialization array shape: {frames.shape}")

        timestamp = VidObj.get(cv2.CAP_PROP_POS_FRAMES) / fps

        curr_timestamps.append(timestamp)
        frames[curr_frame] = frame
        success, frame = VidObj.read()
        curr_frame += 1
        if curr_frame == max_frames_split:
            print(f"Split {curr_split} saved!")
            curr_frame = 0
            split_path = os.path.join(DUMP_FRAMES_PATH, f"frames_split_{curr_split}.npy")
            np.save(split_path, frames)
            curr_split += 1
            frames = None
            splits_paths.append(split_path)
            splits_timestamps.append(curr_timestamps)
            curr_timestamps = []
        
    #The last split ended before max_frames_split
    # In this case the last last frames are 0, which is ok since we need to pad in order to have sequences of length 100.
    # We will discard the result anyway since we have 1 prediction per frame, and all the other prediction will be discarded.
    # if curr_frame != 0:
    #     print(f"Split {curr_split} saved!")
    #     # suppress the linting errors
    #     frames: np.ndarray
    #     split_path = os.path.join(DUMP_FRAMES_PATH, f"frames_split_{curr_split}.npy")
    #     np.save(split_path, frames)
    #     splits_paths.append(split_path)
    #     # Padding curr_timestamps with -1 values
    #     splits_timestamps.append(curr_timestamps + [-1 for _ in range(frames.shape[0] - len(curr_timestamps))])

    print(f"read video completed! \n FPS: {fps} | Num Splits: {curr_split}")
    print(f"timestamps lengths: {[len(ts) for ts in splits_timestamps]}")

    return {"splits_paths": splits_paths,
            "splits_timestamps": splits_timestamps,
            "fps": fps}


def read_png_frames(video_file):
    """Reads a video file, returns frames(T, H, W, 3) """
    frames = list()
    all_png = sorted(glob.glob(video_file + '*.png'))
    for png_path in all_png:
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return np.asarray(frames)

def read_mat(mat_file):
    """Reads a video file in the MATLAB format (.mat), returns frames(T,H,W,3)"""
    try:
        mat = scio.loadmat(mat_file)
    except:
        for _ in range(20):
            print(mat_file)
    frames = np.array(mat['video'])
    return frames

def read_npy_video(self, video_file):
    """Reads a video file in the numpy format (.npy), returns frames(T,H,W,3)"""
    frames = np.load(video_file[0])
    if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
        processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
    elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
        processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
    else:
        print("Failed!")
    return np.asarray(processed_frames)

def save_video_frames(frames, video_name, save_path):
    """Saves video frames as an mp4 video file in the save path"""
    os.makedirs(save_path, exist_ok=True)
    height, width, _ = frames[0].shape
    video_name = video_name + ".mp4"
    video_file = os.path.join(save_path, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
    print(f"Video saved: {video_file}")

# Dataset specific processing functions
def process_ubfc_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "subject*")
    for dir in data_dirs:
        frames = read_video(os.path.join(dir,"vid.avi"))
        subject_name = os.path.split(dir)[-1]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")

def process_phys_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "s*" + os.sep + "*.avi")
    for dir in data_dirs:
        frames = read_video(dir)
        subject_name = os.path.split(dir)[-1]
        subject_name = os.path.splitext(subject_name)[0]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")

def process_pure_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "*-*")
    for dir in data_dirs:
        subject_trail_val = os.path.split(dir)[-1]
        video_name = subject_trail_val
        frames = read_png_frames(os.path.join(dir, "", subject_trail_val, ""))
        save_video_frames(frames, video_name, save_path)
    print("All videos saved!")

def process_afrl_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + "*.avi")
    for dir in data_dirs:
        frames = read_video(dir)
        subject_name = os.path.split(dir)[-1]
        subject_name = os.path.splitext(subject_name)[0]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")

def process_mmpd_folder(folder_path, save_path):
    data_dirs = glob.glob(folder_path + os.sep + 'subject*')
    if not data_dirs:
        raise ValueError(self.dataset_name + ' data paths empty!')
    dirs = list()
    for data_dir in data_dirs:
        subject = int(os.path.split(data_dir)[-1][7:])
        mat_dirs = os.listdir(data_dir)
        for mat_dir in mat_dirs:
            index = mat_dir.split('_')[-1].split('.')[0]
            dirs.append({'index': index, 
                            'path': data_dir+os.sep+mat_dir,
                            'subject': subject})
    for dir in dirs:
        frames = read_mat(dir['path'])
        frames = (np.round(frames * 255)).astype(np.uint8)
        subject_name = os.path.split(dir['path'])[-1]
        subject_name = subject_name.split('.')[0]
        save_video_frames(frames, subject_name, save_path)
    print("All videos saved!")


if __name__ == "__main__":
    # The below code is an example of using the above functions to convert the UBFC-rPPG dataset
    # into a folder of MP4s for subsequent analysis by OpenFace.

    # Dataset Paths
    # Change this to point to the location of your dataset
    dataset_path = '/path/to/UBFC-rPPG'

    # Save Paths
    # Change this to point to the location you'd like to save the MP4 videos
    save_path = '/path/to/converted_mp4_videos'

    # Dataset Processing
    process_ubfc_folder(dataset_path, os.path.join(save_path))
