import glob
import os
from math import ceil
from multiprocessing import Process, Manager

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
# Source code: https://github.com/serengil/retinaface
from retinaface import RetinaFace


def get_frames_from_vid(vid_path: str) -> np.ndarray:
    pass


def parse_frames(frames: np.ndarray, data_format: str) -> np.ndarray:
    # if data_format == 'NDCHW':
    #     frames = np.transpose(frames, (0, 3, 1, 2))
    # elif data_format == 'NCDHW':
    #     frames = np.transpose(frames, (3, 0, 1, 2))
    # elif data_format == 'NDHWC':
    #     pass
    # else:
    #     raise ValueError('Unsupported Data Format!')
        
    # frames = np.transpose(frames, (0, 3, 1, 2))
    frames = np.transpose(frames, (0, 1, 4, 2, 3))
    return frames.astype(np.float32)


def read_npy_video(video_file):
    """Reads a video file in the numpy format (.npy), returns frames(T,H,W,3)"""
    frames = np.load(video_file[0])
    if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
        processed_frames = [frame.astype(np.uint8)[..., :3]
                                         for frame in frames]
    elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
        processed_frames = [
            (np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
    else:
        raise Exception(f'Loaded frames are of an incorrect type or range of values! '
                        + f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
    return np.asarray(processed_frames)

def preprocess_frames(frames: np.ndarray, config_preprocess):
    """Preprocesses a pair of data.

    Args:
        frames(np.array): Frames in a video.
        config_preprocess(CfgNode): preprocessing settings(ref:config.py).
    Returns:
        frame_clips(np.array): processed video data by frames
    """
    # resize frames and crop for face region
    frames = crop_face_resize(
        frames,
        config_preprocess.CROP_FACE.DO_CROP_FACE,
        config_preprocess.CROP_FACE.BACKEND,
        config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
        config_preprocess.CROP_FACE.LARGE_BOX_COEF,
        config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
        config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
        config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
        config_preprocess.RESIZE.W,
        config_preprocess.RESIZE.H)
    # Check data transformation type
    data = list()  # Video data
    for data_type in config_preprocess.DATA_TYPE:
        f_c = frames.copy()
        if data_type == "Raw":
            data.append(f_c)
        elif data_type == "DiffNormalized":
            data.append(diff_normalize_data(f_c))
        elif data_type == "Standardized":
            data.append(standardized_data(f_c))
        else:
            raise ValueError("Unsupported data type!")
    data = np.concatenate(data, axis=-1)  # concatenate all channels

    if config_preprocess.DO_CHUNK:  # chunk data into snippets
        frames_clips = chunk(
            frames=data,
            chunk_length=config_preprocess.CHUNK_LENGTH)
    else:
        frames_clips = np.array([data])

    return frames_clips


def face_detection(frame, backend, use_larger_box=False, larger_box_coef=1.0):
    """Face detection on a single frame.

    Args:
        frame(np.array): a single frame.
        backend(str): backend to utilize for face detection.
        use_larger_box(bool): whether to use a larger bounding box on face detection.
        larger_box_coef(float): Coef. of larger box.
    Returns:
        face_box_coor(List[int]): coordinates of face bouding box.
    """
    if backend == "HC":
        # Use OpenCV's Haar Cascade algorithm implementation for face detection
        # This should only utilize the CPU
        detector = cv2.CascadeClassifier(
            './dataset/haarcascade_frontalface_default.xml')

        # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
        # (x,y) corresponds to the top-left corner of the zone to define using
        # the computed width and height.
        face_zone = detector.detectMultiScale(frame)

        if len(face_zone) < 1:
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        elif len(face_zone) >= 2:
            # Find the index of the largest face zone
            # The face zones are boxes, so the width and height are the same
            max_width_index = np.argmax(
                face_zone[:, 2])  # Index of maximum width
            face_box_coor = face_zone[max_width_index]
            print(
                "Warning: More than one faces are detected. Only cropping the biggest one.")
        else:
            face_box_coor = face_zone[0]
    elif backend == "RF":
        # Use a TensorFlow-based RetinaFace implementation for face detection
        # This utilizes both the CPU and GPU
        res = RetinaFace.detect_faces(frame)
        print(f"{len(res)} face detected: {res}")
        if isinstance(res, tuple):
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        elif len(res) > 0:
            # Pick the highest score
            highest_score_face = max(res.values(), key=lambda x: x['score'])
            face_zone = highest_score_face['facial_area']

            # This implementation of RetinaFace returns a face_zone in the
            # form [x_min, y_min, x_max, y_max] that corresponds to the
            # corners of a face zone
            x_min, y_min, x_max, y_max = face_zone

            # Convert to this toolbox's expected format
            # Expected format: [x_coord, y_coord, width, height]
            x = x_min
            y = y_min
            width = x_max - x_min
            height = y_max - y_min

            # Find the center of the face zone
            center_x = x + width // 2
            center_y = y + height // 2

            # Determine the size of the square (use the maximum of width and height)
            square_size = max(width, height)

            # Calculate the new coordinates for a square face zone
            new_x = center_x - (square_size // 2)
            new_y = center_y - (square_size // 2)
            face_box_coor = [new_x, new_y, square_size, square_size]
        else:
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
    else:
        raise ValueError("Unsupported face detection backend!")

    if use_larger_box:
        face_box_coor[0] = max(0, face_box_coor[0] - \
                               (larger_box_coef - 1.0) / 2 * face_box_coor[2])
        face_box_coor[1] = max(0, face_box_coor[1] - \
                               (larger_box_coef - 1.0) / 2 * face_box_coor[3])
        face_box_coor[2] = larger_box_coef * face_box_coor[2]
        face_box_coor[3] = larger_box_coef * face_box_coor[3]
    return face_box_coor

def crop_face_resize(frames,
                     use_face_detection,
                     backend,
                     use_larger_box,
                     larger_box_coef,
                     use_dynamic_detection,
                     detection_freq,
                     use_median_box,
                     width,
                     height):
    """Crop face and resize frames.

    Args:
        frames(np.array): Video frames.
        use_dynamic_detection(bool): If False, all the frames use the first frame's bouding box to crop the faces
                                     and resizing.
                                     If True, it performs face detection every "detection_freq" frames.
        detection_freq(int): The frequency of dynamic face detection e.g., every detection_freq frames.
        width(int): Target width for resizing.
        height(int): Target height for resizing.
        use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
        use_face_detection(bool):  Whether crop the face.
        larger_box_coef(float): the coefficient of the larger region(height and weight),
                            the middle point of the detected region will stay still during the process of enlarging.
    Returns:
        resized_frames(list[np.array(float)]): Resized and cropped frames
    """
    # Face Cropping
    if use_dynamic_detection:
        num_dynamic_det = ceil(frames.shape[0] / detection_freq)
    else:
        num_dynamic_det = 1
    face_region_all = []
    # Perform face detection by num_dynamic_det" times.
    for idx in range(num_dynamic_det):
        if use_face_detection:
            face_region_all.append(face_detection(
                frames[detection_freq * idx], backend, use_larger_box, larger_box_coef))
        else:
            face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
    face_region_all = np.asarray(face_region_all, dtype='int')
    if use_median_box:
        # Generate a median bounding box based on all detected face regions
        face_region_median = np.median(face_region_all, axis=0).astype('int')

    # Frame Resizing
    resized_frames = np.zeros((frames.shape[0], height, width, 3))
    for i in range(0, frames.shape[0]):
        frame = frames[i]
        # use the (i // detection_freq)-th facial region.
        if use_dynamic_detection:
            reference_index = i // detection_freq
        else:  # use the first region obtrained from the first frame.
            reference_index = 0
        if use_face_detection:
            if use_median_box:
                face_region = face_region_median
            else:
                face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                          max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
        resized_frames[i] = cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frames


def chunk(frames, chunk_length):
    """Chunk the data into small chunks.

    Args:
        frames(np.array): video frames.
        chunk_length(int): the length of each chunk.
    Returns:
        frames_clips: all chunks of face cropped frames
    """

    clip_num = frames.shape[0] // chunk_length
    frames_clips = [
        frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
    return np.array(frames_clips)


def save(frames_clips, filename):
    """Save all the chunked data.

    Args:
        frames_clips(np.array): blood volumne pulse (PPG) labels.
        filename: name the filename
    Returns:
        count: count of preprocessed data
    """

    if not os.path.exists(cached_path):
        os.makedirs(cached_path, exist_ok=True)
    count = 0
    for i in range(len(frames_clips)):
        input_path_name = cached_path + os.sep + \
            "{0}_input{1}.npy".format(filename, str(count))
        inputs.append(input_path_name)
        np.save(input_path_name, frames_clips[i])
        count += 1
    return count


def save_multi_process(frames_clips, filename):
    """Save all the chunked data with multi-thread processing.

    Args:
        frames_clips(np.array): blood volumne pulse (PPG) labels.
        filename: name the filename
    Returns:
        input_path_name_list: list of input path names
    """
    if not os.path.exists(cached_path):
        os.makedirs(cached_path, exist_ok=True)
    count = 0
    input_path_name_list = []
    for i in range(len(frames_clips)):
        input_path_name = cached_path + os.sep + \
            "{0}_input{1}.npy".format(filename, str(count))
        input_path_name_list.append(input_path_name)
        np.save(input_path_name, frames_clips[i])
        count += 1
    return input_path_name_list


def multi_process_manager(data_dirs, config_preprocess, multi_process_quota=1):
    """Allocate dataset preprocessing across multiple processes.

    Args:
        data_dirs(List[str]): a list of video_files.
        config_preprocess(Dict): a dictionary of preprocessing configurations
        multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
    Returns:
        file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
    """
    print('Preprocessing dataset...')
    file_num = len(data_dirs)
    choose_range = range(0, file_num)
    pbar = tqdm(list(choose_range))

    # shared data resource
    manager = Manager()  # multi-process manager
    # dictionary for all processes to store processed files
    file_list_dict = manager.dict()
    p_list = []  # list of processes
    running_num = 0  # number of running processes

    # in range of number of files to process
    for i in choose_range:
        process_flag = True
        while process_flag:  # ensure that every i creates a process
            if running_num < multi_process_quota:  # in case of too many processes
                # send data to be preprocessing task
                # # TODO: remove multiprocessing since everything crashes
                # preprocess_dataset_subprocess(data_dirs, config_preprocess, i, file_list_dict)

                p = Process(target=preprocess_dataset_subprocess,
                            args=(data_dirs, config_preprocess, i, file_list_dict))
                p.start()
                p_list.append(p)
                running_num += 1
                print(f"Running preprocessing processes: {running_num}")
                process_flag = False
            for p_ in p_list:
                if not p_.is_alive():
                    p_list.remove(p_)
                    p_.join()
                    running_num -= 1
                    pbar.update(1)
    # join all processes
    for p_ in p_list:
        p_.join()
        pbar.update(1)
    pbar.close()

    return file_list_dict


def build_file_list(file_list_dict):
    """Build a list of files used by the dataloader for the data split. Eg. list of files used for 
    train / val / test. Also saves the list to a .csv file.

    Args:
        file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
    Returns:
        None (this function does save a file-list .csv file to file_list_path)
    """
    file_list = []
    # iterate through processes and add all processed file paths
    for process_num, file_paths in file_list_dict.items():
        file_list = file_list + file_paths

    if not file_list:
        raise ValueError(dataset_name, 'No files in file list')

    file_list_df = pd.DataFrame(file_list, columns=['input_files'])
    os.makedirs(os.path.dirname(file_list_path), exist_ok=True)
    file_list_df.to_csv(file_list_path)  # save file list to .csv


def build_file_list_retroactive(data_dirs, begin, end):
    """ If a file list has not already been generated for a specific data split build a list of files 
    used by the dataloader for the data split. Eg. list of files used for 
    train / val / test. Also saves the list to a .csv file.

    Args:
        data_dirs(List[str]): a list of video_files.
        begin(float): index of begining during train/val split.
        end(float): index of ending during train/val split.
    Returns:
        None (this function does save a file-list .csv file to file_list_path)
    """

    # get data split based on begin and end indices.
    data_dirs_subset = split_raw_data(data_dirs)

    # generate a list of unique raw-data file names
    filename_list = []
    for i in range(len(data_dirs_subset)):
        filename_list.append(data_dirs_subset[i]['index'])
    filename_list = list(set(filename_list))  # ensure all indexes are unique

    # generate a list of all preprocessed / chunked data files
    file_list = []
    for fname in filename_list:
        processed_file_data = list(
            glob.glob(cached_path + os.sep + "{0}_input*.npy".format(fname)))
        file_list += processed_file_data

    if not file_list:
        raise ValueError(dataset_name,
                         'File list empty. Check preprocessed data folder exists and is not empty.')

    file_list_df = pd.DataFrame(file_list, columns=['input_files'])
    os.makedirs(os.path.dirname(file_list_path), exist_ok=True)
    file_list_df.to_csv(file_list_path)  # save file list to .csv


def load_preprocessed_data():
    """ Loads the preprocessed data listed in the file list.

    Args:
        None
    Returns:
        None
    """
    file_list_path = file_list_path  # get list of files in
    file_list_df = pd.read_csv(file_list_path)
    inputs = file_list_df['input_files'].tolist()
    if not inputs:
        raise ValueError(dataset_name + ' dataset loading data error!')
    inputs = sorted(inputs)  # sort input file name list
    inputs = inputs
    preprocessed_data_len = len(inputs)


def diff_normalize_data(data):
    """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros(
        (diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
            data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(
        diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


def standardized_data(data):
    """Z-score standardization for video data."""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


def resample_ppg(input_signal, target_length):
    """Samples a PPG sequence into specific length."""
    return np.interp(
        np.linspace(
            1, input_signal.shape[0], target_length), np.linspace(
            1, input_signal.shape[0], input_signal.shape[0]), input_signal)
