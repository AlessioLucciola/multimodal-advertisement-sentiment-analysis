""" The main function of rPPG deep learning pipeline."""
import argparse
import random

import os
import numpy as np
import torch
from packages.rppg_toolbox.config import get_config, DUMP_FRAMES_PATH
from packages.rppg_toolbox.dataset.data_loader.CustomLoader import CustomLoader
from packages.rppg_toolbox.neural_methods.trainer.CustomTrainer import CustomTrainer
from packages.rppg_toolbox.dataset.data_loader.InferenceOnlyBaseLoader import InferenceOnlyBaseLoader
from packages.rppg_toolbox.neural_methods.trainer.BaseTrainer import BaseTrainer
from packages.rppg_toolbox.utils import preprocess
from packages.rppg_toolbox.tools.motion_analysis.convert_dataset_to_mp4 import read_video
from packages.rppg_toolbox.evaluation.post_process import get_bvp
from torch.utils.data import DataLoader
from packages.rppg_toolbox.utils.plot import plot_signal
import shutil
from typing import List, Tuple, Optional, Dict, Any, Any
from datetime import datetime

RANDOM_SEED =  42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="packages/rppg_toolbox/configs/infer_configs/UBFC-rPPG_UBFC-PHYS_DEEPPHYS_BASIC_CUSTOM.yaml", type=str, help="The name of the model.")
    return parser

def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == 'CUSTOM':
        model_trainer = CustomTrainer(config)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)

def run():
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = BaseTrainer.add_trainer_args(parser)
    parser = InferenceOnlyBaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    data_loader_dict = dict() # dictionary of data loaders 
    if config.TOOLBOX_MODE != "only_test":
        raise ValueError("Only 'only_test' supported for this smaller version of the toolbox")
    if config.TEST.DATA.DATASET != "CUSTOM":
        raise NotImplementedError("Only custom dataset is supported on this smaller version")
    # if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
    #     raise ValueError("Dataset and datapath have to be defined") 
    test_loader = CustomLoader
        
    test_data = test_loader(
        name="test",
        data_path=config.TEST.DATA.DATA_PATH,
        config_data=config.TEST.DATA)
    data_loader_dict["test"] = DataLoader(
        dataset=test_data,
        num_workers=16,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=general_generator
    )
    test(config, data_loader_dict)

def extract_ppg_from_video(vid_path: Optional[str | List] = None) -> Tuple[torch.Tensor, List[List[float]]]:
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = BaseTrainer.add_trainer_args(parser)
    parser = InferenceOnlyBaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)

    model_trainer = CustomTrainer(config)
    if vid_path is None: 
        vid_path = "/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Multimodal Interaction/Project/multimodal-interaction-project/packages/rppg_toolbox/data/InferenceVideos/RawData/video1/my_video.mp4"
    if isinstance(vid_path, str):
        # Video is a path, we are in an offline demo
        video_data = read_video(vid_path)
    else:
        # Video is an array of frames, we are in online demo
        video_data = parse_live_video_frames(vid_path)
    bvps = torch.tensor([])
    timestamps = video_data["splits_timestamps"]
    for split_path, split_timestamps in zip(video_data["splits_paths"], timestamps):
        print(f"Extracting ppg from split at: {split_path}.")
        if isinstance(split_path, str):
            raw_frames = np.load(split_path)
        else:
            raw_frames = split_path
        assert len(split_timestamps) == raw_frames.shape[0], f"ERROR: timestamps and frames must be of the same length | timestamp of length {len(split_timestamps)}, split of length {raw_frames.shape[0]}"
        frames = preprocess.preprocess_frames(raw_frames, config.TEST.DATA.PREPROCESS, chunk_length=video_data["fps"])
        # print(f"preprocessed frames shape: {frames.shape}")
        frames = preprocess.parse_frames(frames, data_format="NDCHW")
        # print(f"parsed frames shape: {frames.shape}")
        output = model_trainer.test_from_frames(frames).detach() # shape: [num_chunks, 100]
        # plot_signal(output.reshape(-1).numpy(), "model output")

        for item in output:
            npy_bvp = get_bvp(item.squeeze(), diff_flag=True, bandpass=True, fs=round(video_data["fps"]))
            bvp = torch.tensor(npy_bvp.copy()).to(torch.float32)
            bvps = torch.cat((bvps, bvp.view(1, -1)), dim=0)

        # shape: [num_chunks * num_splits, 100]
        print(f"ppgs {bvps} with shape {bvps.shape}")

    shutil.rmtree(DUMP_FRAMES_PATH)
    print(f"Removed temp_frames directory")
    return bvps, timestamps


def parse_live_video_frames(video_frames: list)-> Dict[Any, Any]:
    DESIRED_FR = 128
    splits = []
    timestamps = []
    curr_frames = []
    curr_timestamps = []
    #N * 2 * 480 * 640 * 3
    if os.path.exists(DUMP_FRAMES_PATH):
        print(f"temp_frames already found, removing it!")
        shutil.rmtree(DUMP_FRAMES_PATH)
    os.makedirs(DUMP_FRAMES_PATH, exist_ok=True)
    print(f"Creating new temp_frames directory!")
    
    total_time = datetime.timestamp(video_frames[-1][1]) - datetime.timestamp(video_frames[0][1])
    frame_rate = len(video_frames) / total_time
    print(f"total time duration: {total_time}, frame_rate: {frame_rate}")
    skip_ratio = frame_rate // DESIRED_FR

    for i, (frame, timestamp) in enumerate(video_frames):
        if i % skip_ratio != 0:
            continue
        if i == 0:
            video_starting_time = datetime.timestamp(timestamp)
        curr_timestamp = datetime.timestamp(timestamp) - video_starting_time
        curr_frames.append(frame)
        curr_timestamps.append(curr_timestamp)
        if len(curr_frames) == DESIRED_FR:
            curr_split = i // DESIRED_FR
            print(f"Split {curr_split} saved!")
            split_path = os.path.join(DUMP_FRAMES_PATH, f"frames_split_{curr_split}.npy")
            np.save(split_path, curr_frames)
            splits.append(split_path)
            timestamps.append(curr_timestamps)
            curr_frames = []
            curr_timestamps = []
    
    return {"splits_paths": splits,
            "splits_timestamps": timestamps,
            "fps": DESIRED_FR}
if __name__ == "__main__":
    # run()
    extract_ppg_from_video()

