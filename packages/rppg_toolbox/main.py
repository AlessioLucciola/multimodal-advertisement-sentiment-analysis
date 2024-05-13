""" The main function of rPPG deep learning pipeline."""
import argparse
import random

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

def extract_ppg_from_video(vid_path: str | None = None) -> torch.Tensor:
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
    video_data = read_video(vid_path)
    bvps = torch.tensor([])
    for split_path, split_timestamps in zip(video_data["splits_paths"], video_data["splits_timestamps"]):
        print(f"Extracting ppg from split at: {split_path}.")
        raw_frames = np.load(split_path)
        assert len(split_timestamps) == raw_frames.shape[0], f"ERROR: timestamps and frames must be of the same length | timestamp of length {len(split_timestamps)}, split of length {raw_frames.shape[0]}"
        frames = preprocess.preprocess_frames(raw_frames, config.TEST.DATA.PREPROCESS)
        # print(f"preprocessed frames shape: {frames.shape}")
        frames = preprocess.parse_frames(frames, data_format="NDCHW")
        # print(f"parsed frames shape: {frames.shape}")
        output = model_trainer.test_from_frames(frames).detach()
        # plot_signal(output.reshape(-1).numpy(), "model output")

        for item in output:
            npy_bvp = get_bvp(item.squeeze(), diff_flag=True, bandpass=True, fs=round(fps))
            bvp = torch.tensor(npy_bvp.copy()).to(torch.float32)
            bvps = torch.cat((bvps, bvp.view(1, -1)), dim=0)
        print(f"rppg bvp is {bvps} with shape {bvps.shape}")

    shutil.rmtree(DUMP_FRAMES_PATH)
    print(f"Removed temp_frames directory")
    return bvps

if __name__ == "__main__":
    # run()
    extract_ppg_from_video()

