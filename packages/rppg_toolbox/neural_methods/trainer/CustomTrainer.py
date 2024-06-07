
"""Trainer for DeepPhys."""
import os
import torch
import numpy as np
from typing import Optional, Dict, List
from packages.rppg_toolbox.neural_methods.model.DeepPhys import DeepPhys
from packages.rppg_toolbox.neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class CustomTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.model_dir = config.MODEL.MODEL_DIR
        self.chunk_len = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        
        if config.TOOLBOX_MODE != "only_test":
            raise ValueError("Custom trainer only supports 'only_test' as a TOOLBOX_MODE")
        self.model = DeepPhys(img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

    def train(self, data_loader):
        raise NotImplementedError("Custom trainer doesn't impelement a training loop")

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        raise NotImplementedError("Custom trainer doesn't impelement a validation loop")
    
    def test_from_frames(self, frames: torch.Tensor | np.ndarray, frame_rate: int) -> torch.Tensor:
        """
        Performs a test loop given an array of frames that model a video as input
        """
        self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=torch.device(self.config.DEVICE)))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        frames = frames.to(self.device)
        
        predictions = []
        for chunk in frames:
            chunk = chunk.unsqueeze(0)
            print(f"chunk shape: {chunk.shape}")
            predictions.extend(self.test_step(chunk, frame_rate))
        predictions = torch.cat(predictions, dim=-1).T
        # shape is: [num_chunks, 100]
        # print(f"predictions with shape: {predictions.shape}")
        return predictions
    
    def test_step(self, frames: torch.Tensor, frame_rate: int) -> List[int]:
        predictions = []
        N, D, C, H, W = frames.shape
        frames = frames.view(N * D, C, H, W)
        pred_ppg_test = self.model(frames)
        
        for idx in range(N):
            predictions.append(pred_ppg_test[idx * frame_rate:(idx + 1) * frame_rate])
        return predictions

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("===Testing===")
        self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=torch.device(self.config.DEVICE)))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            predictions = []
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                data_test = test_batch[0].to(self.config.DEVICE) 
                print(f"data test shape is: {data_test.shape}")
                predictions.extend(self.test_step(frames=data_test))

        predictions = torch.cat(predictions, dim=-1)
        print(f"predictions are: {predictions} with shape: {predictions.shape}")

    def save_model(self, index):
        """Inits parameters from args and the writer for TensorboardX."""
        raise NotImplementedError("CustomTrainer doesn't allow model saving")
 
