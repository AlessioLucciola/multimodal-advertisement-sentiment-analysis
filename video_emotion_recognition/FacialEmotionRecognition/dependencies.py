'''
file: dependencies.py
author: @vincit0re
brief: This file contains the dependencies and import libraries for the project.
date: 20230-05-05
'''

'''All the required libraries for the project'''
# import tqdm

import numpy as np
import pandas as pd
from torchvision.utils import make_grid, save_image
import torchvision.models as models
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchsummary import summary
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import copy
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")


sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
pd.set_option('display.max_columns', 20)

use_gpu = True
device = torch.device("cuda" if (
    use_gpu and torch.cuda.is_available()) else "cpu")
print(f"Using Device: {device}")
