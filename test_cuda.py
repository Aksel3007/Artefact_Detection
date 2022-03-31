

import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from dataset import EEGDataset
from torch.utils.data import random_split
import neptune.new as neptune
from torchinfo import summary


# use pytorch to check if cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)