import os
import torch
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import numpy as np
import math
import random
import tracemalloc
import torch.utils.data as data


class EEGDataset(Dataset):
    """EEG dataset"""

    def __init__(self, root_dir, transform=None, skips = 0):
        """
        Args:
            root_dir (string): Directory with EEG data
            skips (int): Number of nights to skip (so if you want the first night, skips = 0)
        """
        tracemalloc.start() # Enable memory profiling
        
        good_skips = 0
        bad_skips = 0
        self.segmentLength = 0
        
        for subdir, dirs, files in sorted(os.walk(root_dir)):
            for file in files:
                
                if "good" in file and good_skips == skips:
                    good_data = np.load(os.path.join(subdir, file))
                    print(os.path.join(subdir, file))
                    good_skips += 1
                elif "good" in file:
                    good_skips += 1
                    
                    
                if "bad" in file and bad_skips == skips:
                    bad_data = np.load(os.path.join(subdir, file))
                    print(os.path.join(subdir, file))
                    bad_skips += 1
                elif "bad" in file:
                    bad_skips += 1
        
        self.good_data = good_data
        self.bad_data = bad_data            
        print(f'Memory usage: {tracemalloc.get_traced_memory()[0]/1000000} MB\n')
    
    
    def __len__(self):
        return 2 * min(self.good_data.shape[0], self.bad_data.shape[0])
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if idx % 2: 
            return self.good_data[idx//2], float(0)
        else:
            return self.bad_data[idx//2], float(1)
        

# Test the dataset

if True:
    ds1 = EEGDataset('../data', skips = 0)
    print(len(ds1))
    print(ds1[0])
    print("\n")