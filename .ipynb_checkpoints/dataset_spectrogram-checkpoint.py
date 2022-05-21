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

    def __init__(self, root_dir, transform=None, skips = 0, normalized = True):
        """
        Args:
            root_dir (string): Directory with EEG data
            skips (int): Number of nights to skip (so if you want the first night, skips = 0)
        """
        tracemalloc.start() # Enable memory profiling
        
        good_skips = 0
        bad_skips = 0
        self.segmentLength = 0
        
        if normalized:
            good_filename = "good_segments.npy"
            bad_filename = "bad_segments.npy"
        else:
            good_filename = "good_segments_unnormalized.npy"
            bad_filename = "bad_segments_unnormalized.npy"
        
        for subdir, dirs, files in sorted(os.walk(root_dir)):
            for file in files:
                
                
                if good_filename in file and good_skips == skips:
                    good_data = np.load(os.path.join(subdir, file), mmap_mode='c')
                    print(os.path.join(subdir, file))
                    good_skips += 1
                elif good_filename in file:
                    good_skips += 1
                    
                    
                if bad_filename in file and bad_skips == skips:
                    bad_data = np.load(os.path.join(subdir, file), mmap_mode='c')
                    print(os.path.join(subdir, file))
                    bad_skips += 1
                elif bad_filename in file:
                    bad_skips += 1
        
        self.good_data = good_data
        self.bad_data = bad_data            
        print(f'Memory usage: {tracemalloc.get_traced_memory()[0]/1000000} MB\n')
        
        print("Lengths:\n")
        print(f"Good data length: {self.good_data.shape[0]}")
        print(f"Bad data length: {self.bad_data.shape[0]}")
        print(f"Caluculated length: {self.__len__()}")
    
    
    def __len__(self):
        
        return (2 * min(self.good_data.shape[0], self.bad_data.shape[0])) - 1
     
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if idx % 2: #If the index is odd, return the good data
            log_data = np.nan_to_num(self.good_data[idx//2])
            log_data[log_data < 0.00001] = 0.00001
            #return np.log(log_data), float(0)
            return log_data, float(0)
        
        else: # If the index is even, return the bad data
            log_data = np.nan_to_num(self.bad_data[idx//2])
            log_data[log_data < 0.00001] = 0.00001
            #return np.log(log_data), float(1)
            return log_data, float(1)
        
def load_dataset(nights,root_dir, normalized = True):
    datasets = []
    for i in nights:
        try: # Allowed to fail if reading non excisting files
            datasets.append(EEGDataset(root_dir,skips = i, normalized = normalized))
        except:
            print(f"Data for nigth {i} does not exist")
    
    # Return concatenated datasets
    return data.ConcatDataset(datasets)


print("Spectrogram dataset version 19 log")


# Test the dataset

if False:
    ds1 = EEGDataset('../data', skips = 10)
    print(len(ds1))
    print(ds1[0])
    print("\n")
    
if False:
    ds1 = load_dataset([11],'../data')
    print("\n")
    