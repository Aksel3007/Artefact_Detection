import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import math


class EEGDataset(Dataset):
    """EEG dataset"""

    def __init__(self, root_dir, numNights, sectionLength, transform=None):
        """
        Args:
            root_dir (string): Directory with EEG data
            numNights: Number of nights to include in the dataset
        """

        #Load in the number of nights
        nightsLoaded = 0
        labelsLoaded = 0
        
        #Run through all the cleaned EEG files
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                
                if "EEG_raw_250hz" in file: #First, load in the downsampled EEG data
                    print(os.path.join(subdir, file))

                    if nightsLoaded == 0: # If first night, save array in data, otherwise, append it to data
                        data = np.load(os.path.join(subdir, file))
                    else:
                        data = np.hstack((data,np.load(os.path.join(subdir, file))))

                    print(f'Night {nightsLoaded} data loaded')
                    nightsLoaded += 1

                if 'artefact_annotation' in file:    #Load in the annotation
                    print(os.path.join(subdir, file))
                    if labelsLoaded == 0:
                        labels = np.load(os.path.join(subdir, file))
                    else:
                        np.hstack((labels,np.load(os.path.join(subdir, file))))

                    print(f'Lables for night {labelsLoaded} loaded')
                    labelsLoaded += 1
                    

            if nightsLoaded == numNights and labelsLoaded == numNights : # When the correct number of nights has been loaded
                break

        
        self.data = data
        self.labels = labels
        self.sectionLength = sectionLength


    def __len__(self):
        samples = 0
        
        samples = self.data.shape[0]*self.data.shape[1]

        return int(samples/self.sectionLength)-5 # Drop the last samples avoid incomplete series. TODO: Zeropadding or some better solution

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        index = idx * self.sectionLength
        channel = math.floor(index/self.data.shape[1])
        start = index % self.data.shape[1]
        data_seg = self.data[channel, start : start + self.sectionLength]
        data_seg = np.fft.fft(data_seg) #TODO: abs Todo: FFTW er hurtigere
        artefacts = self.labels[channel, start : start + self.sectionLength]
        
        # TODO: Window: 3 sekunder spektrogram og originale samples, træn som image classification

        # TODO: Alternativt: 1dconv kernel 100 -> 2dconv
        


        containsArtefact = 0
        for i in artefacts:
            if i:
                containsArtefact = 1
                break
        

        return data_seg, float(containsArtefact)



# Test if the dataset works

# raw_data_dir = '//uni.au.dk/dfs/Tech_EarEEG/Students/RD2022_Artefact_AkselStark/data/1A/study_1A_mat_simple'

# ds1 = EEGDataset(raw_data_dir,2, 250)

# print('debug')