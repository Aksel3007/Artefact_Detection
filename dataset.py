import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np


class EEGDataset(Dataset):
    """EEG dataset"""

    def __init__(self, root_dir, numNights, transform=None):
        """
        Args:
            root_dir (string): Directory with EEG data
            numNights: Number of nights to include in the dataset
        """

        #Load in the number of nights
        clean_data_dir = '//uni.au.dk/dfs/Tech_EarEEG/Students/RD2022_Artefact_AkselStark/data/1A/study_1A_mat_simple_cleaned'
        raw_data_dir = '//uni.au.dk/dfs/Tech_EarEEG/Students/RD2022_Artefact_AkselStark/data/1A/study_1A_mat_simple'

        nightsLoaded = 0
        data = []
        labels = []
        
        #Run through all the cleaned EEG files
        for subdir, dirs, files in os.walk(raw_data_dir):
            for file in files:
                
                if "250hz" in file: #First, load in the downsampled EEG data
                    print(os.path.join(subdir, file))
                    data.append(np.load(os.path.join(subdir, file)))
                    print(f'Night {nightsLoaded} data loaded')

                if 'artefact_annotation' in file:    #Load in the annotation
                    print(os.path.join(subdir, file))
                    labels.append(np.load(os.path.join(subdir, file)))
                    
                    nightsLoaded += 1
                    print(f'Night {nightsLoaded} data loaded')

            if nightsLoaded == numNights: # When the correct number of nights have
                break

        


                



        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample