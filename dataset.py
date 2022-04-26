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

    def __init__(self, root_dir, numNights, sectionLength, transform=None, skips = 0, filtered = False):
        """
        Args:
            root_dir (string): Directory with EEG data
            numNights: Number of nights to include in the dataset
            sectionLength: Amount of samples in a segment
        """

        tracemalloc.start() # Enable memory profiling
        
        #Load in the number of nights
        nightsLoaded = 0
        nightsSkipped = 0
        labelsLoaded = 0
        labelsSkipped = 0
        currentNight = 0
        shape_list = []
        
        if filtered:
            filename = "EEG_raw_250hz.npy"
        else:
            filename = "EEG_raw_250hz_unfiltered.npy"
                
        
        #Run through all the cleaned EEG files
        for subdir, dirs, files in sorted(os.walk(root_dir)):
            #print(os.path.join(subdir))
            for file in files:
                                
                if filename in file: #First, load in the downsampled EEG data
                    #print(os.path.join(subdir, file))
                    if nightsSkipped == skips:                        
                        data = (np.load(os.path.join(subdir, file), mmap_mode='c'))
                        
                        print(os.path.join(subdir, file))
                        
                        #Print memory usage
                        print(f'Memory usage: {tracemalloc.get_traced_memory()[0]/1000000} MB\n')
                        
                        nightsLoaded += 1
                    else:
                        nightsSkipped +=1

                elif 'artefact_annotation' in file:    #Load in the annotation
                    #print(os.path.join(subdir, file))
                    if labelsSkipped == skips:
                        #shape = tuple(np.load(os.path.join(subdir, "data_shape.npy")))
                        if labelsLoaded == 0:
                            labels = np.load(os.path.join(subdir, file), mmap_mode='c')
                        else:
                            labels = np.hstack((labels,np.load(os.path.join(subdir, file), mmap_mode='c')))

                        print(os.path.join(subdir, file))
                        labelsLoaded += 1
                    else: 
                        labelsSkipped +=1
                
            
            if nightsLoaded == numNights and labelsLoaded == numNights : # When the correct number of nights has been loaded
                break

        self.artefactIndecies = np.where(labels==1) #Needed to be able to grab an artefact sample
        self.goodIndecies = np.where(labels==0) #Needed to be able to grab a good sample
        self.data = data
        self.labels = labels
        self.sectionLength = sectionLength
        
        


    def __len__(self):
        samples = 0
        
        samples = self.data.shape[0]*self.data.shape[1]

        segments = int(samples/self.sectionLength)
        
        return segments - (segments % 64)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        p = random.random()

        if idx % 2: # Every other sample is a segment from the dataset
            index = idx * self.sectionLength
            channel = math.floor(index/self.data.shape[1])
            start = index % self.data.shape[1]
            while True:
                data_seg = self.data[channel, start : start + self.sectionLength]
                if data_seg.shape[0] == self.sectionLength:
                    break
                else:
                    print("Error: Segment length is not equal to section length")
            
            #std = max([np.std(data_seg),0.00001]) # Standard deviation for normalization and feature
            #mean = np.mean(data_seg) # Mean for normalization and feature

            #data_seg = (data_seg - mean) #Normalize to unit variance and 0 mean
            
            # Fourier transform
            data_seg = np.fft.fft(data_seg) #TODO: abs Todo: FFTW er hurtigere
            
            # Normalize and append mean and stdandard deviation as features
            #data_seg = np.append(data_seg, [mean/150000, std/3000])
            
            #Convert to float32 
            data_seg = np.float32(np.absolute(data_seg))

            artefacts = self.labels[channel, start : start + self.sectionLength]
            
            #data_seg[0] = 0 # Remove DC component
            #data_seg[self.sectionLength-1] = 0 # Remove DC component
            
            containsArtefact = 0
            for i in artefacts:
                if i:
                    containsArtefact = 1
                    break
            return data_seg, float(containsArtefact)#, channel, start 

        else: # every other sample is a randomly selected artefact segment
            randIndex = int(random.random()*self.artefactIndecies[0].size)
            #randIndex = idx # Temporary fix to make dataset deterministic
            channelIndex = self.artefactIndecies[0][randIndex]
            timeIndex = self.artefactIndecies[1][randIndex]
            start = timeIndex - int(self.sectionLength/2)

            # Avoid reading out of range
            if start < 0: start = 0 
            if start > (self.labels.shape[1]-self.sectionLength): start = self.labels.shape[1] - self.sectionLength
            

            while True:
                data_seg = self.data[channelIndex, start : start + self.sectionLength] 
                if data_seg.shape[0] == self.sectionLength:
                    break
                else:
                    print("Error: Segment length is not equal to section length")






            
            #std = max([np.std(data_seg),0.00001]) # Standard deviation for normalization and feature
            #mean = np.mean(data_seg) # Mean for normalization and feature

            #data_seg = (data_seg - mean) #Normalize to unit variance and 0 mean
            
            # Fourier transform
            data_seg = np.fft.fft(data_seg) #TODO: abs Todo: FFTW er hurtigere
            
            # Normalize and append mean and stdandard deviation as features
            #data_seg = np.append(data_seg, [mean/150000, std/3000])
            
            #Convert to float32 
            data_seg = np.float32(np.absolute(data_seg))
            
            #data_seg[0] = 0 # Remove DC component
            #data_seg[self.sectionLength-1] = 0 # Remove DC component
            
            return data_seg, float(1)#, channelIndex, start




# Function to load a dataset concatenating N nights
def load_dataset(nights, sectionLenght, root_dir, filtered = False):
    datasets = []
    for i in nights:
        datasets.append(EEGDataset(root_dir,1, sectionLenght ,skips = i, filtered = filtered))
    
    # Return concatenated datasets
    return data.ConcatDataset(datasets)


            
        
        # TODO: Window: 3 sekunder spektrogram og originale samples, træn som image classification

        # TODO: Alternativt: 1dconv kernel 100 -> 2dconv
        

print("Classification dataset version: apr-25-22-v3")
# # Test if the dataset works, by plotting a random sample
if False:
    import matplotlib.pyplot as plt
    
    raw_data_dir = '../data'

    ds1 = EEGDataset(raw_data_dir,30, 750,skips = 0)

    for i in range(10000):
        a = ds1[random.randint(0,ds1.__len__())]
        #print(a)
        #plt.plot(range(250),a[0])
        #plt.title("Data segment") 
        #plt.show()
    print("Dataset loaded")
    
# # Test if filtering works, by plotting filtered and unfiltered data
if False:
    import matplotlib.pyplot as plt
    
    raw_data_dir = '../data'

    filtered_data = EEGDataset(raw_data_dir,1, 250,skips = 0)
    unfiltered_data = EEGDataset(raw_data_dir,1, 250,skips = 0,filtered = False)

    for i in range(10000):
        index = random.randint(0,filtered_data.__len__())
        filt = filtered_data[index]
        unfilt = unfiltered_data[index]
        plt.plot(range(252),filt[0])
        plt.title("Data segment") 
        plt.show()

# Test concatenation of datasets
if False:
    raw_data_dir = '../data'

    ds1 = load_dataset(range(6), 750, raw_data_dir)
    
    train_dataloader = DataLoader(ds1, batch_size=64, drop_last = True) # Drop_last, to avoid incomplete batches, which won't work with weighted loss

    # Test the dataloader
    for i, data in enumerate(train_dataloader):
        if data[0].shape[0] != 64:
            print(f"index: {i}")
            print("Error: Batch size is not 64")

        if data[0].shape[1] != 750:
            print(f"index: {i}")
            print("Error: Section length is not 750")

        if i % 1000 == 0:
            print(i)
            




