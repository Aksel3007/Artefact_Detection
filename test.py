import numpy as np
import mne
import scipy.io as sio
import hdf5storage
import os


path_raw_data = '//uni.au.dk/dfs/Tech_EarEEG/Students/RD2022_Artefact_AkselStark/data/1A/study_1A_mat_simple/S_01/night_1/EEG_raw_250hz.npy'


datatest = np.load(path_raw_data)

print('debug')