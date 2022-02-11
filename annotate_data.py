import scipy.io as sio
import hdf5storage
import numpy as np

path_raw_data = '//uni.au.dk/dfs/Tech_EarEEG/Students/RD2022_Artefact_AkselStark/data/1A/study_1A_mat_simple/S_01/night_1/EEG_raw.mat'
path_cleaned_data = '//uni.au.dk/dfs/Tech_EarEEG/Students/RD2022_Artefact_AkselStark/data/1A/study_1A_mat_simple_cleaned/S_01/night_1/EEG_raw.mat'

print('loading data')
data_raw = hdf5storage.loadmat(path_raw_data)['data']
print("raw data loaded")

data_cleaned = hdf5storage.loadmat(path_cleaned_data)['data']
print("clean data loaded")

def annotate_single_night_artefacts(clean_data): #Function takes a cleaned matrix and create a matrix of where artefacts occur (finds nan)
    artefact_matrix = np.zeros(clean_data.shape)

    for index_0 in range(clean_data.shape[0]):
        print(f"Annotating channel: {index_0}")
        for index_1 in range(clean_data.shape[1]):
            if np.isnan(clean_data[index_0,index_1]):
                artefact_matrix[index_0] = 1
                
    return artefact_matrix

print('Annotate data')
annotation = annotate_single_night_artefacts(data_raw,data_cleaned)


print("ok?")
print("Debug")



