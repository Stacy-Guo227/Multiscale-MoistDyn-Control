import numpy as np
from ML_model_data import *

import tensorflow as tf
from tensorflow.keras import layers, Input, backend, Model, losses, optimizers, models

# Load vae components
encoder, decoder = Load_ML()
# Load normalized datasets
era5_normIVT_ssn_test = Load_Dataset()['x_test']
era5_normIVT_ssn_val   = Load_Dataset()['x_val']
era5_normIVT_ssn_train = Load_Dataset()['x_train']
temp = np.concatenate((era5_normIVT_ssn_train, era5_normIVT_ssn_val), axis=0)
era5_normIVT_ssn_whole = np.concatenate((temp, era5_normIVT_ssn_test), axis=0)
# Get latent vectors
era5_lvIVT_ssn_test  = encoder.predict(era5_normIVT_ssn_test)[2]
era5_lvIVT_ssn_val   = encoder.predict(era5_normIVT_ssn_val)[2]
era5_lvIVT_ssn_train = encoder.predict(era5_normIVT_ssn_train)[2]
era5_lvIVT_ssn_whole = encoder.predict(era5_normIVT_ssn_whole)[2]
lv_ssn_dict = {'ERA5(test)':era5_lvIVT_ssn_test, 
               'ERA5(val)':era5_lvIVT_ssn_val, 
               'ERA5(train)':era5_lvIVT_ssn_train,  
               'ERA5(whole)':era5_lvIVT_ssn_whole}
# np.save(f'../../data/vae_model/model_dataset/fix_lv_temp.npy', lv_ssn_dict)

"""
!! Notice !!
Due to the inherent random settings of the model optimizer, the latent vectors will vary very slightly every time the vae-model is called.
To avoid the slight variation of latent vectors for subsequent analyses, it is recommended to simply save the latent vectors after one model-call, such as shown in this code file.
The analyses and results of this study are based on such maneuver, which created the file '/data/processed/fix_latent_vectors/V2_sum_lv_era5_taiesm.npy'.
"""
