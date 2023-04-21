import numpy as np
import os
from tqdm import tqdm
import shutil


input_path = "/home/fontanjo/data/TyphonData/Ready/BraTS2019_LGG"
output_path = input_path
# output_path = "/media/jonas/Seagate Expansion Drive/Memoria/master_thesis/Dataset_HNPC_output/masks_small"


# Create out folder
os.makedirs(output_path, exist_ok=True)

for arr_name in tqdm(os.listdir(input_path)):
    # In this case only masks should be compressed. Comment to work on all files
    if arr_name.endswith('mask.npy'):
        # Simply copy
        # shutil.copy(input_path + '/' + arr_name, output_path)
        # Load
        arr = np.load(input_path + "/" + arr_name)
        np.save(output_path + '/' + arr_name, arr.astype('uint8'))
