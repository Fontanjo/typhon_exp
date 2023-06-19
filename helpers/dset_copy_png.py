import numpy as np
from PIL import Image
from pathlib import Path
import re
from matplotlib import pyplot as plt
import os


arguments = {
    'dataset_folder': "datasets_segmentation/BUS",
    'output_folder': "DATASET_BUS_png",
    }


"""
Description: this script takes as input a datasetpath,
and copies it converting npy images to png, maintaining the same structure
Params:
    dataset_folder   - Required  : folder containing the dataset to convert
    output_folder    - Required  : output folder
Returns:
    - No return value
"""

# Get the arguments
dataset_folder = arguments['dataset_folder']
output_folder = arguments['output_folder']

# The slash operator '/' in the pathlib module is similar to os.path.join()
output_path = output_folder
os.makedirs(output_path, exist_ok=True)

for subdir, dirs, files in os.walk(dataset_folder):
    current_dir = subdir.replace(dataset_folder, output_folder)
    os.makedirs(current_dir, exist_ok=True)
    for file in files:
        new_filename = os.path.join(current_dir, file)
        array = np.load(os.path.join(subdir, file)) * 255
        # array.shape = (1, *array.shape)
        image = Image.fromarray(array.astype('uint8'))
        image.save(new_filename, 'JPEG')

#
#
# # glob() matches files with the given pattern
# # It returns a generator containing the matched paths
# # image_paths_list = category_path.glob("*(*).png")
#
# # Need to cast to list because generators do not have len()
# # ntotal = ntotal + len(list(Path.iterdir(category_path)))
#
# for image_path in Path.iterdir(dataset_folder):
#     image_name = image_path.name
#     mask_path = groundtruth_folder / image_name
#     # match = re.search(r"^([a-z]+) \(([0-9]+)\).png$", image_name)
#     # if match:
#     #     # group(1,2) returns a tuple containing both groups submatches for the regex
#     #     category, number = match.group(1,2)
#
#     # Open, convert and save the image
#     # file_name = f"{category}{number}"
#     file_name = image_path.stem
#     destination_path = output_path / Path(file_name).with_suffix('.npy')
#     image = Image.open(image_path)
#     # L for Luminance, 8-bits pixels, 1-channel
#     image = image.convert('L')
#     image = np.asarray(image)
#     image = image / 255
# #    print(image)
# #    print(type(image))
# #    plt.imshow(image)
# #    plt.show()
#     np.save(destination_path, image)
#
#     # Can have more then 1 mask per image
#     # mask_paths = category_path.glob(f"{category} ({number})_mask*.png")
#     # for index, mask_path in enumerate(mask_paths):
#     # Open, convert and save the mask
#     mask_name = file_name + f"_mask"
#     mask_destination_path = output_path / Path(mask_name).with_suffix('.npy')
#     mask = Image.open(mask_path)
#     # L for Luminance, 8-bits pixels, 1-channel
#     mask = mask.convert('L')
#     mask = np.asarray(mask)
#     mask = mask / 255
#     np.save(mask_destination_path, mask)
#
#     print('.', end='', flush=True)
#
#
# # Sanity checks
# # print(f"\nTotal numbers of slices (ntotal): {ntotal}")
# # nimg = len(list(Path.iterdir(images_output_path)))
# # nmasks = len(list(Path.iterdir(masks_output_path)))
# # print(f"Expected: nimg + nmasks == ntotal")
# # print(f"{nimg} + {nmasks} = {nimg + nmasks} (expected {ntotal})")
# # assert (nimg + nmasks) == ntotal, "File numbers do not match!"
