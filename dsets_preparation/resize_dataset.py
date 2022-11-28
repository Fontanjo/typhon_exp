from npy_to_image import npy2png
from PIL import Image
import numpy as np
import os
from tqdm import tqdm


NEW_X, NEW_Y = 512, 512

INPUT_PATH = './convert_test/BUSI_ORIG'
# OUTPUT_PATH = './convert_test/BUSI_ORIG'
OUTPUT_PATH = './convert_test/BUSI_reshaped'

TEMP_PATH = './convert_test/temp.png'
def main():
    # Create ouput dir
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Recursively explore all subfolders
    for (root, dirs, files) in os.walk(INPUT_PATH, topdown=True):
        # Create subfolders
        for dir in dirs:
            os.makedirs(root.replace(INPUT_PATH, OUTPUT_PATH) + '/' + dir, exist_ok=True)

        # Convert and save each .npy file
        for file in tqdm([x for x in files if x.endswith('.npy')]):
            file_path = root + '/' + file
            output_path = root.replace(INPUT_PATH, OUTPUT_PATH) + '/' + file

            # Convert to png and save to temp location
            npy2png(file_path, TEMP_PATH)

            # Reshape
            os.system(f'convert {TEMP_PATH} -resize {NEW_X}x{NEW_Y}! {TEMP_PATH}')

            # Load reshaped png
            image = Image.open(TEMP_PATH)
            # L for Luminance, 8-bits pixels, 1-channel
            image = image.convert('L')
            np.save(output_path, image)

    # Remove temp image
    # os.system('rm {TEMP_PATH}')


if __name__ == '__main__':
    main()
