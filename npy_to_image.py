import numpy as np
from PIL import Image



def npy2png(input_path, output_path):
    if not output_path.endswith('.png'):
        output_path = output_path + '.png'
    # Load data
    data = Image.fromarray(np.load(input_path))
    # Save image
    data.save(output_path)




INPUT_PATH = './convert_test/benign1.npy'
OUTPUT_PATH = './convert_test/benign1.png'

if __name__ == "__main__":
    npy2png(INPUT_PATH, OUTPUT_PATH)
