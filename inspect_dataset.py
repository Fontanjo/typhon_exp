import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# input_folder = "./datasets_segmentation/BUSI"
# output_path = 'datasets_info/BUSI_small'

# input_folder = "/media/jonas/Seagate Expansion Drive/Memoria/master_thesis/datasets_segmentation/Brain"

input_folder = "/home/jonas/UnifrServer/data/TyphonData/Ready/BUSI"
output_path = 'datasets_info/BUSI'


def main():

    os.makedirs(output_path, exist_ok=True)

    categories = ['train', 'val', 'test']
    # categories = ['test']
    dct = {x: {} for x in categories}


    for cat in categories:
        for img in tqdm(os.listdir(input_folder + '/' + cat)):
            # Skip masks
            if img.endswith('mask.npy'): continue
            # Load image
            ary = np.load(input_folder + '/' + cat + '/' + img)
            dct[cat][ary.shape] = dct[cat].get(ary.shape, 0) + 1



    # print(sorted(dct['train'].items(), key=lambda x:x[1]))

    for cat in categories:
        # Get max and min elements
        max_x = max(dct[cat].items(), key=lambda x: x[0][0])
        min_x = min(dct[cat].items(), key=lambda x: x[0][0])
        max_y = max(dct[cat].items(), key=lambda x: x[0][1])
        min_y = min(dct[cat].items(), key=lambda x: x[0][1])

        # Open or create file (in write mode, thus overwrite. Change to 'a' to append)
        with open(output_path + '/' + cat, 'w') as file:
            # file.seek(0)
            # Iterate over sorted dict
            file.writelines([f'{el[0]} -> {el[1]}\n' for el in sorted(dct[cat].items(), key=lambda x:x[1], reverse=True)])

            file.write(f'\n Max x: {max_x[0][0]} (in {max_x})')
            file.write(f'\n Min x: {min_x[0][0]} (in {min_x})')
            file.write(f'\n Max y: {max_y[0][1]} (in {max_y})')
            file.write(f'\n Min y: {min_y[0][1]} (in {min_y})')


        num_bins = len(dct[cat])


        x = [t[0][0] for t in dct[cat].items()]
        y = [t[0][1] for t in dct[cat].items()]


        for axis, axis_name in zip([x, y], ['x', 'y']):
            plt.figure()
            n, bins, patches = plt.hist(axis, num_bins,
                                density = 1,
                                color ='green',
                                alpha = 0.7)

            plt.xlabel('Value')
            plt.ylabel('Count')

            plt.title(f'Shape distribution ({axis_name})',
                      fontweight ="bold")

            plt.savefig(f'{output_path}/{cat}_{axis_name}')







if __name__ == "__main__":
    main()
