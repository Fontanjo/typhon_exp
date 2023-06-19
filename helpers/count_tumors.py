import numpy as np
import os
import sys

def main(root_folder):
    empty_mask = 0
    tumor_mask = 0
    others = 0
    o = None

    # Recursively explore all subfolders
    for (root, dirs, files) in os.walk(root_folder, topdown=True):
        # Act on each npy file
        for file in [f for f in files if f.endswith("mask.npy")]:
            # Load file
            ary = np.load(root + '/' + file)
            # Check if there is a 1
            if np.max(ary) == 1:
                tumor_mask += 1
            elif np.max(ary) == 0:
                empty_mask += 1
            else:
                others += 1
                o = np.max(ary)
    print(f"In folder {root_folder}:")
    print(f"Masks with tumor: {tumor_mask}")
    print(f"Masks without tumors: {empty_mask}")
    print(f"Others (pray it's 0): {others} {o}")




if __name__ == "__main__":
    main(sys.argv[1])
