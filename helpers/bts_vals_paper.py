import numpy as np
import re
import matplotlib.pyplot as plt
import mmap



FILE_PATH = "./results/20230511_bootstrap_4/run_logs/230511_2119.log"

def main():

    # Need to reopen the file every time or findall fails..
    with open(FILE_PATH, 'r+') as f:
        data = mmap.mmap(f.fileno(), 0)
        t1RE = re.compile(r'>>> New dice score for BUS_SELECTED: (\d\.\d*)')
        t1_res = t1RE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string

    with open(FILE_PATH, 'r+') as f:
        data = mmap.mmap(f.fileno(), 0)
        t1ceRE = re.compile(r'>>> New dice score for LINK_BRATS_LGG_flair: (\d\.\d*)')
        t1ce_res = t1ceRE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string

    with open(FILE_PATH, 'r+') as f:
        data = mmap.mmap(f.fileno(), 0)
        t2RE = re.compile(r'>>> New dice score for BUSI: (\d\.\d*)')
        t2_res = t2RE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string

    with open(FILE_PATH, 'r+') as f:
        data = mmap.mmap(f.fileno(), 0)
        flairRE = re.compile(r'>>> New dice score for BRAIN: (\d\.\d*)')
        flair_res = flairRE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string
        print(flair_res)



    t1_f = [float(x) for x in t1_res]
    t1ce_f = [float(x) for x in t1ce_res]
    t2_f = [float(x) for x in t2_res]
    flair_f = [float(x) for x in flair_res]


    # Some are in scientific notation
    t1_f = [t for t in t1_f if t < 1]

    plt.hist(t1_f, bins=len(t1_f))
    # plt.hist(t1ce_f, bins=len(t1ce_f))
    # plt.hist(t2_f, bins=len(t2_f))
    # plt.hist(flair_f, bins=len(flair_f))
    plt.figure()


    plt.plot(range(len(t1_f)), t1_f)
    # plt.plot(range(len(t1ce_f)), t1ce_f)
    # plt.plot(range(len(t2_f)), t2_f)
    # plt.plot(range(len(flair_f)), flair_f)
    plt.show()


        #
        # if s:
        #     print(s)
        # else:
        #     print('nothing found')
        # mo = re.search('error: (.*)', data)
        # if mo:
        #     print("found error", mo.group(1))

    # # To match lines as 'From the 8460 new images, 809 added to the list in 104.26s',
    # correctLineRE = re.compile(r'From the \d* new images, \d* added to the list in \d*.\d*s')
    #
    # # Save values
    # added = []
    # time = []
    # with open(FILE_PATH, 'r') as file:
    #     for line in file:
    #         # Match lines
    #         s = correctLineRE.search(line)
    #         # Extract values
    #         if s:
    #             new_img, added_img, added_time = re.search(r'(\b\d+).+(\b\d+).+(\b\d+\.\d+)', s.string).groups()
    #             added.append(int(added_img))
    #             time.append(float(added_time))
    #
    # # Create cumulative sum
    # added_sum = np.cumsum(added)
    # time_sum = np.cumsum(time) / 3600
    #
    # # Create moving average
    # def moving_average(x, w):
    #     return np.convolve(x, np.ones(w) / w, 'same')
    # MA = 10
    # added_ma = moving_average(added, MA)
    #
    #
    # # Plot
    # plt.figure(figsize=(14, 7))
    #
    # plt.subplot(1, 2, 1)
    # plt.scatter(time_sum, added_sum, color='Green')
    # plt.xlabel('Cumulative time (h)')
    # plt.ylabel('Cumulative images')
    # plt.title('Total added images')
    #
    # # plt.figure()
    # plt.subplot(1, 2, 2)
    # plt.scatter(time_sum, added)
    # plt.xlabel('Cumulative time (h)')
    # plt.ylabel('Newly added images')
    # plt.plot(time_sum, added_ma,
    #         label=f'Moving average ({MA})',
    #         color='red')
    #
    # plt.axhline(10,
    #         label='Stopping condition',
    #         color='darkred')
    # plt.title('Newly added images')
    #
    # plt.legend()
    # plt.savefig(FILE_PATH[:-3])
    # plt.show()




if __name__ == "__main__":
    main()
