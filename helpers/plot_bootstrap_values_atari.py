import numpy as np
import re
import matplotlib.pyplot as plt
import mmap



# FILE_PATH = "./results_atari/20221229_VAE5_bootstrap_QFDa_2/run_logs/221229_1550.log"
# FILE_PATH = "./results_atari/20221229_VAE5_bootstrap_QFDa_2/run_logs/221229_1649.log"

# FILE_PATH = "./results_atari/20221229_VAE5_bootstrap_QFDa_2/run_logs/221229_1848.log" # 15k normal
FILE_PATH = "./results_atari/20221229_VAE5_bootstrap_QFDa_2/run_logs/221229_2230.log" # 50k normal
# FILE_PATH = "./results_atari/20221229_VAE5_nomp_bootstrap_QFDa_1/run_logs/221229_2217.log" # 50k nomp

def main():
    vals = []
    ys = []
    for env_name in ['Qbert', 'Frostbite', 'DemonAttack']:
        with open(FILE_PATH, 'r+') as f:
            data = mmap.mmap(f.fileno(), 0)
            RE = re.compile(rf'>>> ep (\d*): New loss score for {env_name}-v5: (\d*\.\d*)')
            res = RE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string

        vals.append([env_name, [float(x) for _, x in res]])
        ys.append([float(y) for y, _ in res])



    avg = [sum(x) / 3 for x in zip(*[v for n, v in vals])]

    # Since all the y are the same, just consider the first list
    ys = ys[0]

    for n, v in vals:
        plt.plot(ys, v, label=n)



    plt.plot(ys, avg, label='AVG', linewidth=5)
    plt.legend()
    plt.show()


    # # Need to reopen the file every time or findall fails..
    # with open(FILE_PATH, 'r+') as f:
    #     data = mmap.mmap(f.fileno(), 0)
    #     QbertRE = re.compile(r'>>> New loss score for Qbert-v5: (\d*\.\d*)')
    #     Qbert_res = QbertRE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string
    #
    # with open(FILE_PATH, 'r+') as f:
    #     data = mmap.mmap(f.fileno(), 0)
    #     FrostbiteRE = re.compile(r'>>> New loss score for Frostbite-v5: (\d*\.\d*)')
    #     Frostbite_res = FrostbiteRE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string
    #     print(Frostbite_res)
    #
    # with open(FILE_PATH, 'r+') as f:
    #     data = mmap.mmap(f.fileno(), 0)
    #     DemonAttackRE = re.compile(r'>>> New loss score for DemonAttack-v5: (\d*\.\d*)')
    #     DemonAttack_res = DemonAttackRE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string
    #
    #
    # Qbert_f = [float(x) for x in Qbert_res]
    # Frostbite_f = [float(x) for x in Frostbite_res]
    # DemonAttack_f = [float(x) for x in DemonAttack_res]
    #
    # avg = [sum(x) / 3 for x in zip(Qbert_f, Frostbite_f, DemonAttack_f)]
    #
    #
    # # plt.hist(t1_f, bins=len(t1_f))
    # # plt.hist(t1ce_f, bins=len(t1ce_f))
    # # plt.hist(t2_f, bins=len(t2_f))
    # # plt.hist(flair_f, bins=len(flair_f))
    # # plt.figure()
    #
    #
    # plt.plot(range(len(Qbert_f)), Qbert_f)
    # plt.plot(range(len(Frostbite_f)), Frostbite_f)
    # plt.plot(range(len(DemonAttack_f)), DemonAttack_f)
    # plt.plot(range(len(avg)), avg, linewidth=5)
    #
    # plt.show()


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
