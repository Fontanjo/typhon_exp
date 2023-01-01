import numpy as np
import re
import matplotlib.pyplot as plt
import mmap
import scipy



# FILE_PATH = "./results_atari/20221231_AE6_bootstrap_PDaSi0_1/run_logs/221231_0037.log"
FILE_PATH = "./results_atari/20221231_AE6_bootstrap_all0_2/run_logs/221231_0037.log"

FILE_PATH = "./results_atari/20221231_AE6_s2_0_3/run_logs/221231_0038.log" # AE6_s2

NB_ENVS = 10 # To rescale the sum

def main():
    vals = []
    ys = []
    # envs = ['DemonAttack-v5', 'FishingDerby-v5', 'Frostbite-v5', 'Kangaroo-v5', 'NameThisGame-v5', 'Phoenix-v5', 'Qbert-v5', 'Seaquest-v5', 'SpaceInvaders-v5', 'TimePilot-v5', 'testwrong']
    envs = []
    for env_name in envs:
        if env_name.endswith('-v5'): env_name = env_name.replace('-v5', '')
        with open(FILE_PATH, 'r+') as f:
            data = mmap.mmap(f.fileno(), 0)
            RE = re.compile(rf'>>> ep (\d*): loss score for {env_name}-v5: (\d*\.\d*)')
            res = RE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string

        if len(res) > 0:
            vals.append([env_name, [float(x) for _, x in res]])
            # ys.append([float(y) for y, _ in res])


    with open(FILE_PATH, 'r+') as f:
        data = mmap.mmap(f.fileno(), 0)
        RE = re.compile(rf'>>> ep (\d*) sum: (\d*\.\d*)')
        res = RE.findall(data.read().decode('utf-8')) # Data is bytes-like object, decode into string

    vals.append(['avg', [float(x) / NB_ENVS for _, x in res]])
    # ys.append([float(y) for y, _ in res])


    # avg = [sum(x) / len(vals) for x in zip(*[v for n, v in vals])]


    for n, v in vals:
        # plt.plot(ys, v, label=n)
        plt.hist(v, label=n, density=True, bins=50)

    # Get mean and std of average (/sum)
    mu = np.mean(vals[-1][1])
    std = np.std(vals[-1][1])

    # Plot normal distribution
    # x_min, x_max = 0.0, mu * 2
    x_min, x_max, y_min, y_max = plt.axis()

    x = np.linspace(x_min, x_max, 1000)
    y = scipy.stats.norm.pdf(x, mu, std)

    plt.plot(x, y, color='red', label='Normal distribution')
    plt.fill_between(x, y, color='red', alpha=0.35)

    plt.text(x_min, y_max * 0.9, f'Mu: {np.round(mu, 5)}')
    plt.text(x_min, y_max * 0.85, f'Std: {np.round(std, 5)}')

    plt.title(f'Distribution of bootstrap values ({NB_ENVS} envs)')
    plt.xlabel('AVERAGE value (bootstrap uses sum)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
