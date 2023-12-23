import numpy as np
import matplotlib.pyplot as plt

from utils import log_info

def run_delta_plot_distribution():
    """ distribution of delta between predicted noise and ground truth noise """
    bin_cnt = 200
    eps = 1e-3

    def read_floats_from_file(f):
        x_arr = []
        with open(f, 'r') as fptr:
            lines = fptr.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line == '': continue
                f_tmp = float(line)
                x_arr.append(f_tmp)
            # for
        # with
        log_info(f"Read {len(x_arr)} floats from file {f}")
        return x_arr

    def set_plt_ui():
        plt.tick_params('both', labelsize=25)
        # plt.title(r"Distribution of prediction error", fontsize=25)
        plt.xlabel(f"$\\Delta_{{{c},{h},{w}}}$", fontsize=38)
        plt.ylabel("Frequency", fontsize=38)
        plt.legend(fontsize=25, loc='upper left')

    # ts_all = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
    ts_list_list = [
        [199, 399, 599],
        [699, 799, 899],
        # [799, 899, 999],
    ]
    c, h, w = 1, 128, 128
    root_dir = "./RectifiedFlow_Pytorch/configs/charts/prediction_error_distribution"
    for ts_list in ts_list_list:
        fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(1, 1, 1)
        for ts in ts_list:
            x = read_floats_from_file(f"{root_dir}/ts{ts:03d}_dim{c}_{h:03d}_{w:03d}.txt")
            std = np.std(x)
            bins = np.linspace(-std * 3, std * 3, num=bin_cnt + 1, endpoint=True)
            ts_continue = float(ts) / 1000. * (1. - eps) + eps
            plt.hist(x, bins=bins, histtype='step', label=f"t={ts_continue:.4f}")
        # for
        set_plt_ui()
        f_path = f"{root_dir}/fig_error_dist_dim{c}_{h:03d}_{w:03d}_ts{ts_list[0]:03d}.png"
        fig.savefig(f_path, bbox_inches='tight')
        print(f"saved file: {f_path}")
        plt.close()
    # for

def fid_train_from_scratch_celeba():
    fid_s1  = [228.4, 209.6, 195.5, 187.7, 179.9, 173.7, 171.1, 165.1, 165.4, 160.8, 158.6]
    fid_s2  = [111.9, 102.1, 94.71, 91.67, 88.30, 87.55, 86.71, 85.43, 86.61, 85.56, 85.24]
    fid_s3  = [77.84, 68.59, 62.84, 58.78, 56.39, 57.04, 56.95, 55.90, 57.18, 56.83, 57.68]
    fid_s4  = [55.02, 47.01, 43.36, 40.06, 38.79, 38.92, 39.04, 39.06, 40.35, 40.92, 42.19]
    fid_s5  = [40.46, 34.25, 31.63, 29.40, 28.79, 29.04, 29.61, 29.86, 31.20, 32.17, 33.54]
    fid_s6  = [31.18, 26.43, 24.93, 23.29, 23.09, 23.41, 24.16, 25.02, 26.31, 27.58, 28.90]
    fid_s8  = [21.43, 18.79, 18.06, 17.07, 17.31, 17.81, 18.93, 20.01, 21.24, 22.72, 24.47]
    fid_s10 = [16.84, 14.74, 14.49, 14.09, 14.67, 15.12, 16.39, 17.51, 18.81, 20.69, 22.29]
    lambda_arr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.tick_params('both', labelsize=20)
    ax2.tick_params('both', labelsize=20)
    ax1.plot(lambda_arr, fid_s2, linestyle='-', color='r', marker='o')
    ax1.plot(lambda_arr, fid_s3, linestyle='-', color='g', marker='s')
    ax1.plot(lambda_arr, fid_s4, linestyle='-', color='b', marker='d')
    ax1.legend(['2 steps', '3 steps', '4 steps'], fontsize=20, loc='upper center')

    ax2.plot(lambda_arr, fid_s6, linestyle='-', color='r', marker='o')
    ax2.plot(lambda_arr, fid_s8, linestyle='-', color='g', marker='s')
    ax2.plot(lambda_arr, fid_s10, linestyle='-', color='b', marker='d')
    ax2.legend(['6 steps', '8 steps', '10 steps'], fontsize=20, loc='upper center')

    fig.supylabel('    FID', fontsize=25, rotation=0)  # make it horizontal
    fig.supxlabel('value of $\\lambda$', fontsize=25)
    fig.suptitle("FID Comparison on CelebA by Different Sampling Steps", fontsize=30)
    # plt.show()
    f_path = './RectifiedFlow_Pytorch/configs/charts/fig_fid_train_from_scratch_celeba.png'
    fig.savefig(f_path, bbox_inches='tight')
    print(f"file saved: {f_path}")
    plt.close()

def main():
    """ entry point """
    # run_delta_plot_distribution()
    fid_train_from_scratch_celeba()

if __name__ == '__main__':
    main()
