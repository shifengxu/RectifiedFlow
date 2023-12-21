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
        plt.xlabel(f"$\\Delta_g$", fontsize=38)
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


def main():
    """ entry point """
    run_delta_plot_distribution()

if __name__ == '__main__':
    main()
