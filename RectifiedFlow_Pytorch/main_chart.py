import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision as tv
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
        std = None
        for ts in ts_list:
            x = read_floats_from_file(f"{root_dir}/ts{ts:03d}_dim{c}_{h:03d}_{w:03d}.txt")
            if std is None:
                std = np.std(x)
            # std = 0.7
            bins = np.linspace(-std * 3.5, std * 3.5, num=bin_cnt + 1, endpoint=True)
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
    lambda_arr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.tick_params('both', labelsize=22)
    ax2.tick_params('both', labelsize=22)
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
    f_path = './RectifiedFlow_Pytorch/charts/fig_fid_train_from_scratch_celeba.png'
    fig.savefig(f_path, bbox_inches='tight')
    print(f"file saved: {f_path}")
    plt.close()

def merge_image_col_row():
    root_dir = './output1_church_sampling'
    lambda_arr = ['0.0', '0.1', '0.2', '0.3', '0.4']
    step_arr = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    # step_arr = ['06', '07', '08', '09', '10']
    save_dir = os.path.join(root_dir, f"merged_all_lambda_step01-10")
    if not os.path.exists(save_dir):
        print(f"mkdirs: {save_dir}")
        os.makedirs(save_dir)
    first_dir = os.path.join(root_dir, f"gen_lambda{lambda_arr[0]}_step{step_arr[0]}")
    fname_list = os.listdir(first_dir)
    fname_list.sort()
    # fname_list = ['00084.png']
    for fname in fname_list:
        img_arr = []
        for step in step_arr:
            for lda in lambda_arr:
                img_path = os.path.join(root_dir, f"gen_lambda{lda}_step{step}", fname)
                img = tv.io.read_image(img_path)
                img = img.float()
                img /= 255.0
                img_arr.append(img)
                # tv.utils.save_image(img, os.path.join(save_dir, f"lambda{lda}_step{step}.png"))
            # for
        # for
        save_img_path = os.path.join(save_dir, fname)
        tv.utils.save_image(img_arr, save_img_path, nrow=len(lambda_arr))
        print(f"save image: {save_img_path}")
    # for fname
    with open(os.path.join(save_dir, f"ReadMe.txt"), 'w') as fptr:
        line = ", ".join(step_arr)
        fptr.write(f"row step: {line}\n")
        line = ", ".join(lambda_arr)
        fptr.write(f"col lambda: {line}\n")
    # with

def gen_img_of_step_count():
    root_dir = "./RectifiedFlow_Pytorch/charts/"
    step_arr = [1, 2, 3, 4, 5]
    for step in step_arr:
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.get_xaxis().set_visible(False)       # hide ticks
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)     # hide lines
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        text_kwargs = dict(ha='center', va='center', fontsize=40, color='k')
        plt.text(0.5, 0.5, f"$k={step}$", text_kwargs)

        f_path = os.path.join(root_dir, f"fig_step_count_{step}.png")
        fig.savefig(f_path, bbox_inches='tight')
        print(f"saved: {f_path}")
        plt.close()
    # for

def gen_img_of_vertical_text():
    root_dir = "./RectifiedFlow_Pytorch/charts/"
    step_arr = [1, 2]
    for s in step_arr:
        fig = plt.figure(figsize=(1.1, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.get_xaxis().set_visible(False)       # hide ticks
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)     # hide lines
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        text_kwargs = dict(ha='center', va='center', fontsize=40, color='k', rotation=90)
        plt.text(0.5, 0.5, f"{s}-Rectified Flow", text_kwargs)

        f_path = os.path.join(root_dir, f"fig_lbl_{s}_rectified_flow.png")
        fig.savefig(f_path, bbox_inches='tight')
        print(f"saved: {f_path}")
        plt.close()
    # for

def change_background_local():
    # step_arr = ['01', '02', '03']
    # lambda_arr = ["0.5"]
    scale = 0.90  # for lambda 0.1, 0.3, scale = 0.8
    old_dir = "D:/Coding2/RectifiedFlow/checkpoint/image-compare/bedroom_1RF_2RF"
    new_dir = "D:/Coding2/RectifiedFlow/checkpoint/image-compare/bedroom_1RF_2RF"
    file_id_arr = ["00030", "00059", "00084"]
    for fi in file_id_arr:
        old_image_path = f"{old_dir}/{fi}_1RF_lambdaFalse_step3_old.png"
        new_image_path = f"{new_dir}/{fi}_1RF_lambdaFalse_step3.png"
        img = plt.imread(old_image_path)  # dimension: [64, 64, 3]
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        img[1, :] *= scale  # green
        # img[0, :] /= scale  # red
        # img[2, :] /= scale  # blue
        # img[:] *= 1.6
        torch.clamp(img, 0., 1., out=img)
        tv.utils.save_image(img, new_image_path)
        print(new_image_path)
    # for

def reverse_time_ode_gradient():
    def dm_grad_fn(x_0, epsilon, t):
        grad = -(9.95 * t + 0.05) * math.exp(-4.975 * t * t - 0.05 * t) * x_0
        exp2 = math.exp(-9.95 * t * t - 0.1 * t)
        grad += (9.95 * t + 0.05) * exp2 / math.sqrt(1 - exp2) * epsilon
        return grad

    ts_arr = list(range(1, 1000, 1)) # todo: change step back.
    ts_arr = [step / 1000 for step in ts_arr] # 0.01 ~ 0.999
    ep_arr = [0.6, 0.0, -0.6]   # epsilon array
    x0_arr = [0.7, 0.1, -0.5]   # x_0 array
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', labelsize=24)
    line_arr = []
    # (0, (5, 10)): loosely dash
    # (5, (10, 3)): long dash with offset
    for ep, l_stype in zip(ep_arr, ['-', (0, (5, 10)), ':']):
        for x0, l_color in zip(x0_arr, ['r', 'g', 'b']):
            grad_arr = []
            for ts in ts_arr:
                gr = dm_grad_fn(x0, ep, ts)
                grad_arr.append(gr)
            # for
            lbl = f"$\\epsilon$:{ep}, $y_0$:{x0}"
            line, = ax.plot(ts_arr, grad_arr, label=lbl, linestyle=l_stype, color=l_color)
            line_arr.append(line)
        # for
    # for
    plt.gca().invert_xaxis() # reverse time ODE. So reverse X axis.
    legend1 = ax.legend(handles=line_arr[:3], fontsize=28, loc='upper left') # multiple legend
    plt.gca().add_artist(legend1)
    legend2 = ax.legend(handles=line_arr[3:6], fontsize=28, loc=(0.277, 0.655))
    plt.gca().add_artist(legend2)
    ax.legend(handles=line_arr[6:], fontsize=28, loc='lower left')

    fig.supylabel(r'    $\frac{dy_t}{dt}$', fontsize=40, rotation=0)  # make it horizontal
    fig.supxlabel(r'reverse-time timestep $t$', fontsize=30)
    fig.suptitle("Diffusion model reverse-time ODE on specific points", fontsize=30)
    # plt.show()
    f_path = './charts/fig_reverse_time_ode_grad_dm.png'
    fig.savefig(f_path, bbox_inches='tight')
    print(f"file saved: {f_path}")
    plt.close()

def main():
    """ entry point """
    # run_delta_plot_distribution()
    # fid_train_from_scratch_celeba()
    # merge_image_col_row()
    # gen_img_of_step_count()
    # change_background_local()
    # gen_img_of_vertical_text()
    reverse_time_ode_gradient()

if __name__ == '__main__':
    main()
