import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision as tv
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
    fid_s7  = [25.48, 21.90, 20.98, 19.56, 19.63, 20.03, 20.91, 22.04, 23.17, 24.60, 26.07]
    fid_s8  = [21.43, 18.79, 18.06, 17.07, 17.31, 17.81, 18.93, 20.01, 21.24, 22.72, 24.47]
    fid_s9  = [18.81, 16.25, 16.16, 15.23, 15.85, 16.36, 17.56, 18.52, 19.99, 21.77, 23.22]
    fid_s10 = [16.84, 14.74, 14.49, 14.09, 14.67, 15.12, 16.39, 17.51, 18.81, 20.69, 22.29]
    lambda_arr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    ax1.tick_params('both', labelsize=22)
    ax2.tick_params('both', labelsize=22)
    # ax3.tick_params('both', labelsize=22)
    ax1.plot(lambda_arr, fid_s2, linestyle='-', color='r', marker='o')
    ax1.plot(lambda_arr, fid_s3, linestyle='-', color='g', marker='s')
    ax1.plot(lambda_arr, fid_s4, linestyle='-', color='b', marker='d')
    ax1.legend(['2 steps', '3 steps', '4 steps'], fontsize=20, loc='upper center')

    ax2.plot(lambda_arr, fid_s6, linestyle='-', color='r', marker='o')
    ax2.plot(lambda_arr, fid_s8, linestyle='-', color='g', marker='s')
    ax2.plot(lambda_arr, fid_s10, linestyle='-', color='b', marker='d')
    ax2.legend(['6 steps', '8 steps', '10 steps'], fontsize=20, loc='upper center')

    # ax3.plot(lambda_arr, fid_s8, linestyle='-', color='r', marker='o')
    # ax3.plot(lambda_arr, fid_s9, linestyle='-', color='g', marker='s')
    # ax3.plot(lambda_arr, fid_s10, linestyle='-', color='b', marker='d')
    # ax3.legend(['8 steps', '9 steps', '10 steps'], fontsize=20, loc='upper center')

    fig.supylabel('FID  ', fontsize=25, rotation=0)  # make it horizontal
    fig.supxlabel('$\\lambda$', fontsize=25)
    fig.suptitle("FID Comparison", fontsize=30)
    # plt.show()
    f_path = './charts/fig_fid_train_from_scratch_celeba.png'
    fig.savefig(f_path, bbox_inches='tight')
    print(f"file saved: {f_path}")
    plt.close()

def track_fid_when_training():
    epoch_arr = [0, # 1, 2, 5,10,
                 20, 30, 40,
                 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    fid_l000_s1 = [447.22, # 253.62, 288.49, 321.12, 345.06,
                   364.52, 368.16, 371.87,
                   370.48, 364.62, 363.35, 364.24, 357.74, 358.20, 357.27, 356.17, 357.15, 356.13]
    fid_l0_2_s1 = [447.22, # 258.55, 273.72, 281.29, 283.54,
                   288.04, 288.05, 290.43,
                   290.53, 289.90, 286.79, 285.97, 284.57, 283.33, 280.58, 283.82, 282.96, 283.48]
    fid_l000_s2 = [447.22, # 167.68, 176.44, 203.65, 222.47,
                   203.78, 187.23, 184.14,
                   171.30, 168.40, 164.47, 165.67, 162.90, 162.08, 161.18, 160.00, 159.67, 160.87]
    fid_l0_2_s2 = [447.22, # 146.25, 120.48, 133.02, 132.34,
                   133.22, 132.94, 129.47,
                   128.47, 126.86, 128.10, 128.24, 126.11, 124.85, 124.43, 123.54, 122.78, 123.79]
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.tick_params('both', labelsize=22)
    ax2.tick_params('both', labelsize=22)
    ax1.plot(epoch_arr, fid_l000_s1, linestyle='-', color='g', marker='o')
    ax1.plot(epoch_arr, fid_l0_2_s1, linestyle='-', color='r', marker='s')
    ax1.legend(['vanilla rectified flow', 'consistent-gradient method'], fontsize=20, loc='upper right')
    ax1.set_title("1-Step Sampling", fontsize=25)

    ax2.plot(epoch_arr, fid_l000_s2, linestyle='-', color='g', marker='o')
    ax2.plot(epoch_arr, fid_l0_2_s2, linestyle='-', color='r', marker='s')
    ax2.legend(['vanilla rectified flow', 'consistent-gradient method'], fontsize=20, loc='upper right')
    ax2.set_title("2-Step Sampling", fontsize=25)

    fig.subplots_adjust(hspace=0.4)
    fig.supylabel('FID  ', fontsize=25, rotation=0)  # make it horizontal
    fig.supxlabel('Epoch', fontsize=25)
    # fig.suptitle("FID Metrics When Training", fontsize=30)

    f_path = './charts/fig_track_fid_when_training.png'
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
    root_dir = "./charts/"
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
        plt.text(0.5, 0.5, f"{s}-rectified flow", text_kwargs)

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

def gradient_variance():
    """
    step_cnt: 20
    :return:
    """
    def read_floats_from_file(f):
        x_arr = []
        with open(f, 'r') as fptr:
            lines = fptr.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('#') or line == '': continue
                x = line.split()[0]
                x = float(x)
                x_arr.append(x)
            # for
        # with
        return x_arr

    root_dir = "./charts/gradient_variance"
    fn1 = 'var_ckpt_gnobitab_RF_CIFAR10_ReRF3_img5000_step20.txt'
    fn2 = 'var_ckpt_gnobitab_RF_CIFAR10_ReRF2-RefineL0.1_img5000_step20.txt'
    var_refl3_origin = read_floats_from_file(os.path.join(root_dir, fn1))
    var_refl2_refine = read_floats_from_file(os.path.join(root_dir, fn2))
    print(f"var_refl3_origin: {len(var_refl3_origin)}")
    print(f"var_refl2_refine: {len(var_refl2_refine)}")
    avg_refl3_origin, avg_refl2_refine = [], []
    var_cnt, var_sum = 0, 0.
    for var in var_refl3_origin:
        var_cnt += 1
        var_sum += var
        avg = var_sum / var_cnt * 10000
        avg_refl3_origin.append(avg)
    var_cnt, var_sum = 0, 0.
    for var in var_refl2_refine:
        var_cnt += 1
        var_sum += var
        avg = var_sum / var_cnt * 10000
        avg_refl2_refine.append(avg)
    x_axis = list(range(1, 1+len(var_refl2_refine)))
    ignore_first_part = 100
    x_axis = x_axis[ignore_first_part:]
    avg_refl3_origin = avg_refl3_origin[ignore_first_part:]
    avg_refl2_refine = avg_refl2_refine[ignore_first_part:]

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.tick_params('both', labelsize=22)
    ax1.plot(x_axis, avg_refl3_origin, linestyle='-', color='r')
    ax1.plot(x_axis, avg_refl2_refine, linestyle='-', color='g')
    ax1.legend(['3-rectified flow', '3-rectified flow + consistent-gradient method'], fontsize=20, loc='upper right')

    fig.supylabel('Average Variance ($\\times 10^{-4}$)', fontsize=25)  # make it horizontal: rotation=0
    fig.supxlabel('Number of Samples', fontsize=25)
    fig.suptitle("Average Variance of Predicted Gradients", fontsize=30)
    # plt.show()
    f_path = './charts/gradient_variance/fig_variance_of_gradients_ReRF3_vs_ReRF2Refine.png'
    fig.savefig(f_path, bbox_inches='tight')
    print(f"file saved: {f_path}")
    plt.close()

def trajectory_diffusion_vs_rectified_flow():
    y1k_df = [0.952998, 0.952969, 0.952942, 0.952916, 0.952891, 0.952865, 0.952839, 0.952813, 0.952787, 0.952760,
              0.952733, 0.952707, 0.952680, 0.952653, 0.952626, 0.952598, 0.952569, 0.952541, 0.952513, 0.952484,
              0.952455, 0.952425, 0.952394, 0.952363, 0.952332, 0.952301, 0.952271, 0.952241, 0.952210, 0.952177,
              0.952145, 0.952114, 0.952081, 0.952048, 0.952016, 0.951982, 0.951948, 0.951913, 0.951876, 0.951840,
              0.951803, 0.951767, 0.951730, 0.951692, 0.951654, 0.951616, 0.951577, 0.951538, 0.951499, 0.951460,
              0.951420, 0.951380, 0.951339, 0.951297, 0.951255, 0.951213, 0.951168, 0.951123, 0.951078, 0.951032,
              0.950987, 0.950941, 0.950893, 0.950844, 0.950795, 0.950745, 0.950695, 0.950644, 0.950594, 0.950542,
              0.950490, 0.950437, 0.950383, 0.950328, 0.950272, 0.950216, 0.950158, 0.950099, 0.950039, 0.949979,
              0.949917, 0.949854, 0.949792, 0.949728, 0.949664, 0.949598, 0.949532, 0.949464, 0.949395, 0.949325,
              0.949254, 0.949182, 0.949109, 0.949035, 0.948959, 0.948883, 0.948806, 0.948727, 0.948647, 0.948565,
              0.948483, 0.948400, 0.948316, 0.948232, 0.948146, 0.948058, 0.947969, 0.947879, 0.947788, 0.947694,
              0.947601, 0.947506, 0.947410, 0.947311, 0.947212, 0.947112, 0.947011, 0.946909, 0.946805, 0.946700,
              0.946593, 0.946485, 0.946375, 0.946263, 0.946150, 0.946036, 0.945920, 0.945802, 0.945683, 0.945562,
              0.945440, 0.945316, 0.945190, 0.945063, 0.944935, 0.944806, 0.944675, 0.944542, 0.944408, 0.944272,
              0.944135, 0.943995, 0.943853, 0.943709, 0.943564, 0.943417, 0.943270, 0.943120, 0.942968, 0.942813,
              0.942658, 0.942500, 0.942341, 0.942179, 0.942015, 0.941849, 0.941681, 0.941510, 0.941336, 0.941159,
              0.940981, 0.940800, 0.940618, 0.940433, 0.940247, 0.940058, 0.939866, 0.939671, 0.939474, 0.939275,
              0.939072, 0.938867, 0.938660, 0.938450, 0.938238, 0.938024, 0.937806, 0.937586, 0.937363, 0.937137,
              0.936909, 0.936678, 0.936444, 0.936207, 0.935966, 0.935723, 0.935477, 0.935228, 0.934976, 0.934720,
              0.934461, 0.934199, 0.933936, 0.933668, 0.933397, 0.933123, 0.932845, 0.932565, 0.932281, 0.931994,
              0.931702, 0.931408, 0.931110, 0.930809, 0.930505, 0.930198, 0.929886, 0.929570, 0.929251, 0.928927,
              0.928600, 0.928269, 0.927934, 0.927596, 0.927254, 0.926907, 0.926556, 0.926202, 0.925844, 0.925483,
              0.925117, 0.924746, 0.924371, 0.923992, 0.923609, 0.923223, 0.922832, 0.922436, 0.922036, 0.921632,
              0.921224, 0.920812, 0.920393, 0.919969, 0.919541, 0.919110, 0.918676, 0.918237, 0.917792, 0.917342,
              0.916888, 0.916428, 0.915965, 0.915496, 0.915022, 0.914544, 0.914062, 0.913575, 0.913083, 0.912587,
              0.912085, 0.911577, 0.911066, 0.910549, 0.910028, 0.909502, 0.908971, 0.908435, 0.907894, 0.907347,
              0.906795, 0.906238, 0.905675, 0.905108, 0.904536, 0.903958, 0.903375, 0.902787, 0.902195, 0.901596,
              0.900993, 0.900385, 0.899773, 0.899156, 0.898534, 0.897906, 0.897274, 0.896637, 0.895995, 0.895348,
              0.894696, 0.894040, 0.893378, 0.892712, 0.892040, 0.891364, 0.890682, 0.889996, 0.889306, 0.888612,
              0.887913, 0.887209, 0.886500, 0.885786, 0.885067, 0.884344, 0.883617, 0.882884, 0.882147, 0.881405,
              0.880659, 0.879907, 0.879151, 0.878389, 0.877622, 0.876850, 0.876073, 0.875292, 0.874506, 0.873714,
              0.872918, 0.872117, 0.871310, 0.870499, 0.869683, 0.868862, 0.868036, 0.867205, 0.866367, 0.865525,
              0.864677, 0.863825, 0.862968, 0.862106, 0.861237, 0.860364, 0.859484, 0.858599, 0.857710, 0.856815,
              0.855915, 0.855008, 0.854096, 0.853178, 0.852255, 0.851326, 0.850392, 0.849452, 0.848508, 0.847558,
              0.846602, 0.845641, 0.844672, 0.843698, 0.842719, 0.841734, 0.840744, 0.839748, 0.838747, 0.837740,
              0.836729, 0.835713, 0.834690, 0.833662, 0.832628, 0.831588, 0.830543, 0.829492, 0.828436, 0.827373,
              0.826306, 0.825233, 0.824154, 0.823070, 0.821980, 0.820885, 0.819784, 0.818677, 0.817566, 0.816448,
              0.815325, 0.814198, 0.813065, 0.811927, 0.810784, 0.809635, 0.808482, 0.807323, 0.806158, 0.804989,
              0.803814, 0.802633, 0.801448, 0.800258, 0.799062, 0.797861, 0.796656, 0.795446, 0.794231, 0.793011,
              0.791786, 0.790557, 0.789323, 0.788083, 0.786839, 0.785590, 0.784337, 0.783080, 0.781819, 0.780554,
              0.779283, 0.778008, 0.776729, 0.775446, 0.774159, 0.772869, 0.771573, 0.770273, 0.768969, 0.767661,
              0.766349, 0.765033, 0.763713, 0.762389, 0.761061, 0.759729, 0.758394, 0.757055, 0.755713, 0.754368,
              0.753017, 0.751664, 0.750307, 0.748947, 0.747584, 0.746217, 0.744847, 0.743474, 0.742098, 0.740720,
              0.739339, 0.737954, 0.736567, 0.735176, 0.733783, 0.732387, 0.730989, 0.729588, 0.728184, 0.726778,
              0.725369, 0.723957, 0.722544, 0.721128, 0.719710, 0.718291, 0.716870, 0.715446, 0.714020, 0.712593,
              0.711164, 0.709732, 0.708299, 0.706864, 0.705427, 0.703988, 0.702548, 0.701107, 0.699665, 0.698221,
              0.696775, 0.695327, 0.693879, 0.692429, 0.690977, 0.689525, 0.688071, 0.686617, 0.685161, 0.683705,
              0.682248, 0.680789, 0.679330, 0.677869, 0.676408, 0.674947, 0.673485, 0.672023, 0.670560, 0.669097,
              0.667634, 0.666171, 0.664707, 0.663243, 0.661778, 0.660313, 0.658849, 0.657384, 0.655919, 0.654455,
              0.652991, 0.651527, 0.650064, 0.648601, 0.647138, 0.645675, 0.644213, 0.642752, 0.641291, 0.639831,
              0.638371, 0.636911, 0.635452, 0.633992, 0.632534, 0.631076, 0.629619, 0.628163, 0.626708, 0.625254,
              0.623801, 0.622350, 0.620899, 0.619450, 0.618001, 0.616554, 0.615108, 0.613662, 0.612218, 0.610776,
              0.609334, 0.607894, 0.606455, 0.605018, 0.603582, 0.602148, 0.600715, 0.599284, 0.597855, 0.596428,
              0.595002, 0.593578, 0.592156, 0.590737, 0.589319, 0.587903, 0.586490, 0.585079, 0.583669, 0.582261,
              0.580856, 0.579452, 0.578050, 0.576650, 0.575252, 0.573856, 0.572462, 0.571071, 0.569682, 0.568296,
              0.566911, 0.565530, 0.564151, 0.562774, 0.561400, 0.560029, 0.558660, 0.557295, 0.555932, 0.554571,
              0.553213, 0.551857, 0.550504, 0.549154, 0.547806, 0.546462, 0.545120, 0.543782, 0.542446, 0.541114,
              0.539785, 0.538458, 0.537135, 0.535816, 0.534499, 0.533186, 0.531875, 0.530567, 0.529261, 0.527959,
              0.526661, 0.525366, 0.524074, 0.522785, 0.521500, 0.520217, 0.518938, 0.517663, 0.516392, 0.515123,
              0.513858, 0.512596, 0.511338, 0.510083, 0.508832, 0.507585, 0.506341, 0.505100, 0.503865, 0.502632,
              0.501403, 0.500178, 0.498956, 0.497738, 0.496525, 0.495315, 0.494108, 0.492905, 0.491706, 0.490511,
              0.489320, 0.488132, 0.486947, 0.485768, 0.484592, 0.483419, 0.482250, 0.481085, 0.479925, 0.478769,
              0.477616, 0.476467, 0.475322, 0.474182, 0.473046, 0.471913, 0.470785, 0.469662, 0.468543, 0.467428,
              0.466316, 0.465210, 0.464108, 0.463010, 0.461915, 0.460826, 0.459740, 0.458659, 0.457581, 0.456509,
              0.455440, 0.454375, 0.453315, 0.452260, 0.451208, 0.450160, 0.449117, 0.448079, 0.447044, 0.446013,
              0.444987, 0.443966, 0.442948, 0.441934, 0.440926, 0.439921, 0.438920, 0.437926, 0.436935, 0.435947,
              0.434965, 0.433987, 0.433013, 0.432043, 0.431078, 0.430117, 0.429160, 0.428209, 0.427261, 0.426317,
              0.425380, 0.424446, 0.423515, 0.422591, 0.421670, 0.420752, 0.419840, 0.418932, 0.418027, 0.417128,
              0.416233, 0.415341, 0.414455, 0.413573, 0.412694, 0.411821, 0.410953, 0.410087, 0.409228, 0.408372,
              0.407520, 0.406674, 0.405831, 0.404993, 0.404159, 0.403328, 0.402503, 0.401682, 0.400863, 0.400051,
              0.399242, 0.398437, 0.397637, 0.396840, 0.396049, 0.395262, 0.394478, 0.393700, 0.392925, 0.392155,
              0.391389, 0.390626, 0.389870, 0.389116, 0.388368, 0.387623, 0.386882, 0.386147, 0.385414, 0.384686,
              0.383962, 0.383242, 0.382527, 0.381815, 0.381109, 0.380405, 0.379706, 0.379012, 0.378320, 0.377635,
              0.376951, 0.376273, 0.375598, 0.374927, 0.374261, 0.373598, 0.372941, 0.372285, 0.371636, 0.370989,
              0.370347, 0.369709, 0.369074, 0.368444, 0.367817, 0.367195, 0.366576, 0.365962, 0.365350, 0.364745,
              0.364141, 0.363543, 0.362947, 0.362356, 0.361768, 0.361184, 0.360604, 0.360027, 0.359455, 0.358886,
              0.358321, 0.357759, 0.357202, 0.356647, 0.356097, 0.355550, 0.355007, 0.354468, 0.353932, 0.353400,
              0.352872, 0.352346, 0.351825, 0.351307, 0.350793, 0.350282, 0.349775, 0.349271, 0.348770, 0.348274,
              0.347779, 0.347290, 0.346802, 0.346320, 0.345838, 0.345363, 0.344889, 0.344421, 0.343953, 0.343491,
              0.343031, 0.342574, 0.342121, 0.341670, 0.341224, 0.340779, 0.340340, 0.339901, 0.339468, 0.339036,
              0.338608, 0.338184, 0.337761, 0.337344, 0.336926, 0.336516, 0.336105, 0.335699, 0.335296, 0.334894,
              0.334498, 0.334101, 0.333711, 0.333322, 0.332936, 0.332554, 0.332171, 0.331796, 0.331421, 0.331050,
              0.330682, 0.330313, 0.329952, 0.329591, 0.329232, 0.328878, 0.328524, 0.328176, 0.327828, 0.327483,
              0.327143, 0.326802, 0.326466, 0.326132, 0.325798, 0.325471, 0.325144, 0.324819, 0.324499, 0.324178,
              0.323861, 0.323548, 0.323234, 0.322925, 0.322619, 0.322311, 0.322011, 0.321711, 0.321411, 0.321118,
              0.320824, 0.320532, 0.320245, 0.319959, 0.319673, 0.319393, 0.319113, 0.318834, 0.318561, 0.318288,
              0.318015, 0.317749, 0.317483, 0.317216, 0.316957, 0.316697, 0.316437, 0.316184, 0.315931, 0.315678,
              0.315429, 0.315183, 0.314937, 0.314693, 0.314454, 0.314214, 0.313975, 0.313742, 0.313510, 0.313277,
              0.313049, 0.312823, 0.312598, 0.312371, 0.312153, 0.311935, 0.311716, 0.311500, 0.311288, 0.311077,
              0.310864, 0.310658, 0.310454, 0.310249, 0.310044, 0.309846, 0.309648, 0.309450, 0.309252, 0.309062,
              0.308872, 0.308682, 0.308490, 0.308308, 0.308125, 0.307942, 0.307758, 0.307581, 0.307406, 0.307230,
              0.307053, 0.306881, 0.306713, 0.306545, 0.306376, 0.306207, 0.306047, 0.305886, 0.305725, 0.305563,
              0.305405, 0.305252, 0.305099, 0.304945, 0.304791, 0.304642, 0.304497, 0.304351, 0.304205, 0.304058,
              0.303917, 0.303780, 0.303642, 0.303504, 0.303365, 0.303229, 0.303100, 0.302971, 0.302841, 0.302711,
              0.302580, 0.302457, 0.302336, 0.302215, 0.302094, 0.301973, 0.301852, 0.301738, 0.301626, 0.301515,
              0.301403, 0.301291, 0.301180, 0.301071, 0.300970, 0.300868, 0.300767, 0.300665, 0.300564, 0.300463,
              0.300366, 0.300275, 0.300185, 0.300095, 0.300004, 0.299914, 0.299825, 0.299736, 0.299653, 0.299574,
              0.299495, 0.299417, 0.299339, 0.299261, 0.299184, 0.299107, 0.299030, 0.298962, 0.298896, 0.298830,
              0.298764, 0.298699, 0.298635, 0.298570, 0.298506, 0.298443, 0.298380, 0.298323, 0.298270, 0.298218,
              0.298168, 0.298119, 0.298070, 0.298023, 0.297977, 0.297932, 0.297887, 0.297844, 0.297800, 0.297757,
              0.297714] # diffusion
    y1k_df = [y - 0.07 for y in y1k_df]  # adjust value of y
    arr_dm = y1k_df
    ttl = len(arr_dm)
    xd_arr = np.linspace(1., 0., num=ttl, endpoint=True)
    arr_rf = [arr_dm[0], arr_dm[-1]]
    xr_arr = [1., 0.]
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', labelsize=30)
    ax.plot(xd_arr, arr_dm, linestyle='-', color='r')
    ax.plot(xr_arr, arr_rf, linestyle='-', color='b')
    legends = ['Diffusion models trajectory', 'Rectified Flow trajectory']
    ax.legend(legends, fontsize=25, loc='upper right')
    ax.set_title(f"Sampling Trajectories", fontsize=40)
    ax.set_xlabel(r"$t$",  fontsize=35)
    ax.set_ylabel(r"$x_t[a]$        ", fontsize=35, rotation=0)
    ax.set_ylim((0.1, 1.0))
    ax.invert_xaxis()

    img_noise = plt.imread("./charts/trajectory_dm_vs_rf/fig_trajectory1_noise.png")
    bbx_noise = OffsetImage(img_noise, zoom=0.6)
    bbx_noise.image.axes = ax
    ab = AnnotationBbox(bbx_noise, [0.99, 0.94], pad=0)
    ax.add_artist(ab)

    img_sample = plt.imread("./charts/trajectory_dm_vs_rf/fig_trajectory1_sample.png")
    bbx_sample = OffsetImage(img_sample, zoom=0.6)
    bbx_sample.image.axes = ax
    ab = AnnotationBbox(bbx_sample, [0.01, 0.16], pad=0)
    ax.add_artist(ab)

    fig.tight_layout()
    f_path = './charts/trajectory_dm_vs_rf/fig_trajectory1_diffusion_vs_rectified_flow.png'
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
    # reverse_time_ode_gradient()
    # gradient_variance()
    # track_fid_when_training()
    trajectory_diffusion_vs_rectified_flow()

if __name__ == '__main__':
    main()
