import argparse
import sys
import os
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
    print(f"sys.path.append({cur_dir})")
if prt_dir not in sys.path:
    sys.path.append(prt_dir)
    print(f"sys.path.append({prt_dir})")

# This is for "ninja", which is necessary in model construction.
# "ninja" is an exe file, locates in the same folder as "python".
# Sample location: ~/anaconda3/envs/restflow/bin/
exe_dir = os.path.dirname(sys.executable)
env_path = os.environ['PATH']
if exe_dir not in env_path:
    os.environ['PATH'] = f"{exe_dir}:{env_path}"
    print(f"Environment variable PATH has inserted new dir: {exe_dir}")

from RectifiedFlow_Pytorch.configs.rectified_flow import bedroom_rf_gaussian
from RectifiedFlow_Pytorch.configs.rectified_flow import bedroom_rf_gaussian_reflow_gen_data
from RectifiedFlow_Pytorch.configs.rectified_flow import bedroom_rf_gaussian_reflow_train
from RectifiedFlow_Pytorch.configs.rectified_flow import cifar10_rf_gaussian_reflow_generate_data
from RectifiedFlow_Pytorch.configs.rectified_flow import cifar10_rf_gaussian_reflow_train
from RectifiedFlow_Pytorch.configs.rectified_flow import cifar10_rf_gaussian_ddpmpp
from RectifiedFlow_Pytorch.configs.rectified_flow import church_rf_gaussian
from RectifiedFlow_Pytorch.rectified_flow_sampling import RectifiedFlowSampling
from RectifiedFlow_Pytorch.rectified_flow_training import RectifiedFlowTraining
from RectifiedFlow_Pytorch.rectified_flow_misc import RectifiedFlowMiscellaneous
from RectifiedFlow_Pytorch.reflow_gen_data import ReflowGenerateData
from RectifiedFlow_Pytorch.reflow_training import ReflowTraining
from utils import str2bool, calc_fid
from utils import log_info as log_info

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default='bedroom2')
    parser.add_argument("--todo", type=str, default='train', help="train|sample|sample_all")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7, 6])
    parser.add_argument("--seed", type=int, default=1234, help="Random seed. 0 means ignore")
    parser.add_argument("--log_interval", type=int, default=5)

    # data
    parser.add_argument("--data_dir", type=str, default="../ddim/exp/datasets")
    parser.add_argument("--batch_size", type=int, default=8, help="0 mean to use size from config file")
    parser.add_argument("--train_ds_limit", type=int, default=100, help="training dataset limit")
    parser.add_argument("--test_ds_limit", type=int, default=0, help="testing dataset limit")

    # training
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=1, help="0 mean epoch number from config file")
    parser.add_argument('--ema_rate', type=float, default=0.999, help='mu in EMA. 0 means using value from config')
    parser.add_argument("--resume_ckpt_path", type=str, default='./checkpoint/ckpt_gnobitab_RF_LSUN_Bedroom.pth')
    parser.add_argument("--save_ckpt_path", type=str, default='./checkpoint_refined/ckpt.pth')
    parser.add_argument("--save_ckpt_interval", type=int, default=50, help="count by epoch")
    parser.add_argument("--save_ckpt_eval", type=str2bool, default=False, help="Calculate FID/IS when save ckpt")
    parser.add_argument("--loss_dual", type=str2bool, default=True, help="use dual loss")
    parser.add_argument("--loss_lambda", type=float, default=0.1, help="lambda when dual loss")

    # sampling
    parser.add_argument("--sample_count", type=int, default='50000', help="sample image count")
    parser.add_argument("--sample_batch_size", type=int, default=5, help="0 mean from config file")
    parser.add_argument("--sample_ckpt_path", type=str, default='./checkpoint/ckpt_gnobitab_RF_LSUN_Bedroom.pth')
    parser.add_argument("--sample_output_dir", type=str, default="./output5/generated")
    parser.add_argument("--sample_steps_arr", nargs='*', type=int, default=[10])
    parser.add_argument("--sample_init_ts_arr", nargs='*', type=float, default=[940])
    parser.add_argument("--sample_isc_flag", type=str2bool, default=False, help="calculate IS for samples")
    parser.add_argument("--fid_input1", type=str, default="../ddim/exp/datasets/lsun/bedroom_train")

    args = parser.parse_args()
    if args.config == 'bedroom' and 'reflow_gen' in args.todo:
        config = bedroom_rf_gaussian_reflow_gen_data.get_config()
    elif args.config == 'bedroom' and 'reflow_train' in args.todo:
        config = bedroom_rf_gaussian_reflow_train.get_config()
    elif args.config == 'bedroom':
        config = bedroom_rf_gaussian.get_config()
    elif args.config == 'bedroom2':
        config = bedroom_rf_gaussian.get_config()
        config.data.dataset = 'LSUN2'
    elif args.config in ['church', 'church_outdoor']:
        config = church_rf_gaussian.get_config()
    elif args.config == 'cifar10' and 'reflow_gen' in args.todo:
        config = cifar10_rf_gaussian_reflow_generate_data.get_config()
    elif args.config == 'cifar10' and 'reflow_train' in args.todo:
        config = cifar10_rf_gaussian_reflow_train.get_config()
    elif args.config == 'cifar10':
        config = cifar10_rf_gaussian_ddpmpp.get_config()
    else:
        raise ValueError(f"Invalid args.config: {args.config}")

    # add device
    gpu_ids = args.gpu_ids
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    args.device = device
    log_info(f"gpu_ids : {gpu_ids}")
    log_info(f"device  : {device}")

    # set random seed
    seed = args.seed  # if seed is 0. then ignore it.
    log_info(f"args.seed : {seed}")
    if seed:
        log_info(f"  torch.manual_seed({seed})")
        log_info(f"  np.random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
    if seed and torch.cuda.is_available():
        log_info(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed_all(seed)
    log_info(f"final seed: torch.initial_seed(): {torch.initial_seed()}")

    cudnn.benchmark = True
    return args, config

def sample_all(args, config):
    steps_arr = args.sample_steps_arr
    init_ts_arr = args.sample_init_ts_arr
    basename = os.path.basename(args.sample_ckpt_path)
    stem, ext = os.path.splitext(basename)
    result_file = f"./sample_all_rf_{stem}.txt"
    log_info(f"main_pytorch->sample_all()")
    log_info(f"  init_ts_arr : {init_ts_arr}")
    log_info(f"  steps_arr   : {steps_arr}")
    log_info(f"  result_file : {result_file}")
    for init_ts in init_ts_arr:
        result_file = f"./sample_all_rf_{stem}_initTS{init_ts:.3f}.txt"
        res_arr = []
        for steps in steps_arr:
            runner = RectifiedFlowSampling(args, config)
            runner.sample(sample_steps=steps, init_ts=init_ts)
            del runner  # delete the GPU memory. And can calculate FID
            torch.cuda.empty_cache()
            log_info(f"sleep 5 seconds to empty the GPU cache. . .")
            time.sleep(5)
            fid = calc_fid(args.gpu_ids[0], True, input1=args.fid_input1)
            msg = f"FID: {fid:7.3f}. steps:{steps:2d}"
            res_arr.append(msg)
            with open(result_file, 'w') as fptr: [fptr.write(f"{m}\n") for m in res_arr]
            log_info(msg)
            log_info("")
            log_info("")
            log_info("")
        # for
        [log_info(f"{msg}") for msg in res_arr]
    # for

def main():
    args, config = parse_args_and_config()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"host: {os.uname().nodename}")
    log_info(f"args: {args}")

    log_info(f"main_pytorch -> {args.todo} ===================================")
    if args.todo == 'sample':
        runner = RectifiedFlowSampling(args, config)
        runner.sample(sample_steps=args.sample_steps_arr[0])
    elif args.todo == 'train':
        runner = RectifiedFlowTraining(args, config)
        runner.train()
    elif args.todo == 'run_delta':
        runner = RectifiedFlowMiscellaneous(args, config)
        runner.run_delta_between_prediction_and_ground_truth()
    elif args.todo == 'compare_distance':
        runner = RectifiedFlowMiscellaneous(args, config)
        runner.compare_distance()
    elif args.todo == 'calc_gradient_var':
        runner = RectifiedFlowMiscellaneous(args, config)
        runner.calc_gradient_var()
    elif args.todo == 'sample_all':
        sample_all(args, config)
    elif args.todo == 'reflow_gen':
        runner = ReflowGenerateData(args, config)
        runner.gen_data()
    elif args.todo == 'reflow_train':
        runner = ReflowTraining(args, config)
        runner.train()
    else:
        raise Exception(f"Invalid todo: {args.todo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
