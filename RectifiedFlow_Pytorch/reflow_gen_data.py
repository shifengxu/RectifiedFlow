# Re-Flow
# generate data pair from z0 by existing model checkpoint
import os.path
import time

import numpy as np
import torch

import sde_lib, utils
from RectifiedFlow_Pytorch.rectified_flow_base import RectifiedFlowBase
from utils import log_info

class ReflowGenerateData(RectifiedFlowBase):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.sample_ckpt_path  = args.sample_ckpt_path
        self.sample_output_dir = args.sample_output_dir
        self.sample_batch_size = args.sample_batch_size
        self.sample_count      = args.sample_count
        log_info(f"ReflowGenerateData()")
        log_info(f"  device           : {self.device}")
        log_info(f"  sample_ckpt_path : {self.sample_ckpt_path}")
        log_info(f"  sample_output_dir: {self.sample_output_dir}")
        log_info(f"  sample_batch_size: {self.sample_batch_size}")
        log_info(f"  sample_count     : {self.sample_count}")

    def gen_data(self):
        args, config = self.args, self.config
        data_root = self.sample_output_dir
        if not os.path.exists(data_root):
            log_info(f"os.makedirs({data_root})")
            os.makedirs(data_root)
        b_sz = self.sample_batch_size   # batch size
        img_cnt = self.sample_count     # image count
        b_cnt = img_cnt // b_sz         # batch count
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        c_data = config.data
        c, h, w = c_data.num_channels, c_data.image_size, c_data.image_size
        sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type,
                                    noise_scale=config.sampling.init_noise_scale,
                                    reflow_flag=True,
                                    reflow_t_schedule=config.reflow.reflow_t_schedule,
                                    reflow_loss=config.reflow.reflow_loss,
                                    use_ode_sampler=config.sampling.use_ode_sampler)
        score_model = self.load_ckpt(self.sample_ckpt_path, eval_mode=True, only_return_model=True)
        log_info(f"ReflowGenerateData::gen_data()")
        log_info(f"  seed   : {args.seed}")
        log_info(f"  save_to: {data_root}")
        log_info(f"  img_cnt: {img_cnt}")
        log_info(f"  b_cnt  : {b_cnt}")
        log_info(f"  b_sz   : {b_sz}")
        log_info(f"  c      : {c}")
        log_info(f"  h      : {h}")
        log_info(f"  w      : {w}")
        seed_dir = os.path.join(data_root, str(args.seed))
        if not os.path.exists(seed_dir):
            log_info(f"os.makedirs({seed_dir})")
            os.makedirs(seed_dir)
        start_time = time.time()
        z1_arr, z0_arr = [], []
        with torch.no_grad():
            for b_idx in range(b_cnt):
                # if img_cnt is 10 and b_sz is 4, then last batch has size 2.
                n = b_sz if b_idx + 1 < b_cnt else img_cnt - b_sz * b_idx
                z0 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                z1, nfe = sde.ode(z0, score_model)
                # np.save(os.path.join(seed_dir, f"B{b_idx:03d}_z1.npy"), z1.cpu().numpy())
                # np.save(os.path.join(seed_dir, f"B{b_idx:03d}_z0.npy"), z0.cpu().numpy())
                z0_arr.append(z0.cpu())
                z1_arr.append(z1.cpu())
                elp, eta = utils.get_time_ttl_and_eta(start_time, b_idx, b_cnt)
                log_info(f"B{b_idx:03d}/{b_cnt}: NFE: {nfe:3d}. elp:{elp}, eta:{eta}")
            # for
        # with
        z0 = torch.cat(z0_arr, dim=0)
        z1 = torch.cat(z1_arr, dim=0)
        np.save(os.path.join(seed_dir, "z0.npy"), z0.numpy())
        np.save(os.path.join(seed_dir, "z1.npy"), z1.numpy())

# class
