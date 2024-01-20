# Re-Flow
# generate data pair from z0 by existing model checkpoint
import os.path
import time
import torch
import numpy as np
import torchvision.utils as tvu
from torch import Tensor

import sde_lib
import utils
from RectifiedFlow_Pytorch.datasets import data_inverse_scaler
from RectifiedFlow_Pytorch.rectified_flow_base import RectifiedFlowBase
from RectifiedFlow_Pytorch.rectified_flow_sampling import RectifiedFlowSampling
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

        def save_at_final():
            """ Save data at final. This is for small image, such as 32*32 size """
            seed_dir = os.path.join(data_root, str(args.seed))
            if not os.path.exists(seed_dir):
                log_info(f"os.makedirs({seed_dir})")
                os.makedirs(seed_dir)
            sde = sde_lib.RectifiedFlow(init_type=config.sampling.init_type,
                                        noise_scale=config.sampling.init_noise_scale,
                                        reflow_flag=True,
                                        reflow_t_schedule=config.reflow.reflow_t_schedule,
                                        reflow_loss=config.reflow.reflow_loss,
                                        use_ode_sampler=config.sampling.use_ode_sampler)
            z0_arr, z1_arr = [], []
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
                if b_idx == 0:
                    img_dir = seed_dir + '_sample'
                    self.save_images(z1, img_dir, b_idx, b_sz)
            # for
            z0 = torch.cat(z0_arr, dim=0)
            z1 = torch.cat(z1_arr, dim=0)
            np.save(os.path.join(seed_dir, "z0.npy"), z0.numpy())
            np.save(os.path.join(seed_dir, "z1.npy"), z1.numpy())

        def save_by_item():
            """
             Save data by item. This is for big image, such as 256*256 size.
             This means it saves noise one by one, and saves image one by one.
             A benefit is: we can calculate the FID for the generated images.
              """
            rf_sampler = RectifiedFlowSampling(args, config)
            image_dir = os.path.join(data_root, f"{args.seed}_image")
            noise_dir = os.path.join(data_root, f"{args.seed}_noise")
            if not os.path.exists(image_dir):
                log_info(f"os.makedirs({image_dir})")
                os.makedirs(image_dir)
            if not os.path.exists(noise_dir):
                log_info(f"os.makedirs({noise_dir})")
                os.makedirs(noise_dir)
            nfe = self.args.sample_steps_arr[0]
            for b_idx in range(b_cnt):
                # if img_cnt is 10 and b_sz is 4, then last batch has size 2.
                n = b_sz if b_idx + 1 < b_cnt else img_cnt - b_sz * b_idx
                z0 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                z1 = rf_sampler.sample_batch(z0, score_model, nfe, b_idx=b_idx)
                elp, eta = utils.get_time_ttl_and_eta(start_time, b_idx, b_cnt)
                log_info(f"B{b_idx:03d}/{b_cnt}: NFE: {nfe:3d}. elp:{elp}, eta:{eta}.")
                self.save_images(z1, image_dir, b_idx, b_sz)
                self.save_noises(z0, noise_dir, b_idx, b_sz)
            # for
            del rf_sampler
            torch.cuda.empty_cache()
            log_info(f"sleep 5 seconds to empty the GPU cache. . .")
            time.sleep(5)
            log_info(f"save_by_item() FID input1: {args.fid_input1}")
            log_info(f"save_by_item() FID input2: {image_dir}")
            fid = utils.calc_fid(args.gpu_ids[0], True, input1=args.fid_input1, input2=image_dir)
            log_info(f"save_by_item() FID: {fid:7.3f}. steps:{nfe:2d}")

        start_time = time.time()
        with torch.no_grad():
            if w < 100 and h < 100:
                save_at_final()
            else:
                save_by_item()
        # with

    def save_images(self, x0, img_dir, b_idx, b_sz):
        """ save x0 """
        x0 = data_inverse_scaler(self.config, x0)
        img_cnt = len(x0)
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x0[i], img_path)

    @staticmethod
    def save_noises(noise_batch: Tensor, noise_dir, b_idx, b_sz):
        n_cnt = len(noise_batch)
        for i in range(n_cnt):
            n_id = b_idx * b_sz + i
            n_path = os.path.join(noise_dir, f"{n_id:05d}.npy")
            np.save(n_path, noise_batch[i].cpu().numpy())

# class
