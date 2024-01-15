import os
import time

import numpy as np
import torch

from RectifiedFlow_Pytorch.rectified_flow_base import RectifiedFlowBase
from utils import log_info, get_time_ttl_and_eta

class ReflowTraining(RectifiedFlowBase):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.data_dir = args.data_dir
        self.seed = args.seed
        self.resume_ckpt_path = args.resume_ckpt_path
        self.eps = 1e-3
        log_info(f"ReflowTraining()")
        log_info(f"  data_dir: {self.data_dir}")
        log_info(f"  seed    : {self.seed}")
        log_info(f"  device  : {self.device}")
        log_info(f"  eps     : {self.eps}")
        self.model = None
        self.ema = None
        self.optimizer = None
        self.step = 0
        self.step_new = 0

    def get_data_pair(self):
        seed_dir = os.path.join(self.data_dir, str(self.seed))
        if not os.path.exists(seed_dir):
            raise ValueError(f"Dir not exist: {seed_dir}")
        # load data
        file_list = os.listdir(seed_dir)
        file_list.sort()
        log_info(f"  seed_dir   : {seed_dir}")
        log_info(f"  file cnt   : {len(file_list)}")
        z0_arr, z1_arr = [], []
        for f in file_list:
            f = str(f)
            if f.endswith("z0.npy"):
                z0 = np.load(os.path.join(seed_dir, f))
                z0 = torch.from_numpy(z0).cpu()
                z0_arr.append(z0)
            elif f.endswith("z1.npy"):
                z1 = np.load(os.path.join(seed_dir, f))
                z1 = torch.from_numpy(z1).cpu()
                z1_arr.append(z1)
            else:
                log_info(f" Warn: Unknown file: {f}")
        # for
        log_info(f"  z0 file cnt: {len(z0_arr)}")
        log_info(f"  z1 file cnt: {len(z1_arr)}")
        z0_ds = torch.cat(z0_arr, dim=0)    # z0 dataset
        z1_ds = torch.cat(z1_arr, dim=0)
        log_info(f"  z0 data cnt: {len(z0_ds)}")
        log_info(f"  z1 data cnt: {len(z1_ds)}")
        return z0_ds, z1_ds

    def train(self):
        log_info(f"ReflowTraining::train()")
        z0_ds, z1_ds = self.get_data_pair()
        states = self.load_ckpt(self.resume_ckpt_path, eval_mode=False, only_return_model=False)
        self.model     = states['model']
        self.ema       = states['ema']
        self.optimizer = states['optimizer']
        self.step      = states['step']
        args = self.args
        b_sz = args.batch_size
        e_cnt = args.n_epochs
        log_itv = args.log_interval
        img_cnt, c, h, w = z0_ds.shape
        b_cnt = img_cnt // b_sz
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        eb_cnt = e_cnt * b_cnt
        ds_limit = args.train_ds_limit
        lr = args.lr
        log_info(f"  loss_dual  : {args.loss_dual}")
        log_info(f"  loss_lambda: {args.loss_lambda}")
        log_info(f"  ds_limit   : {ds_limit}")
        log_info(f"  lr     : {lr}")
        log_info(f"  img_cnt: {img_cnt}")
        log_info(f"  channel: {c}")
        log_info(f"  height : {h}")
        log_info(f"  width  : {w}")
        log_info(f"  log_itv: {log_itv}")
        log_info(f"  b_sz   : {b_sz}")
        log_info(f"  b_cnt  : {b_cnt}")
        log_info(f"  e_cnt  : {e_cnt}")
        log_info(f"  eb_cnt : {eb_cnt}")
        start_time = time.time()
        for epoch in range(1, e_cnt+1):
            log_info(f"Epoch {epoch}/{e_cnt} ----------------lr:{lr}")
            indices = torch.randperm(img_cnt)
            counter = 0
            for b_idx in range(b_cnt):
                s = b_idx * b_sz    # start & end index of indices
                e = (b_idx + 1) * b_sz if b_idx + 1 < b_cnt else b_cnt
                idx = indices[s:e]  # index of data or noise
                data = z1_ds[idx].to(self.device).float()
                z0   = z0_ds[idx].to(self.device).float()
                counter += len(z0)
                loss, loss_adj, decay = self.train_batch(z0, data)
                if log_itv > 0 and b_idx % log_itv == 0:
                    elp, eta = get_time_ttl_and_eta(start_time, epoch * b_cnt + b_idx, eb_cnt)
                    log_info(f"B{b_idx:03d}/{b_cnt} loss:{loss:6.4f}, adj:{loss_adj:6.4f}. elp:{elp}, eta:{eta}")
                if 0 < ds_limit <= counter:
                    log_info(f"break data iteration as counter({counter}) reaches {ds_limit}")
                    break
            # for
        # for
        self.save_ckpt(self.model, self.ema, self.optimizer, e_cnt, self.step, self.step_new, False)

    def train_batch(self, z0, data):
        # the code logic is copied from:
        # RectifiedFlow\ImageGeneration\losses.py, line 161, step_fn()
        self.optimizer.zero_grad()
        if self.args.loss_dual:
            loss, loss_adj = self.calc_loss_dual(z0, data)
            loss_sum = loss + loss_adj * self.args.loss_lambda
        else:
            loss_sum = loss = self.calc_loss(z0, data)
            loss_adj = torch.tensor(0.)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
        self.optimizer.step()
        decay = self.ema.update(self.model.parameters())
        self.step_new += 1
        self.step += 1
        # here, we must return loss.item(), and not loss.
        # If return loss, it will cause memory leak.
        return loss.item(), loss_adj.item(), decay

    def calc_loss(self, z0, data):
        n, c, h, w = data.shape
        t = torch.rand(n, device=self.device) * (1.0 - self.eps)
        t = torch.add(t, self.eps)
        t_expand = t.view(-1, 1, 1, 1).repeat(1, c, h, w)
        perturbed_data = t_expand * data + (1. - t_expand) * z0
        target = data - z0
        score = self.model(perturbed_data, t * 999)
        loss = torch.square(score - target).mean()
        return loss

    def calc_loss_dual(self, z0, data):
        n, c, h, w = data.shape
        target = data - z0
        t1 = torch.rand(n, device=self.device)
        t1 = torch.mul(t1, 1.0 - self.eps)
        t1 = torch.add(t1, self.eps)
        t1_expand = t1.view(-1, 1, 1, 1)
        perturbed_data1 = t1_expand * data + (1. - t1_expand) * z0
        predict1 = self.model(perturbed_data1, t1 * 999)
        loss1 = self.compute_mse(predict1, target)

        t2 = torch.rand(n, device=self.device)
        t2 = torch.mul(t2, 1.0 - self.eps)
        t2 = torch.add(t2, self.eps)
        t2_expand = t2.view(-1, 1, 1, 1)
        perturbed_data2 = t2_expand * data + (1. - t2_expand) * z0
        predict2 = self.model(perturbed_data2, t2 * 999)
        loss2 = self.compute_mse(predict2, target)

        loss = (loss1 + loss2) / 2.
        loss_adj = self.compute_mse(predict1, predict2)
        return loss, loss_adj

    @staticmethod
    def compute_mse(value1, value2):
        # mse = (value1 - value2).square().sum(dim=(1, 2, 3)).mean(dim=0)
        mse = (value1 - value2).square().mean()
        return mse

# class
