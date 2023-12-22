"""
Rectified Flow sampling
"""
import os
import time
import torch
import torchvision.utils as tvu
from RectifiedFlow_Pytorch import utils
from RectifiedFlow_Pytorch.models.ema import ExponentialMovingAverage
from RectifiedFlow_Pytorch.models.ncsnpp import NCSNpp
from utils import log_info as log_info


class RectifiedFlowSampling:
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device

    def create_model(self):
        """Create the score model."""
        args, config = self.args, self.config
        model_name = config.model.name
        log_info(f"  config.model.name: {model_name}")
        if model_name.lower() == 'ncsnpp':
            model = NCSNpp(config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        log_info(f"  model = model.to({self.device})")
        model = model.to(self.device)
        ckpt_path = args.sample_ckpt_path
        log_info(f"  load ckpt: {ckpt_path} . . .")
        states = torch.load(ckpt_path, map_location=self.device)
        # print(states['model'].keys())
        # states is like this:
        # 'optimizer': states['optimizer'].state_dict(),
        # 'model'    : states['model'].state_dict(),
        # 'ema'      : states['ema'].state_dict(),
        # 'step'     : states['step']
        log_info(f"  states['step']       : {states['step']}")
        log_info(f"  states['step_new']   : {states.get('step_new')}")
        log_info(f"  states['loss_dual']  : {states.get('loss_dual')}")
        log_info(f"  states['loss_lambda']: {states.get('loss_lambda')}")
        log_info(f"  states['pure_flag']  : {states.get('pure_flag')}")
        if states.get('pure_flag'):
            log_info(f"  model.load_state_dict(states['model'], strict=True)")
            model.load_state_dict(states['model'], strict=True)
            log_info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        else:
            # The checkpoint has key like "module.sigma",
            # so here model needs to be DataParallel.
            log_info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
            log_info(f"  model.load_state_dict(states['model'], strict=True)")
            model.load_state_dict(states['model'], strict=True)
        model.eval()
        log_info(f"  model.eval()")
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(states['ema'])
        ema.copy_to(model.parameters())
        log_info(f"  ema.load_state_dict(states['ema'])")
        log_info(f"  ema.copy_to(model.parameters())")
        log_info(f"  load ckpt: {ckpt_path} . . . Done")
        return model

    def sample(self, sample_steps=10):
        args, config = self.args, self.config
        log_info(f"RectifiedFlowSampling::sample(sample_steps={sample_steps})")
        model = self.create_model()
        img_cnt = args.sample_count
        b_sz = args.sample_batch_size
        b_cnt = img_cnt // b_sz
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        c_data = config.data
        c, h, w = c_data.num_channels, c_data.image_size, c_data.image_size
        log_info(f"  b_sz  : {b_sz}")
        log_info(f"  b_cnt : {b_cnt}")
        log_info(f"  c     : {c}")
        log_info(f"  h     : {h}")
        log_info(f"  w     : {w}")
        log_info(f"  steps : {sample_steps}")
        time_start = time.time()
        with torch.no_grad():
            for b_idx in range(b_cnt):
                n = img_cnt - b_idx * b_sz if b_idx == b_cnt - 1 else b_sz
                x1 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                x0 = self.sample_batch(x1, model, sample_steps, b_idx=b_idx)
                self.save_images(x0, time_start, b_cnt, b_idx, b_sz)
            # for
        # with
        return 0

    def sample_batch(self, x1, model, sample_steps, eps=1e-3, b_idx=-1):
        """
        sample a batch, starting from x1.
        From losses.py, where z0 is Gaussian noise:
            # standard rectified flow loss
            t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0
            target = batch - z0
            model_fn = mutils.get_model_fn(model, train=train)
            score = model_fn(perturbed_data, t*999)
        """
        b_sz = x1.size(0)
        dt = 1. / sample_steps
        x = x1
        for i in range(sample_steps):
            num_t = i / sample_steps * (1.0 - eps) + eps
            if b_idx == 0:
                log_info(f"sample_batch() i:{i:2d}, num_t:{num_t:.6f}")
            t = torch.ones(b_sz, requires_grad=False, device=self.device) * num_t
            pred = model(x, t * 999)
            x = x + pred * dt
        return x

    def save_images(self, x0, time_start, b_cnt, b_idx, b_sz):
        """ save x0 """
        x0 = (x0 + 1.0) / 2.0  # invert: [-1, 1] ==> [0, 1]
        img_cnt = len(x0)
        img_dir = self.args.sample_output_dir
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x0[i], img_path)
        elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx+1, b_cnt)
        log_info(f"saved {img_cnt}. B:{b_idx:3d}/{b_cnt}. {img_path}. elp:{elp}, eta:{eta}")

# class
