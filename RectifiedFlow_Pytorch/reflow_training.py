import os
import time
import torch
import torchvision.transforms as T
from torch.utils import data as tu_data

from RectifiedFlow_Pytorch.datasets import data_scaler
from RectifiedFlow_Pytorch.datasets.ImageNoiseDataset import ImageNoiseDataset
from RectifiedFlow_Pytorch.datasets.ImageNoiseNumpyDataset import ImageNoiseNumpyDataset
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
        self.dataset = None
        self.result_arr = []

    def get_data_loader(self):
        args = self.args
        if args.config == 'cifar10':
            ds = ImageNoiseNumpyDataset(self.data_dir)
        else:
            image_dir = os.path.join(self.data_dir, f"{args.seed}_image")
            noise_dir = os.path.join(self.data_dir, f"{args.seed}_noise")
            tfm = T.Compose([T.ToTensor()])
            ds = ImageNoiseDataset(image_dir, noise_dir, image_transform=tfm)
        self.dataset = ds
        dl = tu_data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        return dl

    def train(self):
        data_loader = self.get_data_loader()
        log_info(f"ReflowTraining::train()")
        ds_name = type(self.dataset).__name__
        scalar_flag = ds_name == 'ImageNoiseDataset'    # data scalar or not
        states = self.load_ckpt(self.resume_ckpt_path, eval_mode=False, only_return_model=False)
        self.model     = states['model']
        self.ema       = states['ema']
        self.optimizer = states['optimizer']
        self.step      = states['step']
        args, config = self.args, self.config
        b_sz = args.batch_size
        e_cnt = args.n_epochs
        log_itv = args.log_interval
        b_cnt = len(data_loader)
        eb_cnt = e_cnt * b_cnt
        ds_limit = args.train_ds_limit
        calc_var_int = self.args.calc_var_interval
        lr = args.lr
        log_info(f"  loss_dual  : {args.loss_dual}")
        log_info(f"  loss_lambda: {args.loss_lambda}")
        log_info(f"  ds_limit   : {ds_limit}")
        log_info(f"  ds_name    : {ds_name}")
        log_info(f"  scalar_flag: {scalar_flag}")
        log_info(f"  calc_var_int:{calc_var_int}")
        log_info(f"  lr     : {lr}")
        log_info(f"  img_cnt: {len(self.dataset)}")
        log_info(f"  log_itv: {log_itv}")
        log_info(f"  b_sz   : {b_sz}")
        log_info(f"  b_cnt  : {b_cnt}")
        log_info(f"  e_cnt  : {e_cnt}")
        log_info(f"  eb_cnt : {eb_cnt}")
        chw_flag = False
        start_time = time.time()
        b_ter = 0 # batch counter
        for epoch in range(1, e_cnt+1):
            log_info(f"Epoch {epoch}/{e_cnt} ----------------lr:{lr}")
            d_counter = 0 # data counter
            loss_sum, loss_adj_sum, loss_cnt = 0.0, 0.0, 0
            for b_idx, (data, z0) in enumerate(data_loader):
                b_ter += 1
                d_counter += len(z0)
                data, z0 = data.to(self.device), z0.to(self.device)
                if scalar_flag:
                    data = data_scaler(config, data)
                if not chw_flag:
                    chw_flag = True
                    n, c, h, w = data.shape
                    log_info(f"  channel: {c}")
                    log_info(f"  height : {h}")
                    log_info(f"  width  : {w}")
                loss, loss_adj, decay = self.train_batch(z0, data)
                loss_sum += loss
                loss_adj_sum += loss_adj
                loss_cnt += 1
                if log_itv > 0 and b_idx % log_itv == 0:
                    elp, eta = get_time_ttl_and_eta(start_time, b_ter, eb_cnt)
                    log_info(f"B{b_idx:03d}/{b_cnt} loss:{loss:8.6f}, adj:{loss_adj:8.6f}. elp:{elp}, eta:{eta}")
                if 0 < ds_limit <= d_counter:
                    log_info(f"break data iteration as counter({d_counter}) reaches {ds_limit}")
                    break
            # for
            loss_avg, loss_adj_avg = loss_sum / loss_cnt, loss_adj_sum / loss_cnt
            log_info(f"Epoch {epoch}/{e_cnt} loss_avg:{loss_avg:8.6f}, loss_adj_avg:{loss_adj_avg:8.6f}.")
            if calc_var_int > 0 and epoch % calc_var_int == 0:
                self.ema_calc_variance(epoch)
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

    def ema_calc_variance(self, epoch):
        def calc_gradient_var(eps=1e-3):
            dt = 1. / steps
            x = z0
            grad_arr = []
            for i in range(steps):
                num_t = i / steps * (1.0 - eps) + eps
                if b_idx == 0:
                    log_info(f"sample_batch() i:{i:2d}, num_t:{num_t:.6f}")
                t = torch.ones(n, requires_grad=False, device=self.device) * num_t
                grad = self.model(x, t * 999)
                grad_arr.append(grad)
                x = x + grad * dt
            return grad_arr

        log_info(f"ema_calc_variance()")
        args, config = self.args, self.config
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        self.model.eval()
        img_cnt     = args.sample_count
        b_sz        = args.sample_batch_size
        steps       = args.sample_steps_arr[0]
        b_cnt = img_cnt // b_sz
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        c_data = config.data
        c, h, w = c_data.num_channels, c_data.image_size, c_data.image_size
        log_info(f"  epoch  : {epoch}")
        log_info(f"  img_cnt: {img_cnt}")
        log_info(f"  b_sz   : {b_sz}")
        log_info(f"  b_cnt  : {b_cnt}")
        log_info(f"  c      : {c}")
        log_info(f"  h      : {h}")
        log_info(f"  w      : {w}")
        log_info(f"  steps  : {steps}")
        var_sum = 0.
        var_cnt = 0
        with torch.no_grad():
            for b_idx in range(b_cnt):
                log_info(f"calcVar B{b_idx:03d}/{b_cnt}")
                n = img_cnt - b_idx * b_sz if b_idx == b_cnt - 1 else b_sz
                z0 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                grad_arr = calc_gradient_var()
                grad_arr_t = torch.stack(grad_arr, dim=0)
                var_b_chw = torch.var(grad_arr_t, dim=0)
                var_mean = var_b_chw.mean()
                if b_idx == 0:
                    log_info(f"grad_arr_t: {grad_arr_t.shape}")
                    log_info(f"var_b_chw : {var_b_chw.shape}")
                    log_info(f"var_mean  : {var_mean.shape}")
                var_sum += var_mean
                var_cnt += 1
            # for
        # with
        self.ema.restore(self.model.parameters())
        self.model.train()
        var_avg = var_sum / var_cnt
        log_info(f"var_avg E{epoch:04d}:{var_avg:.8f}")
        self.result_arr.append(f"E{epoch:04d}: {var_avg:.8f}\n")
        basename = os.path.basename(args.save_ckpt_path)
        stem, ext = os.path.splitext(basename)
        f_path = f"./predicted_gradient_variance_{stem}.txt"
        with open(f_path, 'w') as fptr:
            fptr.write(f"# steps: {steps}\n")
            [fptr.write(s) for s in self.result_arr]
        # with
        return var_avg

# class
