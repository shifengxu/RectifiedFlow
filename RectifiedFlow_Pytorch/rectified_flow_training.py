"""
Rectified Flow Training
"""
import os.path
import time
import torch
import torch.utils.data as data
from torch import optim

from RectifiedFlow_Pytorch import utils
from RectifiedFlow_Pytorch.datasets import get_train_test_datasets
from RectifiedFlow_Pytorch.models.ncsnpp import NCSNpp
from utils import log_info as log_info
from models.ema import ExponentialMovingAverage

class RectifiedFlowTraining:
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device
        self.resume_ckpt_path = args.resume_ckpt_path
        self.model = None
        self.ema = None
        self.optimizer = None
        self.ema_rate = args.ema_rate
        self.eps = 1e-3
        self.step = 0
        self.step_new = 0
        log_info(f"RectifiedFlowTraining()")
        log_info(f"  resume_ckpt: {self.resume_ckpt_path}")
        log_info(f"  device     : {self.device}")
        log_info(f"  ema_rate   : {self.ema_rate}")
        log_info(f"  eps        : {self.eps}")
        log_info(f"  step       : {self.step}")
        log_info(f"  step_new   : {self.step_new}")

    def init_model_ema_optimizer(self):
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
        # The checkpoint has key like "module.sigma",
        # so here model needs to be DataParallel.
        log_info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        ckpt_path = self.resume_ckpt_path
        log_info(f"  load ckpt: {ckpt_path} . . .")
        states = torch.load(ckpt_path, map_location=self.device)
        # print(states['model'].keys())
        # states is like this:
        # 'optimizer': states['optimizer'].state_dict(),
        # 'model'    : states['model'].state_dict(),
        # 'ema'      : states['ema'].state_dict(),
        # 'step'     : states['step']

        log_info(f"  model.load_state_dict(states['model'], strict=True)")
        model.load_state_dict(states['model'], strict=True)
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
        log_info(f"  ema constructed.")
        log_info(f"  ema.decay      : {ema.decay}")
        log_info(f"  ema.num_updates: {ema.num_updates}")
        log_info(f"  ema.load_state_dict(states['ema'])")
        ema.load_state_dict(states['ema'])
        log_info(f"  ema.decay      : {ema.decay}")
        log_info(f"  ema.num_updates: {ema.num_updates}")
        optimizer = self.get_optimizer(model.parameters())
        log_info(f"  optimizer.load_state_dict(states['optimizer'])")
        optimizer.load_state_dict(states['optimizer'])
        step = states['step']
        log_info(f"  states['step']: {step}")
        log_info(f"  load ckpt: {ckpt_path} . . . Done")

        self.model = model
        self.ema = ema
        self.optimizer = optimizer
        self.step = step

    def save_checkpoint(self):
        ckpt_path = self.args.save_ckpt_path
        save_ckpt_dir = os.path.dirname(ckpt_path)
        if not os.path.exists(save_ckpt_dir):
            log_info(f"os.makedirs({save_ckpt_dir})")
            os.makedirs(save_ckpt_dir)
        log_info(f"resume_ckpt_path: {self.resume_ckpt_path}")
        log_info(f"Save latest ckpt: {ckpt_path} . . .")
        pure_model = self.model
        if isinstance(pure_model, torch.nn.DataParallel):
            # save pure model, not DataParallel.
            pure_model = pure_model.module
        saved_state = {
            'pure_flag'  : True,  # flag for pure model.
            'optimizer'  : self.optimizer.state_dict(),
            'model'      : pure_model.state_dict(),
            'ema'        : self.ema.state_dict(),
            'step'       : self.step,
            'step_new'   : self.step_new,
            'loss_dual'  : self.args.loss_dual,
            'loss_lambda': self.args.loss_lambda,
        }
        log_info(f"  pure_flag  : {saved_state['pure_flag']}")
        log_info(f"  optimizer  : {type(self.optimizer).__name__}")
        log_info(f"  model      : {type(pure_model).__name__}")
        log_info(f"  ema        : {type(self.ema).__name__}")
        log_info(f"  step       : {saved_state['step']}")
        log_info(f"  step_new   : {saved_state['step_new']}")
        log_info(f"  loss_dual  : {saved_state['loss_dual']}")
        log_info(f"  loss_lambda: {saved_state['loss_lambda']}")
        torch.save(saved_state, ckpt_path)
        log_info(f"Save latest ckpt: {ckpt_path} . . . Done")

    def get_optimizer(self, params):
        """Returns a flax optimizer object based on `config`."""
        args, co = self.args, self.config.optim
        lr, beta1, eps, w_decay = args.lr, co.beta1, co.eps, co.weight_decay
        optimizer = optim.Adam(params, lr=lr, betas=(beta1, 0.999), eps=eps, weight_decay=w_decay)
        log_info(f"  optimizer: {co.optimizer}")
        log_info(f"  lr       : {lr}")
        log_info(f"  beta1    : {beta1}")
        log_info(f"  eps      : {eps}")
        log_info(f"  w_decay  : {w_decay}")
        return optimizer

    def get_data_loaders(self, train_shuffle=True, test_shuffle=False):
        args, config = self.args, self.config
        batch_size = args.batch_size
        num_workers = 4
        train_ds, test_ds = get_train_test_datasets(args, config)
        train_loader = data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
        )
        log_info(f"train dataset and data loader:")
        log_info(f"  root       : {train_ds.root}")
        log_info(f"  split      : {train_ds.split}") if hasattr(train_ds, 'split') else None
        log_info(f"  classes    : {train_ds.classes}") if hasattr(train_ds, 'classes') else None
        log_info(f"  len        : {len(train_ds)}")
        log_info(f"  batch_cnt  : {len(train_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : {train_shuffle}")
        log_info(f"  num_workers: {num_workers}")

        test_loader = data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=test_shuffle,
            num_workers=num_workers,
        )
        log_info(f"test dataset and loader:")
        log_info(f"  root       : {test_ds.root}")
        log_info(f"  split      : {test_ds.split}") if hasattr(test_ds, 'split') else None
        log_info(f"  classes    : {test_ds.classes}") if hasattr(test_ds, 'classes') else None
        log_info(f"  len        : {len(test_ds)}")
        log_info(f"  batch_cnt  : {len(test_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : {test_shuffle}")
        log_info(f"  num_workers: {num_workers}")
        return train_loader, test_loader

    def train(self):
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders()
        self.init_model_ema_optimizer()
        log_interval = args.log_interval
        e_cnt = args.n_epochs       # epoch count
        b_cnt = len(train_loader)   # batch count
        eb_cnt = e_cnt * b_cnt      # epoch * batch
        lr = args.lr
        start_time = time.time()
        self.model.train()
        log_info(f"RectifiedFlowTraining::train()")
        log_info(f"  log_interval: {log_interval}")
        log_info(f"  image_size  : {config.data.image_size}")
        log_info(f"  b_sz        : {args.batch_size}")
        log_info(f"  lr          : {lr}")
        log_info(f"  loss_dual   : {args.loss_dual}")
        log_info(f"  loss_lambda : {args.loss_lambda}")
        log_info(f"  b_cnt       : {b_cnt}")
        log_info(f"  e_cnt       : {e_cnt}")
        log_info(f"  eb_cnt      : {eb_cnt}")
        for epoch in range(e_cnt):
            msg = f"lr={lr:8.7f}; ema_rate={self.ema_rate}"
            log_info(f"Epoch {epoch}/{e_cnt} ---------- {msg}")
            counter = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                x = 2.0 * x - 1.0
                loss, loss_adj, ema_decay = self.train_batch(x)
                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = utils.get_time_ttl_and_eta(start_time, epoch * b_cnt + i, eb_cnt)
                    loss_str = f"loss:{loss.item():6.4f}"
                    if self.args.loss_dual: loss_str += f", loss_adj:{loss_adj:6.4f}"
                    log_info(f"E{epoch}.B{i:03d}/{b_cnt} {loss_str}; ema:{ema_decay}. elp:{elp}, eta:{eta}")
                counter += x.size(0)
                if 0 < args.train_ds_limit <= counter:
                    log_info(f"break epoch: counter >= train_ds_limit ({counter} >= {args.train_ds_limit})")
                    break
            # for
        # for
        self.save_checkpoint()
        return 0

    def train_batch(self, x_batch):
        b_sz, ch, h, w = x_batch.shape # batch_size, channel, height, width
        z0 = torch.randn(b_sz, ch, h, w, device=self.device)
        self.optimizer.zero_grad()
        if self.args.loss_dual:
            loss, loss_adj = self.calc_loss_dual(x_batch, z0, b_sz)
            loss = loss + loss_adj * self.args.loss_lambda
        else:
            loss = self.calc_loss(x_batch, z0, b_sz)
            loss_adj = 0.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
        self.optimizer.step()
        decay = self.ema.update(self.model.parameters())
        self.step_new += 1
        self.step += 1
        return loss, loss_adj, decay

    def calc_loss(self, x_batch, z0, b_sz):
        target = x_batch - z0
        t = torch.rand(b_sz, device=self.device)
        t = torch.mul(t, 1.0 - self.eps)
        t = torch.add(t, self.eps)
        t_expand = t.view(-1, 1, 1, 1)
        perturbed_data = t_expand * x_batch + (1. - t_expand) * z0
        predict = self.model(perturbed_data, t * 999)
        loss = self.compute_mse(predict, target)
        return loss

    def calc_loss_dual(self, x_batch, z0, b_sz):
        target = x_batch - z0
        t1 = torch.rand(b_sz, device=self.device)
        t1 = torch.mul(t1, 1.0 - self.eps)
        t1 = torch.add(t1, self.eps)
        t1_expand = t1.view(-1, 1, 1, 1)
        perturbed_data1 = t1_expand * x_batch + (1. - t1_expand) * z0
        predict1 = self.model(perturbed_data1, t1 * 999)
        loss1 = self.compute_mse(predict1, target)

        t2 = torch.rand(b_sz, device=self.device)
        t2 = torch.mul(t2, 1.0 - self.eps)
        t2 = torch.add(t2, self.eps)
        t2_expand = t2.view(-1, 1, 1, 1)
        perturbed_data2 = t2_expand * x_batch + (1. - t2_expand) * z0
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
