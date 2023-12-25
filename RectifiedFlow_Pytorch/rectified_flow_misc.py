"""
Rectified Flow sampling
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

class RectifiedFlowMiscellaneous:
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.device = device
        self.model = None
        self.ema = None
        self.optimizer = None
        self.eps = 1e-3
        log_info(f"RectifiedFlowMiscellaneous()")
        log_info(f"  eps        : {self.eps}")
        log_info(f"  device     : {self.device}")

    def init_model_ema_optimizer(self):
        """Create the score model."""
        args, config = self.args, self.config

        def get_optimizer(params):
            """Returns a flax optimizer object based on `config`."""
            co = self.config.optim
            lr, beta1, eps, w_decay = args.lr, co.beta1, co.eps, co.weight_decay
            opt = optim.Adam(params, lr=lr, betas=(beta1, 0.999), eps=eps, weight_decay=w_decay)
            log_info(f"  optimizer: {co.optimizer}")
            log_info(f"  lr       : {lr}")
            log_info(f"  beta1    : {beta1}")
            log_info(f"  eps      : {eps}")
            log_info(f"  w_decay  : {w_decay}")
            return opt

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
        ckpt_path = args.sample_ckpt_path
        log_info(f"  load ckpt: {ckpt_path} . . .")
        states = torch.load(ckpt_path, map_location=self.device)
        # print(states['model'].keys())
        # states is like this:
        # 'optimizer': states['optimizer'].state_dict(),
        # 'model'    : states['model'].state_dict(),
        # 'ema'      : states['ema'].state_dict(),
        # 'step'     : states['step']

        log_info(f"  states['step']: {states['step']}")
        model.load_state_dict(states['model'], strict=True)
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(states['ema'])
        optimizer = get_optimizer(model.parameters())
        optimizer.load_state_dict(states['optimizer'])
        log_info(f"  model.load_state_dict(states['model'], strict=True)")
        log_info(f"  ema.load_state_dict(states['ema'])")
        log_info(f"  optimizer.load_state_dict(states['optimizer'])")
        log_info(f"  load ckpt: {ckpt_path} . . . Done")

        self.model = model
        self.ema = ema
        self.optimizer = optimizer

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

    def save_image_one_by_one(self):
        """ Load images from dataset or data-loader, and save them one by one """
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders(train_shuffle=False)
        import torchvision.utils as tvu
        # config.data.category is like: "church_outdoor"
        root_dir = os.path.join("../ddim/exp/datasets/lsun", f"{config.data.category}_train")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            log_info(f"os.makedirs({root_dir})")
        counter = 0
        for x_batch, y_batch in train_loader:
            for x, y in zip(x_batch, y_batch):
                path = os.path.join(root_dir, f"{counter:06d}.png")
                tvu.save_image(x, path)
                counter += 1
                if counter % 100 == 0: log_info(path)
            # for
        # for
        root_dir = os.path.join("../ddim/exp/datasets/lsun", f"{config.data.category}_val")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            log_info(f"os.makedirs({root_dir})")
        counter = 0
        for x_batch, y_batch in test_loader:
            for x, y in zip(x_batch, y_batch):
                path = os.path.join(root_dir, f"{counter:06d}.png")
                tvu.save_image(x, path)
                counter += 1
                if counter % 100 == 0: log_info(path)
            # for
        # for
        return True

    def run_delta_between_prediction_and_ground_truth(self):
        """ get the statistic info of the delta between prediction and ground_truth """
        train_loader, test_loader = self.get_data_loaders()
        self.init_model_ema_optimizer()
        self.model.eval()

        # ts_list = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        ts_list = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        loc_list = [
            (0, 0, 0), (0, 128, 128), (0, 255, 255),
            (1, 0, 0), (1, 128, 128), (1, 255, 255),
            (2, 0, 0), (2, 128, 128), (2, 255, 255),
        ]  # channel-height-width dimension
        b_cnt = len(train_loader)   # batch count
        b_sz = self.args.batch_size or self.config.training.batch_size
        ts_cnt = len(ts_list)
        log_info(f"RectifiedFlowMiscellaneous::run_delta_between_prediction_and_ground_truth()")
        log_info(f"  b_cnt   : {b_cnt}")
        log_info(f"  b_sz    : {b_sz}")
        log_info(f"  ts_cnt  : {ts_cnt}")
        log_info(f"  ts_list : {ts_list}")
        log_info(f"  loc_list: {loc_list}")
        b_ts_total = b_cnt * ts_cnt
        time_start = time.time()
        loc_cnt = len(loc_list)
        with torch.no_grad():
            for ts_idx, ts in enumerate(ts_list):
                train_loader, test_loader = self.get_data_loaders()
                delta_lst_lst = []
                for j in range(loc_cnt): delta_lst_lst.append([])
                for b_idx, (x, y) in enumerate(train_loader):
                    x = x.to(self.device)
                    x = x * 2.0 - 1.0
                    b_sz, ch, h, w = x.shape  # batch_size, channel, height, width
                    t = torch.ones((b_sz,), dtype=torch.float, device=self.device)
                    t = t * ts / 1000
                    t = torch.mul(t, 1.0 - self.eps)
                    t = torch.add(t, self.eps)
                    z0 = torch.randn(b_sz, ch, h, w, device=self.device)
                    t_expand = t.view(-1, 1, 1, 1)
                    perturbed_data = x * t_expand + z0 * (1. - t_expand)
                    target = x - z0
                    predict = self.model(perturbed_data, t * 999)
                    delta = predict - target
                    for i in range(b_sz):
                        for j in range(loc_cnt):
                            c, h, w = loc_list[j]
                            delta_lst_lst[j].append(delta[i][c][h][w].item())
                            # the ".item()" above is important.
                            # Else, the GPU memory will accumulate, and finally out of memory.
                        # for
                    # for
                    elp, eta = utils.get_time_ttl_and_eta(time_start, ts_idx * b_cnt + b_idx, b_ts_total)
                    log_info(f"B{b_idx:03d}/{b_cnt}.ts:{ts:03d}, t:{t[0]:.4f}. elp:{elp}, eta:{eta}")
                # for loader
                for delta_lst, (c, h, w) in zip(delta_lst_lst, loc_list):
                    f_path = f"./output5_lostats/ts{ts:03d}_dim{c}_{h:03d}_{w:03d}.txt"
                    log_info(f"write {len(delta_lst)} lines to {f_path}")
                    with open(f_path, 'w') as fptr:
                        [fptr.write(f"{d:10.7f}\r\n") for d in delta_lst]
                    # with
                # for
                del delta_lst_lst
                del train_loader
                del test_loader
                torch.cuda.empty_cache()
            # for ts
        # with

    # class
