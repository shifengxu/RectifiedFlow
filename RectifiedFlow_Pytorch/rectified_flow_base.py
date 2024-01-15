# Re-Flow
# generate data pair from z0 by existing model checkpoint
import os
import torch
from torch import optim

from utils import log_info
from models.ema import ExponentialMovingAverage
from models.ncsnpp import NCSNpp

class RectifiedFlowBase:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = args.device

    def load_ckpt(self, ckpt_path, eval_mode, only_return_model):
        args, config = self.args, self.config
        model_name = config.model.name
        log_info(f"  config.model.name: {model_name}")
        if model_name.lower() == 'ncsnpp':
            model = NCSNpp(config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        log_info(f"  model = model.to({self.device})")
        model = model.to(self.device)
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
        if eval_mode:
            model.eval()
            log_info(f"  model.eval()")
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
        log_info(f"  ema constructed.")
        ema.load_state_dict(states['ema'])
        log_info(f"  ema.load_state_dict(states['ema'])")
        log_info(f"  ema.num_updates: {ema.num_updates}")
        log_info(f"  ema.decay (old): {ema.decay}")
        if eval_mode:
            ema.copy_to(model.parameters())
            log_info(f"  ema.copy_to(model.parameters())")
        else:
            ema.decay = args.ema_rate
            log_info(f"  ema.decay (new): {ema.decay}")

        if only_return_model:
            log_info(f"  load ckpt: {ckpt_path} . . . Done")
            return model

        # return all as a dict
        optimizer = self.get_optimizer(model.parameters())
        log_info(f"  optimizer.load_state_dict(states['optimizer'])")
        optimizer.load_state_dict(states['optimizer'])
        step = states['step']
        log_info(f"  states['step'] : {step}")
        epoch = states.get('epoch')
        log_info(f"  states['epoch']: {epoch}")

        log_info(f"  load ckpt: {ckpt_path} . . . Done")
        return {
            'model'     : model,
            'ema'       : ema,
            'optimizer' : optimizer,
            'step'      : step,
            'epoch'     : epoch,
        }

    def save_ckpt(self, model, ema, optimizer, epoch, step, step_new=None, epoch_in_file_name=False):
        ckpt_path = self.args.save_ckpt_path
        save_ckpt_dir, base_name = os.path.split(ckpt_path)
        if not os.path.exists(save_ckpt_dir):
            log_info(f"os.makedirs({save_ckpt_dir})")
            os.makedirs(save_ckpt_dir)
        if epoch_in_file_name:
            stem, ext = os.path.splitext(base_name)
            ckpt_path = os.path.join(save_ckpt_dir, f"{stem}_E{epoch:03d}{ext}")
        log_info(f"Save ckpt: {ckpt_path} . . .")
        pure_model = model
        if isinstance(pure_model, torch.nn.DataParallel):
            # save pure model, not DataParallel.
            pure_model = pure_model.module
        saved_state = {
            'pure_flag'  : True,  # flag for pure model.
            'optimizer'  : optimizer.state_dict(),
            'model'      : pure_model.state_dict(),
            'ema'        : ema.state_dict(),
            'epoch'      : epoch,
            'step'       : step,
            'step_new'   : step_new,
            'loss_dual'  : self.args.loss_dual,
            'loss_lambda': self.args.loss_lambda,
        }
        log_info(f"  pure_flag  : {saved_state['pure_flag']}")
        log_info(f"  optimizer  : {type(optimizer).__name__}")
        log_info(f"  model      : {type(pure_model).__name__}")
        log_info(f"  ema        : {type(ema).__name__}")
        log_info(f"  epoch      : {saved_state['epoch']}")
        log_info(f"  step       : {saved_state['step']}")
        log_info(f"  step_new   : {saved_state['step_new']}")
        log_info(f"  loss_dual  : {saved_state['loss_dual']}")
        log_info(f"  loss_lambda: {saved_state['loss_lambda']}")
        torch.save(saved_state, ckpt_path)
        log_info(f"Save ckpt: {ckpt_path} . . . Done")

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

# class
