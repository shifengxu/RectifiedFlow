import argparse
import subprocess
import re
import torch_fidelity
import torch.nn as nn
import math
import datetime
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_parameters(model: nn.Module, log_fn=None):
    def prt(x):
        if log_fn: log_fn(x)

    def convert_size_str(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    prt(f"count_parameters({type(model)}) ------------")
    prt('  requires_grad  name  count  size')
    counter = 0
    for name, param in model.named_parameters():
        s_list = list(param.size())
        prt(f"  {param.requires_grad} {name} {param.numel()} = {s_list}")
        c = param.numel()
        counter += c
    # for
    str_size = convert_size_str(counter)
    prt(f"  total  : {counter} {str_size}")
    return counter, str_size

def log_info(*args):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{dtstr}]", *args)

def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
    """
    Get estimated total time and ETA time.
    :param time_start:
    :param elapsed_iter:
    :param total_iter:
    :return: string of elapsed time, string of ETA
    """

    def sec_to_str(sec):
        val = int(sec)  # seconds in int type
        s = val % 60
        val = val // 60  # minutes
        m = val % 60
        val = val // 60  # hours
        h = val % 24
        d = val // 24  # days
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"

    elapsed_time = time.time() - time_start  # seconds elapsed
    elp = sec_to_str(elapsed_time)
    if elapsed_iter == 0:
        eta = 'NA'
    else:
        # seconds
        eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
        eta = sec_to_str(eta)
    return elp, eta

def calc_fid(gpu, fid_subprocess: bool = True, input1="cifar10-train", input2="./generated", logger=log_info):
    if fid_subprocess:
        cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
        logger(f"cmd: {cmd}")
        cmd_arr = cmd.split(' ')
        res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
        output = str(res.stdout)
        logger(f"out: {output}")  # frechet_inception_distance: 16.5485\n
        m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
        fid = float(m.group(1))
    else:
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=input1,
            input2=input2,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
            samples_find_deep=True,
        )
        fid = metrics_dict['frechet_inception_distance']
    return fid
