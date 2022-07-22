import torch
import json
import os
import sys
import shutil
import torch
from pathlib import Path
from termcolor import colored
from easydict import EasyDict
from collections import OrderedDict
from utils.options import get_config
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from stereobit import grey2jet

def visualize_correct_pred(src_img, pred, ground_truth):

    # src, pred, ground with [1,1,H,W]
    src_img = src_img.cpu()
    pred = pred.cpu()
    ground_truth = ground_truth.cpu()

    if src_img.shape[1] == 1:
        src_img = src_img.expand(1, 3, src_img.shape[2], src_img.shape[3])
    img_err = src_img.div(255).mul(50).add(150).div(255)

    valid_mask = (ground_truth > 0)
    pred_valid = pred*valid_mask
    target_valid = ground_truth*valid_mask
    actual = target_valid.mul(-1).add(pred_valid).abs()

    pred_bad= actual.gt(3).mul(valid_mask)
    pred_good= actual.le( 3).mul(valid_mask)

    scalar =  0.2
    img_err[:, 0,...] = img_err[:, 0,...].add(pred_bad.mul(scalar))
    img_err[:, 1,...] = img_err[:, 1,...].add(pred_bad.mul(-scalar))
    img_err[:, 2,...] = img_err[:, 2,...].add(pred_bad.mul(-scalar))
    img_err[:, 0,...] = img_err[:, 0,...].add(pred_good.mul(-scalar))
    img_err[:, 1,...] = img_err[:, 1,...].add(pred_good.mul(scalar))
    img_err[:, 2,...] = img_err[:, 2,...].add(pred_good.mul(-scalar))

    return img_err



def visualize_with_color(pred):
    if pred.ndim == 2 :
        color = torch.Tensor(3, pred.shape[0], pred.shape[1])
        grey2jet(pred.div(128).cpu(), color)
    elif pred.ndim == 3:
        pred = pred[0].clone()
        color = torch.Tensor(3, pred.shape[0], pred.shape[1])
        grey2jet(pred.div(128).cpu(), color)
    elif pred.ndim == 4:
        pred = pred[0][0].clone()
        color = torch.Tensor(3, pred.shape[0], pred.shape[1])
        grey2jet(pred.div(128).cpu(), color)

    return color

def save_png(img, str):
    if img.ndimension() == 4 :
        img = img.squeeze(0)
    if img.shape[0] != 3:
        img = img*256
    img = ToPILImage()(img.cpu().to(torch.float32))
    img.save( str + '.png')

def floating_lidar(img, lidar):
    img = img.squeeze(0)
    mask = lidar>0
    # for i in range(lidar.shape[1]):
    #     for j in range( lidar.shape[2]):
    #         if lidar[0,i,j] == 0 and lidar[1,i,j]==0:
    #             lidar[2,i,j] = 0
    img[mask] = lidar[mask]
    return img


def rgb_to_gray(rgb):
    transform = transforms.Grayscale(1)
    return transform(rgb)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

class Logger(object):
    """ Logger that can print on terminal and save log to file simultaneously """

    def __init__(self, log_path, mode='w'):
        """ Constructor of Logger
            Args:
                `log_path` (str): full path to log file
        """
        if mode == 'a':
            self._log_fout = open(log_path, 'a')
        elif mode == 'w':
            self._log_fout = open(log_path, 'w')
        else:
            raise ValueError('Invalid mode')

    def write(self, out_str, color='white', end='\n', print_out=True):
        """ Write log
            Args:
                `out_str` (str): string to be printed out and written to log file
        """
        self._log_fout.write(out_str + end)
        self._log_fout.flush()
        if print_out:
            print(colored(out_str, color), end=end)
        sys.stdout.flush()

def setup_workspace(name, mode):
    """ Setup workspace and backup important files """
    workspace = EasyDict()
    workspace.root = validate_dir(name)
    workspace.ckpt = validate_dir(os.path.join(name, 'ckpt'))
    workspace.log = validate_dir(os.path.join(name, 'log'))
    workspace.predictions = validate_dir(os.path.join(name, 'predictions'))

    # NOTE: check paths to options.py and train.py
    shutil.copyfile('./utils/options.py', os.path.join(workspace.root, '{}_options.py'.format(name.split('/')[-1])))
    shutil.copyfile('./train_{}.py'.format(mode), os.path.join(workspace.root, '{}_train.py'.format(name.split('/')[-1])))

    return workspace


def validate_dir(*dir_name, **kwargs):
    """
    Check and validate a directory
    Args:
        *dir_name (str / a list of str): a directory
        **kwargs:
            auto_mkdir (bool): automatically make directories. Default: True.
        Returns:
            dir_name (str): path to the directory
        Notes:
            1. `auto_mkdir` is performed recursively, e.g. given a/b/c,
               where a/b does not exist, it will create a/b and then a/b/c.
            2. using **kwargs is for future extension.
    """
    # parse argument
    if kwargs:
        auto_mkdir = kwargs.pop('auto_mkdir')
        if kwargs:
            raise ValueError('Invalid arguments: {}'.format(kwargs))
    else:
        auto_mkdir = True

    # check and validate directory
    dir_name = os.path.abspath(os.path.join(*dir_name))
    if auto_mkdir and not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    return dir_name