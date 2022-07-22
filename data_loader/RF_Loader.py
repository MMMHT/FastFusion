import os
import random
import numpy as np
from easydict import EasyDict
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils import options
from utils import save_png
import torch

cfg= options.get_config()

class RFLoader(Dataset):
    FIXED_SHAPE = (352, 1216)

    def __init__(self, root_dir, mode, output_size, fix_random_seed=True):
        dataset = '/kitti2017/selected'
        root_dir = root_dir + dataset
        self.left_all = np.load(os.path.join(root_dir,'selected_left.npy'))
        self.right_all = np.load(os.path.join(root_dir,'selected_right.npy'))

        self.left_all = torch.from_numpy(self.left_all).float()
        self.right_all = torch.from_numpy(self.right_all).float()

    def __getitem__(self, idx):
        data = dict()

        # pre_left:   0 dim: img
        #             1 dim: gt disparity
        #             2 dim: negativate gt disparity
        #             3 dim: sparse lidar input
        #             4 dim: inference by trained model(training set of RFNet)
        #             5 dim : inference by LEANet model (kitti2015)
        data['left_img'] = self.left_all[idx,:,:,0].unsqueeze(0)
        data['right_img'] = self.right_all[idx,:,:,0].unsqueeze(0)
        data['left_disp'] = torch.cat((self.left_all[idx,:,:,1].unsqueeze(0),self.left_all[idx,:,:,2].unsqueeze(0)))
        data['right_disp'] = torch.cat((self.right_all[idx,:,:,1].unsqueeze(0),self.right_all[idx,:,:,2].unsqueeze(0)))
        data['left_slidar'] = self.left_all[idx,:,:,3].unsqueeze(0)
        data['right_slidar'] = self.right_all[idx,:,:,3].unsqueeze(0)
        data['left_pre'] = self.left_all[idx,:,:,4].unsqueeze(0)
        # data['right_pre'] = self.right_all[idx,:,:,4].unsqueeze(0)
        data['left_leaDisp'] =  self.left_all[idx,:,:,5].unsqueeze(0)
        # data['right_leaDisp'] =  self.right_all[idx,:,:,5].unsqueeze(0)

        return data

    def __len__(self):
        return self.left_all.shape[0]
