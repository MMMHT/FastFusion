import os
import os.path
import random
import cv2
import torch
import numpy as np
from skimage import io
from utils import *
import data_loader.readpfm as rp
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data.dataset import Dataset

cfg= options.get_config()

class SceneFlowLoader(Dataset):

    FIXED_SHAPE = (540, 960)

    def __init__(self, root_dir):

        self.left_all = np.load(os.path.join(root_dir,'left.npy'))
        self.right_all = np.load(os.path.join(root_dir,'right.npy'))

        self.left_all = torch.from_numpy(self.left_all.astype(np.float32)).float()
        self.right_all = torch.from_numpy(self.right_all.astype(np.float32)).float()


    def __getitem__(self, idx):
        data = dict()

        # pre_left:   0 dim: img
        #             1 dim: gt disparity
        #             2 dim: left slidar
        #             3 dim: pred

        data['left_img'] = self.left_all[idx,:,:,0].unsqueeze(0)
        data['right_img'] = self.right_all[idx,:,:,0].unsqueeze(0)
        data['left_disp'] = self.left_all[idx,:,:,1].unsqueeze(0)
        data['right_disp'] = self.right_all[idx,:,:,1].unsqueeze(0)
        data['left_slidar'] = self.left_all[idx,:,:,2].unsqueeze(0)
        data['right_slidar'] = self.right_all[idx,:,:,2].unsqueeze(0)
        data['stereobit_pre'] = self.left_all[idx,:,:,3].unsqueeze(0)
        data['stereobit_pre'] = self.right_all[idx,:,:,3].unsqueeze(0)

        return data

    def __len__(self):
        return self.left_all.shape[0]

