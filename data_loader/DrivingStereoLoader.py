import os
import random
import subprocess

import cv2
import numpy as np
from easydict import EasyDict
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils import options
import torch

cfg= options.get_config()

class DrivingStereoLoader(Dataset):
    # FIXED_SHAPE = (cfg.height, cfg.width)
    FIXED_SHAPE = (400, 879)

    def  selecter(self, scene):
        return {'cloudy':self.cloudy,
                'rainy':self.rainy,
                'foggy':self.foggy,
                'sunny':self.sunny,
                'testing':self.testing}.get(scene)

    def __init__(self, root_dir, mode, fix_random_seed=True):
        # Check arguments
        self.root_dir = root_dir
        self.mode = mode

        if fix_random_seed:
            random.seed(cfg.seed)
            np.random.seed(seed=cfg.seed)

        self.cloudy, self.rainy, self.sunny, self.foggy, self.testing = dict(), dict(), dict(), dict(),dict(),

        # Get all data path
        self.left_data_path, self.right_data_path, self.disp_data_path,\
            self.pred_data_path, self.slidar_left_data_path, self.slidar_right_data_path= get_DrivingStereo_datapath(self.root_dir, self.mode)

    def __getitem__(self, idx):


        data  =  dict()
        # Get data
        if self.mode == 'fourweather':
            for scene in ['cloudy', 'rainy', 'sunny', 'foggy']:
                img_left = read_gray(self.left_data_path[scene][idx])
                img_right = read_gray(self.right_data_path[scene][idx])
                img_left = img_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]
                img_right = img_right[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                disp_left = read_depth(self.disp_data_path[scene][idx])
                disp_left = disp_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                pred_left = read_disparity(self.pred_data_path[scene][idx])
                pred_left = pred_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                slidar_left = read_depth(self.slidar_left_data_path[scene][idx])
                slidar_right = read_depth(self.slidar_right_data_path[scene][idx])
                slidar_left = slidar_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]
                slidar_right = slidar_right[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                self.selecter(scene)['img_left'] = img_left
                self.selecter(scene)['img_right'] = img_right
                self.selecter(scene)['disp_left'] = disp_left
                self.selecter(scene)['pred_left'] = pred_left
                self.selecter(scene)['slidar_left'] = slidar_left
                self.selecter(scene)['slidar_right'] = slidar_right
                data[scene] = self.selecter(scene)
        else:
            for scene in ['testing']:
                img_left = read_gray(self.left_data_path[scene][idx])
                img_right = read_gray(self.right_data_path[scene][idx])
                img_left = img_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]
                img_right = img_right[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                disp_left = read_depth(self.disp_data_path[scene][idx])
                disp_left = disp_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                pred_left = read_disparity(self.pred_data_path[scene][idx])
                pred_left = pred_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                slidar_left = read_depth(self.slidar_left_data_path[scene][idx])
                slidar_right = read_depth(self.slidar_right_data_path[scene][idx])
                slidar_left = slidar_left[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]
                slidar_right = slidar_right[:, 0:self.FIXED_SHAPE[0], 0:self.FIXED_SHAPE[1]]

                self.selecter(scene)['img_left'] = img_left
                self.selecter(scene)['img_right'] = img_right
                self.selecter(scene)['disp_left'] = disp_left
                self.selecter(scene)['pred_left'] = pred_left
                self.selecter(scene)['slidar_left'] = slidar_left
                self.selecter(scene)['slidar_right'] = slidar_right
                data[scene] = self.selecter(scene)

        return data

    def __len__(self):
        # return len(self.left_data_path['cloudy'])
        return len(self.left_data_path['testing'])

def read_gray(path):
    """ Read raw RGB and perform rbg -> gray process to the image """
    gray = cv2.imread(path, 0)
    return torch.from_numpy(gray).unsqueeze(0)

def read_disparity(path):
    disp = io.imread(path)
    return torch.from_numpy(disp.astype(np.float32)).unsqueeze(0)

def read_depth(path):
    """ Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
        which can be opened with either MATLAB, libpng++ or the latest version of
        Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
        (ie, no ground truth exists, or the estimation algorithm didn't produce an
        estimate for that pixel). Otherwise, the depth for a pixel can be computed
        in meters by converting the uint16 value to float and dividing it by 256.0:

        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
    """
    depth = Image.open(path)
    depth = np.array(depth).astype(np.float32) / 256.0
    return torch.from_numpy(depth).unsqueeze(0)

def random_sampler(left_disp, sample_percentage):
    # perform random sampling on the left disparity

    sample_mask = torch.rand(left_disp.shape).le(sample_percentage)
    valid_mask = left_disp > 0
    sample_mask = sample_mask * valid_mask
    left_slidar =left_disp * sample_mask

    # perform random sampling on the right disparity
    right_slidar = torch.zeros_like(left_slidar)
    for i in range(left_slidar.shape[1]):
        for j in range(left_slidar.shape[2]):
            if sample_mask[0, i, j] == 1:
                val = left_slidar[0][i][j]
                # val = val.round().item()
                if j - val > 1 and i > 1 and i < left_slidar.shape[2]:
                    right_slidar[0][i][int(j - val)] = val

    return left_slidar, right_slidar

def fix_sampler(input, sample_threshold):

    sum = input.shape[1] * input.shape[2]
    valid = input.gt(0).sum()
    expected = (sum* sample_threshold).__round__()
    sample_percent_lidar = expected/valid

    return sample_percent_lidar



def get_DrivingStereo_datapath(root_dir, mode):
    if mode == 'fourweather':
        # """ Read path to all data from KITTI Stereo 2015 dataset """
        img_left_path_data = {'cloudy': [], 'foggy': [], 'rainy': [], 'sunny': []}
        img_right_path_data = {'cloudy': [], 'foggy': [], 'rainy': [], 'sunny': []}
        disp_left_path_data = {'cloudy': [], 'foggy': [], 'rainy': [], 'sunny': []}
        pred_left_path_data = {'cloudy': [], 'foggy': [], 'rainy': [], 'sunny': []}
        slidar_left_path_data = {'cloudy': [], 'foggy': [], 'rainy': [], 'sunny': []}
        slidar_right_path_data = {'cloudy': [], 'foggy': [], 'rainy': [], 'sunny': []}

        slidar_threshold = 0.03

        for scene in {'cloudy', 'foggy', 'rainy', 'sunny'} :
            for filename in sorted(os.listdir(os.path.join(root_dir, 'DrivingStereo', 'training', 'disparity-map-half-size', scene, 'disparity-map-half-size' ))):

                filename = filename[0:-4]

                # prepare image and disparity
                img_left_path = os.path.join(os.path.join(root_dir, 'DrivingStereo', 'training', 'left-image-half-size', scene, 'left-image-half-size' , filename+'.jpg'))
                img_right_path = os.path.join(os.path.join(root_dir, 'DrivingStereo', 'training', 'right-image-half-size', scene, 'right-image-half-size' , filename+'.jpg'))
                disp_left_path = os.path.join(os.path.join(root_dir, 'DrivingStereo', 'training', 'disparity-map-half-size', scene, 'disparity-map-half-size' , filename+'.png'))

                img_left_path_data[scene].append(img_left_path)
                img_right_path_data[scene].append(img_right_path)
                disp_left_path_data[scene].append(disp_left_path)

                # prepare slidar
                slidar_left_path = os.path.join(root_dir, 'DrivingStereo', 'stereobit', scene, 'slidar_left',  filename+'.png')
                slidar_right_path = os.path.join(root_dir, 'DrivingStereo', 'stereobit', scene, 'slidar_right',  filename+'.png')

                if (not os.path.exists(slidar_left_path or slidar_right_path)):

                    disp = read_depth(disp_left_path)
                    percent = fix_sampler(disp, slidar_threshold)
                    slidar_left, slidar_right = random_sampler(disp, percent)

                    print(slidar_left.gt(0).sum())

                    if (not os.path.exists(os.path.join(root_dir,'DrivingStereo', 'stereobit', scene, 'slidar_right'))):
                        os.makedirs(os.path.join(root_dir, 'DrivingStereo', 'stereobit', scene,  'slidar_right'), 0o777)
                    if (not os.path.exists(os.path.join(root_dir, 'DrivingStereo', 'stereobit', scene, 'slidar_left'))):
                        os.makedirs(os.path.join(root_dir, 'DrivingStereo', 'stereobit', scene,  'slidar_left'), 0o777)

                    io.imsave(slidar_left_path , (slidar_left.squeeze(0).numpy() * 256).astype('uint16'))
                    io.imsave(slidar_right_path, (slidar_right.squeeze(0).numpy() * 256).astype('uint16'))

                slidar_left_path_data[scene].append(slidar_left_path)
                slidar_right_path_data[scene].append(slidar_right_path)

                if (not os.path.exists(os.path.join(root_dir,'DrivingStereo', 'stereobit', scene, 'pred'))):
                    os.makedirs(os.path.join(root_dir, 'DrivingStereo', 'stereobit', scene,  'pred'), 0o777)

                pred_path = os.path.join(root_dir,'DrivingStereo', 'stereobit', scene, 'pred', filename+'.png')
                if (not os.path.exists(pred_path)):
                    os.system('./../StereoBit/StereoBit_DATE/cmake-build-debug-neu/tools/StereoBit submit_DrivingStereo 25 175 20 190 '
                              '{} {} {} {} {} {} {}'.format(img_left_path, img_right_path, slidar_left_path, slidar_right_path, filename+'.png',
                                                            os.path.join(root_dir, 'DrivingStereo', 'stereobit', scene,  'pred'),
                                                            '/home/haitao/projects/StereoBit/StereoBit_DATE/models/sb/gray_net_simpleweight_2.86_3.01.sb'))

                # pred = read_disparity(os.path.join(root_dir,'DrivingStereo', 'stereobit', scene, 'pred', filename+'.png'))
                pred_left_path_data[scene].append(pred_path)

        return img_left_path_data, img_right_path_data, disp_left_path_data, pred_left_path_data, slidar_left_path_data, slidar_right_path_data

    else:
        img_left_path_data = {'testing': []}
        img_right_path_data = {'testing': []}
        disp_left_path_data = {'testing': []}
        pred_left_path_data = {'testing': []}
        slidar_left_path_data = {'testing': []}
        slidar_right_path_data = {'testing': []}

        slidar_threshold = 0.03

        # for folder in os.listdir(os.path.join(root_dir, 'DrivingStereo', 'testing', 'disparity-map-half-size' )):
        for folder in ["2018-07-11-14-48-52"]:
            print(folder)
            for filename in sorted(os.listdir(os.path.join(root_dir, 'DrivingStereo', 'testing', 'disparity-map-half-size' , folder))):

                filename = filename[0:-4]

                # prepare image and disparity
                img_left_path = os.path.join(root_dir, 'DrivingStereo', 'testing', 'left-image-half-size', folder, filename+'.jpg')
                img_right_path = os.path.join(root_dir, 'DrivingStereo', 'testing', 'right-image-half-size', folder, filename+'.jpg')
                disp_left_path = os.path.join(root_dir, 'DrivingStereo', 'testing', 'disparity-map-half-size', folder, filename+'.png')

                img_left_path_data['testing'].append(img_left_path)
                img_right_path_data['testing'].append(img_right_path)
                disp_left_path_data['testing'].append(disp_left_path)

                # prepare slidar
                slidar_left_path = os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder, 'slidar_left',  filename+'.png')
                slidar_right_path = os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder, 'slidar_right',  filename+'.png')

                if (not os.path.exists(slidar_left_path or slidar_right_path)):

                    disp = read_depth(disp_left_path)
                    percent = fix_sampler(disp, slidar_threshold)
                    slidar_left, slidar_right = random_sampler(disp, percent)

                    print(slidar_left.gt(0).sum())

                    if (not os.path.exists(os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder, 'slidar_right'))):
                        os.makedirs(os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder, 'slidar_right'), 0o777)
                    if (not os.path.exists(os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder, 'slidar_left'))):
                        os.makedirs(os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder, 'slidar_left'), 0o777)

                    io.imsave(slidar_left_path , (slidar_left.squeeze(0).numpy() * 256).astype('uint16'))
                    io.imsave(slidar_right_path, (slidar_right.squeeze(0).numpy() * 256).astype('uint16'))

                slidar_left_path_data['testing'].append(slidar_left_path)
                slidar_right_path_data['testing'].append(slidar_right_path)

                if (not os.path.exists(os.path.join(root_dir,'DrivingStereo', 'stereobit', 'testing' , folder, 'pred'))):
                    os.makedirs(os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder,  'pred'), 0o777)

                # print(img_left_path)
                # print(img_right_path)
                # print(slidar_left_path)
                # print(slidar_right_path)
                # print(filename)
                # print(os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder,  'pred'))

                pred_path = os.path.join(root_dir,'DrivingStereo', 'stereobit', 'testing' , folder, 'pred', filename+'.png')
                if (not os.path.exists(pred_path)):
                    os.system('./../StereoBit/StereoBit_DATE/cmake-build-debug-neu/tools/StereoBit submit_DrivingStereo 25 175 20 190 '
                              '{} {} {} {} {} {} {}'.format(img_left_path, img_right_path, slidar_left_path, slidar_right_path, filename+'.png',
                                                            os.path.join(root_dir, 'DrivingStereo', 'stereobit', 'testing' , folder,  'pred'),
                                                            '/home/haitao/projects/StereoBit/StereoBit_DATE/models/sb/gray_net_simpleweight_2.86_3.01.sb'))

                # pred = read_disparity(os.path.join(root_dir,'DrivingStereo', 'stereobit', scene, 'pred', filename+'.png'))
                pred_left_path_data['testing'].append(pred_path)

        return img_left_path_data, img_right_path_data, disp_left_path_data, pred_left_path_data, slidar_left_path_data, slidar_right_path_data