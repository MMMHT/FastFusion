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

class Kitti141Loader(Dataset):
    FIXED_SHAPE = (370, 1242)

    def __init__(self, root_dir,  fix_random_seed=True,transform = None, is_rgb = False, is_norm = False):
        # Check arguments
        self.is_norm = is_norm
        self.is_rgb = is_rgb
        self.root_dir = root_dir + 'kitti141'
        self.mode = None
        self.aug_transform = transform

        if fix_random_seed:
            random.seed(cfg.seed)
            np.random.seed(seed=cfg.seed)

        # Get all data path

        self.left_data_path, self.right_data_path = getkitti141datapath(self.root_dir, self.mode)

        # Define data transform
        self.transform = EasyDict()

        if not self.is_rgb:
            self.transform.rgb = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
            ])
        else:
            self.transform.rgb = transforms.Compose([
            ])

        self.transform.depth = transforms.Compose([
            transforms.ToTensor()
        ])


    def __getitem__(self, idx):
        # Get data

        left_rgb = read_rgb(self.left_data_path['image_02'][idx], self.is_norm)
        img_h, img_w = left_rgb.shape[:2]
        right_rgb = read_rgb(self.right_data_path['image_03'][idx], self.is_norm)

        left_disp = read_depth(self.left_data_path['disp_noc_0'][idx])
        right_disp = read_depth(self.right_data_path['disp_noc_1'][idx])

        velodyne_left_64 = read_depth(self.left_data_path['velodyne_points'][idx])
        velodyne_right_64 = read_depth(self.right_data_path['velodyne_points_right'][idx])

        velodyne_left_0 = read_depth(self.left_data_path['velodyne_points0'][idx])
        velodyne_right_0 = read_depth(self.right_data_path['velodyne_points0_right'][idx])

        velodyne_left_1 = read_depth(self.left_data_path['velodyne_points1'][idx])
        velodyne_right_1 = read_depth(self.right_data_path['velodyne_points1_right'][idx])

        velodyne_left_8 = read_depth(self.left_data_path['velodyne_points8'][idx])
        velodyne_right_8 = read_depth(self.right_data_path['velodyne_points8_right'][idx])

        velodyne_left_16 = read_depth(self.left_data_path['velodyne_points16'][idx])
        velodyne_right_16 = read_depth(self.right_data_path['velodyne_points16_right'][idx])

        velodyne_left_32 = read_depth(self.left_data_path['velodyne_points32'][idx])
        velodyne_right_32 = read_depth(self.right_data_path['velodyne_points32_right'][idx])

        # Crop to fixed size
        def crop_fn(x):
            start_h = img_h - self.FIXED_SHAPE[0] if img_h > self.FIXED_SHAPE[0] else 0
            start_w = 0
            end_w = min(img_w, start_w + self.FIXED_SHAPE[1])
            return x[start_h:start_h + self.FIXED_SHAPE[0], start_w:end_w]

        left_rgb, left_disp = list(map(crop_fn, [left_rgb, left_disp]))
        right_rgb, right_disp = list(map(crop_fn, [right_rgb, right_disp]))
        velodyne_left_0, velodyne_right_0 = list(map(crop_fn, [velodyne_left_0, velodyne_right_0]))
        velodyne_left_1, velodyne_right_1 = list(map(crop_fn, [velodyne_left_1, velodyne_right_1]))
        velodyne_left_8, velodyne_right_8 = list(map(crop_fn, [velodyne_left_8, velodyne_right_8]))
        velodyne_left_16, velodyne_right_16 = list(map(crop_fn, [velodyne_left_16, velodyne_right_16]))
        velodyne_left_32, velodyne_right_32 = list(map(crop_fn, [velodyne_left_32, velodyne_right_32]))
        velodyne_left_64, velodyne_right_64 = list(map(crop_fn, [velodyne_left_64, velodyne_right_64]))

        if not self.is_norm :
            left_rgb = left_rgb.permute(2,0,1)
            right_rgb = right_rgb.permute(2,0,1)

        data = dict()
        data['left_img'], data['right_img'] = list(map(self.transform.rgb, [left_rgb, right_rgb]))
        data['left_disp'], data['right_disp'] = list(map(self.transform.depth, [left_disp, right_disp]))
        data['velodyne_points64'], data['velodyne_points_right'] = list(map(self.transform.depth, [velodyne_left_64, velodyne_right_64]))
        data['velodyne_points0'], data['velodyne_points0_right'] = list(map(self.transform.depth, [velodyne_left_0, velodyne_right_0]))
        data['velodyne_points1'], data['velodyne_points1_right'] = list(map(self.transform.depth, [velodyne_left_1, velodyne_right_1]))
        data['velodyne_points8'], data['velodyne_points8_right'] = list(map(self.transform.depth, [velodyne_left_8, velodyne_right_8]))
        data['velodyne_points16'], data['velodyne_points16_right'] = list(map(self.transform.depth, [velodyne_left_16, velodyne_right_16]))
        data['velodyne_points32'], data['velodyne_points32_right'] = list(map(self.transform.depth, [velodyne_left_32, velodyne_right_32]))
        data['width'] = img_w

        return data


    def __len__(self):
        return len(self.left_data_path['image_02'])


def read_rgb(path, is_norm):
    """ Read raw RGB and DO NOT perform any process to the image """
    rgb = io.imread(path)
    if not is_norm :
        rgb_tensor = torch.from_numpy(rgb)
        return rgb_tensor
    else:
        return rgb



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

    depth = np.array(depth).astype(np.float32) / 256
    return depth[:, :, np.newaxis]


def getkitti141datapath(root_dir , mode):

    left = ['velodyne_points', 'velodyne_points32', 'velodyne_points16',
             'velodyne_points8', 'velodyne_points1', 'velodyne_points0',
             'disp_noc_0',  'image_02']
    right = ['velodyne_points_right', 'velodyne_points32_right','velodyne_points16_right',
             'velodyne_points8_right','velodyne_points1_right', 'velodyne_points0_right',
            'disp_noc_1','image_03']

    # """ Read path to all data from converted KITTI Depth Completion dataset """
    left_data_path = {'image_02': [],  'disp_noc_0': [], 'velodyne_points':[], 'velodyne_points32':[], 'velodyne_points16':[], 'velodyne_points8':[], 'velodyne_points1':[], 'velodyne_points0':[] }
    right_data_path = {'image_03': [],  'disp_noc_1': [], 'velodyne_points_right':[], 'velodyne_points32_right':[], 'velodyne_points16_right':[], 'velodyne_points8_right':[], 'velodyne_points1_right':[], 'velodyne_points0_right':[]}

    for dir_name in left:
        file_name_list = sorted(os.listdir(os.path.join(root_dir, dir_name)))
        for file_name in file_name_list:
            path = os.path.join(root_dir, dir_name, file_name)
            left_data_path[dir_name].append(path)


    for dir_name in right:
        file_name_list = sorted(os.listdir(os.path.join(root_dir, dir_name)))
        for file_name in file_name_list:
            path = os.path.join(root_dir, dir_name, file_name)
            right_data_path[dir_name].append(path)

    return left_data_path, right_data_path
