import torch
import numpy as np
import os
import sys
sys.path.append('/data/haitao/codes/stereobit-pytorch')
from torch.utils.data.dataset import Dataset
import data_loader.KittiLidarStereoLoader as DLLS
from torch.utils.data import DataLoader
from model import *
from tqdm import tqdm
from utils import *
from layers.SemiGlobalMatching import *
from layers.PostProcessing import *
from skimage import io
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter

cfg=options.get_config()

torch.cuda.set_device(cfg.gpu)

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

# fix random seeds for reproducibility
cfg=options.get_config()
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

RFkitti2017 = {
    'num':277,
    'width':1216,
    'height':352,
    'data_channels':6,
}

def read_img(path):
    """ Read raw img and DO NOT perform any process to the image
        Keep the image value in the range of (0-255) """

    img = cv2.imread(path)
    img = torch.from_numpy(img)
    img = img.permute(2,0,1)[0:1,:,:]

    return img

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
    return depth[np.newaxis, :, :]
#
def read_img2(path, channels=1):
    """ Read raw img and DO NOT perform any process to the image
        Keep the image value in the range of (0-255) """
    img = io.imread(path)
    img = torch.from_numpy(img)
    # if channels == 1:
    #     img = img.permute(2,0,1)
    #     img = transforms.Grayscale()(img)
    return img


def get_sellected2017_datapath(root_dir):

    """ Read path to all data from converted KITTI Depth Completion dataset """
    left_data_path = {'img': [],  'disp': [], 'slidar':[], 'stereobit_pred':[]}
    right_data_path = {'img': [],  'disp': [], 'slidar':[]}


    # Directory of RGB images
    rgb_left_dir = os.path.join(root_dir ,'image_02')
    rgb_right_dir = os.path.join(root_dir,'image_03')
    # Directory of disp maps
    depth_left_dir = os.path.join(root_dir, 'disp_02')
    depth_right_dir = os.path.join(root_dir, 'disp_03')
    # Directory of slidar maps
    slidar_left_dir = os.path.join(root_dir, 'slidar_02')
    slidar_right_dir = os.path.join(root_dir, 'slidar_03')
    # Directory of pred maps
    pred_left_dir = os.path.join(root_dir, 'stereobit_pred')

    # Get image names (DO NOT obtain from raw data directory since the annotated data is pruned)
    file_name_list = sorted(os.listdir(depth_left_dir))

    for file_name in file_name_list:

        # Path to RGB images
        rgb_left_path = os.path.join(rgb_left_dir, file_name)
        rgb_right_path = os.path.join(rgb_right_dir, file_name)
        # Path to ground truth depth maps
        depth_left_path = os.path.join(depth_left_dir, file_name)
        depth_right_path = os.path.join(depth_right_dir, file_name)
        # Path to sparse depth maps
        slidar_left_path = os.path.join(slidar_left_dir, file_name)
        slidar_right_path = os.path.join(slidar_right_dir, file_name)
        # Path to stereobit prediction
        stereobit_left_path = os.path.join(pred_left_dir, file_name)

        # Add to list
        left_data_path['img'].append(rgb_left_path)
        left_data_path['disp'].append(depth_left_path)
        left_data_path['slidar'].append(slidar_left_path)
        left_data_path['stereobit_pred'].append(stereobit_left_path)

        right_data_path['img'].append(rgb_right_path)
        right_data_path['disp'].append(depth_right_path)
        right_data_path['slidar'].append(slidar_right_path)

    return left_data_path, right_data_path


def make_dataset( root_dir):

    print("RF Data Preporcess!")

    # pre_left = np.load(os.path.join(root_dir,'left.npy'))
    # pre_right = np.load(os.path.join(root_dir,'right.npy'))

    left_data_path, right_data_path = get_sellected2017_datapath(root_dir)

    left_all = np.zeros((RFkitti2017['num'],RFkitti2017['height'],RFkitti2017['width'],RFkitti2017['data_channels']),dtype=np.uint8)
    right_all = np.zeros((RFkitti2017['num'],RFkitti2017['height'],RFkitti2017['width'],RFkitti2017['data_channels']),dtype=np.uint8)

    # pre_left:   0 dim: img
    #             1 dim: gt disparity
    #             2 dim: negativate gt disparity
    #             3 dim: sparse lidar input
    #             4 dim: inference by trained model(training set of RFNet)
    #             5 dim : inference by LEANet model (kitti2015)
    for idx in tqdm(range(len(left_data_path['img']))):
        #read image
        left_img = np.array(read_img(left_data_path['img'][idx]))
        right_img = np.array(read_img(right_data_path['img'][idx]))
        left_img = left_img[:,0:RFkitti2017['height'],0:RFkitti2017['width']]
        right_img = right_img[:,0:RFkitti2017['height'],0:RFkitti2017['width']]
        left_all[idx,:,:,0] = left_img
        right_all[idx,:,:,0] = right_img


        #read ground truth disparity
        left_disp = np.array(read_depth(left_data_path['disp'][idx]))
        right_disp = np.array(read_depth(right_data_path['disp'][idx]))

        left_disp = left_disp[:,0:RFkitti2017['height'],0:RFkitti2017['width']]
        right_disp = right_disp[:,0:RFkitti2017['height'],0:RFkitti2017['width']]

        left_all[idx,:,:,1] = left_disp
        right_all[idx,:,:,1] = right_disp

        #read slidar
        left_slidar = np.array(read_img2(left_data_path['slidar'][idx]))
        right_slidar = np.array(read_img2(right_data_path['slidar'][idx]))
        # print(left_slidar[150:300,800])
        left_slidar = left_slidar[0:RFkitti2017['height'],0:RFkitti2017['width']]
        right_slidar = right_slidar[0:RFkitti2017['height'],0:RFkitti2017['width']]
        left_all[idx,:,:,3] = left_slidar
        right_all[idx,:,:,3] = right_slidar

        disp_path = '/data/kitti2017/selected/stereobit_pred'
        disp_path = os.path.join(disp_path,str(idx)+'.png')
        disp_L = read_img(disp_path)
        disp_L = disp_L[:,0:RFkitti2017['height'],0:RFkitti2017['width']]
        left_all[idx,:,:,4] = disp_L


        path2Lea = '/data/kitti2017/selected/LEANet'
        leaDisp = read_img (os.path.join(path2Lea, str(idx)+'.png'))
        left_all[idx,:,:,5] = leaDisp[:,0:RFkitti2017['height'], 0:RFkitti2017['width']]


    left_all = torch.from_numpy(left_all).float()
    right_all = torch.from_numpy(right_all).float()

    np.save(os.path.join(root_dir , 'selected_left.npy'),left_all.numpy())
    np.save(os.path.join(root_dir , 'selected_right.npy'),right_all.numpy())


if __name__ == '__main__':

    make_dataset('/data/kitti2017/selected')

