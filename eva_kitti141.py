import torch.autograd.profiler as profiler

import cv2
import torch
import numpy as np
import data_loader.Kitti141Loader as DL141
from torch.utils.data import DataLoader
from utils import options, save_png
import matplotlib.pyplot as plt
from layers import UNet
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToPILImage
from utils import *
from skimage import io


torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

# fix random seeds for reproducibility
cfg=options.get_config()
torch.cuda.set_device(cfg.gpu)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def eval():
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    if cfg.cuda:
        model.load_state_dict(torch.load('model/Sceneflow200_Kitti40.pth', map_location='cuda:0'))
    model.cuda()
    model.half()
    model.eval()

    test_loader = DataLoader(DL141.Kitti141Loader(cfg.datapath), batch_size=1, shuffle=False,num_workers=cfg.threads)

    print('start testing RFmodel!')

    meters = metric.Metrics(cfg.train_metric_field)
    avg_meters = metric.MovingAverageEstimator(cfg.train_metric_field)

    def read_img(path):
        """ Read raw img and DO NOT perform any process to the image
            Keep the image value in the range of (0-255) """
        img = cv2.imread(path)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)[0:1,:,:]
        return img


    with torch.no_grad():
        for i in [ 'velodyne_points0', 'velodyne_points1','velodyne_points8',
                   'velodyne_points16', 'velodyne_points32', 'velodyne_points64']:
            for idx, data in enumerate(tqdm(test_loader)):

                for k in data.keys():
                    data[k] = data[k].cuda().half()

                inputs = dict()
                inputs['left_img'] = data['left_img'].half()
                inputs['right_img'] = data['right_img'].half()
                inputs['left_slidar'] = data[i].half()
                inputs['left_disp'] = data['left_disp'][:,:1,:,:].half()  # we dont want negative samples
                inputs['right_disp'] = data['right_disp'][:,:1,:,:].half()
                # inputs['pred'] = data['pred']

                # read coarse dpeth estimation
                filenames = os.listdir(os.path.join(cfg.datapath, 'pred', i))
                filenames.sort()
                disp_L =read_img(os.path.join(cfg.datapath, 'pred', i, filenames[idx]))
                disp_L = disp_L[:,disp_L.shape[1] - 370: disp_L.shape[1], 0:1242]
                disp_L = disp_L.float().cuda().unsqueeze(0).half()

                inp = torch.cat((disp_L, inputs['left_img'], inputs['left_slidar']), 1) # no slidar input

                with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                    with profiler.record_function("model_inference"):
                        pred = model(inp)
                # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                pred_np = pred.squeeze(0).data.cpu().numpy()
                target_np = (inputs['left_disp']).squeeze(0).data.cpu().numpy()
                results = meters.compute(pred_np, target_np)
                avg_meters.update(results)

            avg_results = avg_meters.compute()

            print("Real LiDAR: {} ".format(i))

            for key, val in avg_results.items():
                print('{}: {:5.5f} \n'.format(key, val), end='')

            avg_meters.reset()
            print('\n')






if __name__ == '__main__':
    eval()

