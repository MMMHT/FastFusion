import subprocess

import torch.autograd.profiler as profiler

import cv2
import torch
import numpy as np
import data_loader.DrivingStereoLoader as drivingstereo
from torch.utils.data import DataLoader
from utils import options, save_png
from model import fbnn
from model import HingeLoss
import matplotlib.pyplot as plt
from layers import UNet
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToPILImage
from utils import *
import data_loader.KittiLidarStereoLoader as DLLS
from stereobit import WritePFM
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
    # model = BinaryUNet(n_channels=3, n_classes=1, bilinear=True)
    if cfg.cuda:
        model.load_state_dict(torch.load('/home/haitao/projects/stereobit-pytorch/workspace/RFslim2_Sceneflow_selected_finetune/ckpt/2015_fbnn_30.pth', map_location='cuda:0'))
        # model.load_state_dict(torch.load('/home/haitao/projects/stereobit-pytorch/workspace/RF_Sceneflow_finetune_selected/ckpt/2015_fbnn_70.pth'))
        # RFslim2_Sceneflow_selected_finetune/ckpt/2015_fbnn_30.pth
    model.half()
    model.cuda()
    print(model)
    model.eval()


    test_loader = DataLoader(drivingstereo.DrivingStereoLoader(cfg.datapath, 'testing'), batch_size = 1 , shuffle=False, num_workers=1 )

    print('start testing RFmodel!')

    meters_unary = metric.Metrics(cfg.train_metric_field)
    avg_meters_unary = metric.MovingAverageEstimator(cfg.train_metric_field)
    meters = metric.Metrics(cfg.train_metric_field)
    avg_meters = metric.MovingAverageEstimator(cfg.train_metric_field)

    with torch.no_grad():
        # for scene in ['cloudy', 'sunny', 'foggy', 'rainy']:
        for scene in ['testing']:
            for idx, data in enumerate(tqdm(test_loader)):
                for k in data.keys():
                    for j in data[k].keys():
                        data[k][j] = data[k][j].cuda().half()

                inputs = dict()
                inputs['left_img'] = data[scene]['img_left']
                inputs['right_img'] = data[scene]['img_right']
                inputs['left_slidar'] = data[scene]['slidar_left']
                inputs['right_slidar'] = data[scene]['slidar_right']
                inputs['left_disp'] = data[scene]['disp_left']
                inputs['pred'] = data[scene]['pred_left']

                # print(inputs['left_disp'].shape)

                pred_np = inputs['pred'].squeeze(0).data.cpu().numpy()
                target_np = inputs['left_disp'].squeeze(0).data.cpu().numpy()
                results = meters_unary.compute(pred_np, target_np)
                avg_meters_unary.update(results)

                inp = torch.cat((inputs['pred'], inputs['left_img'], inputs['left_slidar']), 1)

                with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                    with profiler.record_function("model_inference"):
                        #
                        pred = model(inp)
                #
                # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                save_path = 'RFslim2_Sceneflow_selected_finetune'

                if not os.path.exists(os.path.join('workspace', save_path, 'predictions/eva', scene)):
                    os.makedirs(os.path.join('workspace', save_path, 'predictions/eva', scene),0o777)

                io.imsave(os.path.join('workspace', save_path, 'predictions/eva', scene) + '/' + str(idx) + 'pred_disp_gray.png',
                          (pred.squeeze(0).squeeze(0).cpu().numpy() * 256).astype('uint16'))
                save_png(visualize_with_color(inputs['left_disp'].float()), os.path.join('workspace', save_path, 'predictions/eva', scene) + '/' + str(idx)+ 'gt')
                save_png(visualize_correct_pred(inputs['left_img'].float(), pred, inputs['left_disp']), os.path.join('workspace', save_path, 'predictions/eva', scene) + '/' + str(idx) + 'Err')
                save_png(visualize_with_color(inputs['pred'].float()), os.path.join('workspace', save_path, 'predictions/eva', scene) + '/' + str(idx) + 'pred')
                save_png(visualize_with_color(inputs['left_slidar'].float()), os.path.join('workspace', save_path, 'predictions/eva', scene) + '/' + str(idx) + 'slidar')
                save_png(visualize_with_color(pred.float()), os.path.join('workspace', save_path, 'predictions/eva', scene) + '/' + str(idx) + 'pred_disp')
                # save_png(pred,os.path.join('workspace', save_path, 'predictions/eva', scene, str(idx)+'pred'))

                pred_np = pred.squeeze(0).data.cpu().numpy()
                target_np = (inputs['left_disp']).squeeze(0).data.cpu().numpy()
                results = meters.compute(pred_np, target_np)
                avg_meters.update(results)


            avg_results = avg_meters.compute()
            avg_results_unary = avg_meters_unary.compute()

            for key, val in avg_results.items():
                print('{}:{}: {:5.5f} '.format(scene, key, val), end='')
                # logger.write('velodyne:{} {}: {:5.3f} '.format(i, key, val), end='')
                # writer.add_scalar("velodyne {}".format(key), val, iter)

            print('\n')
            for key, val in avg_results_unary.items():
                print('{} {}: {:5.5f} '.format(scene, key, val), end='')


            avg_meters.reset()
            avg_meters_unary.reset()
            print('\n')

if __name__ == '__main__':
    eval()
