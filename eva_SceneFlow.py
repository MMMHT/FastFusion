import torch.autograd.profiler as profiler

import cv2
import torch
import numpy as np
import data_loader.Kitti141Loader as DL141
import data_loader.SceneFlow_loader as SFLodar
from layers.PostProcessing import PostProcessing
from layers.SemiGlobalMatching import SemiGlobalMatching
from model import fbnn_guided
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
        model.load_state_dict(torch.load('/home/haitao/projects/stereobit-pytorch/workspace/RF_Sceneflow_PreTrain2/ckpt/2015_fbnn_70.pth'))
    model.cuda()
    print(model)
    model.eval()

    SMmodel = fbnn_guided.StereoLidarFusion(False)
    SMmodel.cuda()
    SMmodel.load_state_dict(torch.load(cfg.checkpoint))
    SMmodel.eval()
    SMmodel.val_mode()
    print("validation!")

    test_loader = DataLoader(SFLodar.SceneFlowLoader('/data3/dataset/SceneFlow'), batch_size=cfg.batchsize,
                              shuffle=True, num_workers=cfg.threads)


    print('start testing RFmodel!')

    meters_unary = metric.Metrics(cfg.train_metric_field)
    avg_meters_unary = metric.MovingAverageEstimator(cfg.train_metric_field)
    meters = metric.Metrics(cfg.train_metric_field)
    avg_meters = metric.MovingAverageEstimator(cfg.train_metric_field)

    # print(filenames)
    def read_img(path):
        """ Read raw img and DO NOT perform any process to the image
            Keep the image value in the range of (0-255) """
        img = cv2.imread(path)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)[0:1,:,:]
        return img


    with torch.no_grad():
        # for i in [ 'left_slidar' ,'', '0', '1', '8', '16', '32']:
        for i in ['16']:
            for idx, data in enumerate(tqdm(test_loader)):

                for k in data.keys():
                    data[k] = data[k].cuda()

                inputs = dict()
                inputs['left_img'] = data['left_img']
                inputs['right_img'] = data['right_img']
                inputs['left_slidar'] = data['left_slidar']
                inputs['right_slidar'] = data['right_slidar']
                inputs['left_disp'] = data['left_disp']
                inputs['right_disp'] = data['right_disp']
                inputs['pred'] = data['stereobit_pre']

                pred_np = inputs['pred'].squeeze(0).data.cpu().numpy()
                target_np = inputs['left_disp'].squeeze(0).data.cpu().numpy()
                results = meters_unary.compute(pred_np, target_np)
                avg_meters_unary.update(results)

                inp = torch.cat((inputs['pred'], inputs['left_img'] , inputs['left_slidar']), 1)

                # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                #     with profiler.record_function("model_inference"):
                #
                pred = model(inp)
                #
                # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                save_path = 'RF_Sceneflow_PreTrain2'
                input_name_left = ''

                if not os.path.exists(os.path.join('workspace', save_path, 'predictions/eva', input_name_left)):
                    os.makedirs(os.path.join('workspace', save_path, 'predictions/eva', input_name_left),0o777)

                save_png(visualize_with_color(inputs['left_disp']), os.path.join('workspace', save_path, 'predictions/eva', input_name_left) + '/' + str(idx)+ 'gt')
                save_png(visualize_correct_pred(inputs['left_img'], pred, inputs['left_disp']), os.path.join('workspace', save_path, 'predictions/eva', input_name_left) + '/' + str(idx) + 'Err')
                save_png(visualize_with_color(inputs['pred']), os.path.join('workspace', save_path, 'predictions/eva', input_name_left) + '/' + str(idx) + 'pred')
                save_png(visualize_with_color(inputs['left_slidar']), os.path.join('workspace', save_path, 'predictions/eva', input_name_left) + '/' + str(idx) + 'slidar')
                save_png(visualize_with_color(pred), os.path.join('workspace', save_path, 'predictions/eva', input_name_left) + '/' + str(idx) + 'pred_disp')


                pred_np = pred.squeeze(0).data.cpu().numpy()
                target_np = (inputs['left_disp']).squeeze(0).data.cpu().numpy()
                results = meters.compute(pred_np, target_np)
                avg_meters.update(results)

            avg_results = avg_meters.compute()
            avg_results_unary = avg_meters_unary.compute()

            for key, val in avg_results.items():
                print('velodyne {}:{}: {:5.5f} '.format(i, key, val), end='')
                # logger.write('velodyne:{} {}: {:5.3f} '.format(i, key, val), end='')
                # writer.add_scalar("velodyne {}".format(key), val, iter)

            for key, val in avg_results_unary.items():
                print('Unary velodyne:{} {}: {:5.5f} '.format(i, key, val), end='')

            avg_meters.reset()
            avg_meters_unary.reset()
            print('\n')

if __name__ == '__main__':
    eval()
