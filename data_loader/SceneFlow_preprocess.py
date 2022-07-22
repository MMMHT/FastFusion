import os.path
import sys
import matplotlib.pyplot as plt
import cv2
import readpfm as rp
from model import fbnn_guided,SemiGlobalMatching, PostProcessing
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
# import utils
# from utils import *
from utils import options, save_png
cfg = options.get_config()

SMmodel = fbnn_guided.StereoLidarFusion(False)
SMmodel.cuda()
SMmodel.load_state_dict(torch.load('../workspace/lidarStereo_gray_disp128_lr2e-4_pow0.2/ckpt/2015_fbnn.pth'))


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    img = Image.open(path).convert('RGB')
    img = np.asarray(img)
    img = torch.from_numpy(img)
    img = img.permute(2,0,1)
    img = transforms.Grayscale()(img)
    return img.float()

def disparity_loader(path):
    return rp.readPFM(path)

def random_sampler(left_slidar, sample_percentage):
    # perform random sampling on the left disparity
    left_slidar = torch.from_numpy(left_slidar.copy()).unsqueeze(0)
    sample_mask = torch.rand(left_slidar.shape).le(sample_percentage)
    valid_mask = left_slidar > 0
    sample_mask = sample_mask * valid_mask
    left_slidar =left_slidar * sample_mask

    # perform random sampling on the right disparity
    right_slidar = torch.zeros_like(left_slidar)
    for i in range(left_slidar.shape[1]):
        for j in range(left_slidar.shape[2]):
            if sample_mask[0, i, j] == 1:
                val = left_slidar[0][i][j]
                val = val.round().item()
                if j - val > 1 and i > 1 and i < left_slidar.shape[2]:
                    right_slidar[0][i][int(j - val)] = val
                    # print(right_slidar[0][i][int(j - val)])
    # valid_mask = left_slidar > 0
    # sample_mask_r = sample_mask_r * valid_mask
    # right_slidar = left_slidar * sample_mask_r
    return left_slidar, right_slidar

def get_pred(left_path, right_path, disp_path, save_dir, save_sub_dir, filename):
    disp, scale = disparity_loader(disp_path)
    disp = np.ascontiguousarray(disp,dtype=np.float32)
    left_slidar, right_slidar = random_sampler(disp, 0.05)
    # TODO left_Slidar did not round the value while right_slidar did

    inputs = dict()
    left  =  default_loader(left_path)
    right = default_loader(right_path)
    inputs['left_img']  = left.unsqueeze(0)
    inputs['right_img'] = right.unsqueeze(0)
    inputs['left_disp'] = torch.from_numpy(disp.copy()).unsqueeze(0)
    inputs['left_slidar'] = left_slidar.unsqueeze(0)
    inputs['right_slidar'] = right_slidar.unsqueeze(0)

    for k in inputs.keys():
        inputs[k] = inputs[k].cuda()

    with torch.no_grad():
        output_L, output_R = SMmodel(inputs, 0)
        sgm = SemiGlobalMatching()
        postprocess = PostProcessing()
        disp_L, disp_R = sgm(output_L, output_R, inputs)
        disp_L = postprocess(disp_L, disp_R)

    left_slidar, right_slidar =  random_sampler(disp, 0.05)
    return left, right, left_slidar, right_slidar, torch.from_numpy(disp.copy()),  disp_L,


def preprocess_SceneFlow(filepath, prepath):

    monkaa_path = os.path.join(filepath, 'Monkaa', 'frames_finalpass')
    monkaa_disp = os.path.join(filepath, 'Monkaa', 'disparity')

    monkaa_dir  = os.listdir(monkaa_path)

    idx = 0
    count = 0
    thre = 60

    left_all = np.zeros((670,540,960,4),dtype=np.uint16)
    right_all = np.zeros((670,540,960,4),dtype=np.uint16)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Monkaa~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):


            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                left_path = monkaa_path+'/'+dd+'/left/'+im
                disp_path = monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm'

            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                right_path = monkaa_path+'/'+dd+'/right/'+im

            # print(disp_path)

            disp, scale = disparity_loader(disp_path)
            disp = np.ascontiguousarray(disp,dtype=np.float32)
            disp = torch.from_numpy(disp)
            # print(disp.max())

            if count != thre :
                count = count + 1
                # print('skip')
                continue
            else:
                count = 0

            if disp.max()> cfg.disp_max:
                print("{} exceed max disparity".format(disp.max()))
                continue

            left, right, left_slidar, right_slidar, disp, pred\
                = get_pred(left_path, right_path, disp_path, os.path.join(prepath, 'Monkaa', 'frames_finalpass'), dd, im )

            left_all[idx, :, :, 0] = left.cpu().numpy()
            left_all[idx, :, :, 1] = disp.cpu().numpy()
            left_all[idx, :, :, 2] = left_slidar.cpu().numpy()
            left_all[idx, :, :, 3] = pred.cpu().numpy()

            right_all[idx, :, :, 0] = right.cpu().numpy()
            right_all[idx, :, :, 1] = disp.cpu().numpy()
            right_all[idx, :, :, 2] = right_slidar.cpu().numpy()
            right_all[idx, :, :, 3] = pred.cpu().numpy()

            idx = idx + 1
            print(idx)


    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FlyingThings3D~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    flying_path = os.path.join(filepath, 'FlyingThings3D', 'frames_finalpass')
    flying_disp = os.path.join(filepath, 'FlyingThings3D', 'disparity')
    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A','B','C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:

                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    left_path = flying_dir+ss+'/'+ff+'/left/'+im

                disp_path = flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm'

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    right_path = flying_dir+ss+'/'+ff+'/right/'+im


                # print(disp_path)
                disp, scale = disparity_loader(disp_path)
                disp = np.ascontiguousarray(disp,dtype=np.float32)
                disp = torch.from_numpy(disp)
                # print(disp.max())


                if count != thre :
                    count = count + 1
                    # print('skip')
                    continue
                else:
                    count = 0

                if disp.max()> cfg.disp_max:
                    print("{} exceed max disparity".format(disp.max()))
                    continue

                left, right, left_slidar, right_slidar, disp, pred \
                    = get_pred(left_path, right_path, disp_path, os.path.join(prepath, 'FlyingThings3D',  'frames_finalpass','TRAIN' ),
                         ss+'/'+ff, im )


                left_all[idx, :, :, 0] = left.cpu().numpy()
                left_all[idx, :, :, 1] = disp.unsqueeze(0).cpu().numpy()
                left_all[idx, :, :, 2] = left_slidar.cpu().numpy()
                left_all[idx, :, :, 3] = pred.cpu().numpy()

                right_all[idx, :, :, 0] = right.cpu().numpy()
                right_all[idx, :, :, 1] = disp.unsqueeze(0).cpu().numpy()
                right_all[idx, :, :, 2] = right_slidar.cpu().numpy()
                right_all[idx, :, :, 3] = pred.cpu().numpy()

                idx = idx + 1
                print(idx)


    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Driving~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    thre = 20
    driving_dir = os.path.join(filepath, 'Driving', 'frames_finalpass')
    driving_disp = os.path.join(filepath, 'Driving', 'disparity')

    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']
    subdir3 = ['fast','slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir+ '/' +i+'/'+j+'/'+k+'/left/')
                for im in imm_l:


                    if is_image_file(driving_dir+ '/' +i+'/'+j+'/'+k+'/left/'+im):
                       left_path = driving_dir+ '/' +i+'/'+j+'/'+k+'/left/'+im
                    disp_path = driving_disp+'/' +i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm'
                    if is_image_file(driving_dir+ '/' +i +'/'+j+'/'+k+'/right/'+im):
                        right_path =  driving_dir+ '/' +i+'/'+j+'/'+k+'/right/'+im

                    # print(disp_path)
                    disp, scale = disparity_loader(disp_path)
                    disp = np.ascontiguousarray(disp,dtype=np.float32)
                    disp = torch.from_numpy(disp)
                    # print(disp.max())

                    if count != thre :
                        count = count + 1
                        # print('skip')
                        continue
                    else:
                        count = 0

                    if disp.max()> cfg.disp_max:
                        print("{} exceed max disparity".format(disp.max()))
                        continue

                    left, right, left_slidar, right_slidar, disp, pred \
                        = get_pred(left_path, right_path, disp_path, os.path.join(prepath, 'Driving', 'frames_finalpass'),
                             os.path.join(i, j, k), im )

                    left_all[idx, :, :, 0] = left.cpu().numpy()
                    left_all[idx, :, :, 1] = disp.cpu().numpy()
                    left_all[idx, :, :, 2] = left_slidar.cpu().numpy()
                    left_all[idx, :, :, 3] = pred.cpu().numpy()

                    right_all[idx, :, :, 0] = right.cpu().numpy()
                    right_all[idx, :, :, 1] = disp.cpu().numpy()
                    right_all[idx, :, :, 2] = right_slidar.cpu().numpy()
                    right_all[idx, :, :, 3] = pred.cpu().numpy()

                    idx = idx + 1
                    print(idx)

    np.save(os.path.join(prepath, 'left.npy'),left_all)
    np.save(os.path.join(prepath, 'right.npy'),right_all)



if __name__ == '__main__':

    preprocess_SceneFlow('/media/SceneFlow', "/media")