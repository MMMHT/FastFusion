import torch
import numpy as np
import os
import data_loader.RF_Loader as DLRF
from torch.utils.data import DataLoader
import data_loader.Kitti141Loader as DL141
import data_loader.SceneFlow_loader as DLSF
from tqdm import tqdm
from PIL import Image
from torch import optim
from utils import *
import matplotlib.pyplot as plt
from layers import UNet
cfg = options.get_config()
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from layers import SobelGrad,Sobel,Laplace
import cv2
import torch.autograd.profiler as profiler

torch.cuda.set_device(cfg.gpu)
torch.set_printoptions(profile="full")

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

train_loader = DataLoader(DLSF.SceneFlowLoader('/media/SceneFlow'), batch_size=cfg.batchsize,
                          shuffle=True, num_workers=cfg.threads)

test_loader = DataLoader(DL141.Kitti141Loader(cfg.datapath, (352, 1216)), batch_size=1, shuffle=False,num_workers=cfg.threads)

print(cfg.workspace)
workspace = util.setup_workspace(cfg.workspace, "RFNet")
logger = util.Logger(os.path.join(workspace.log, 'train_log.txt'))
logger.write('Workspace: {}'.format(cfg.workspace), 'green')
writer = SummaryWriter(workspace.root)

model = UNet(n_channels=3, n_classes=1, bilinear=True)

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 12, gamma=0.92)
trans = transforms.ToPILImage()

cos = nn.CosineSimilarity(dim=1, eps=0)
get_gradient = Sobel().cuda()
get_laplace = Laplace().cuda()


def get_loss(pred, img, groundtruth, input,  a = 0.5):

    ones = torch.ones(groundtruth.size(0), 1, groundtruth.size(2),groundtruth.size(3)).float().cuda()
    ones = torch.autograd.Variable(ones)

    depth_grad = get_gradient(groundtruth)
    output_grad = get_gradient(pred)
    img_grad = get_gradient(img)
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(groundtruth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(groundtruth)
    output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(groundtruth)
    output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(groundtruth)

    img_grad_dx = img_grad[:, 0, :, :].contiguous().view_as(groundtruth)
    img_grad_dy = img_grad[:, 1, :, :].contiguous().view_as(groundtruth)

    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

    loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 1).sum().div(img.nelement())
    loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 1).sum().div(img.nelement())

    ones2 = torch.ones(depth_normal.shape).cuda()
    ones2 = torch.autograd.Variable(ones2)
    loss_norm  = (ones2 - cos(depth_normal , output_normal)).sum().div(depth_normal.nelement())
    # loss_norm = loss_norm.sum().div(img.nelement())

    pred_lapace = get_laplace(pred)
    pred_lapace_x = pred_lapace[:,0,:,:].contiguous().view_as(pred)
    pred_lapace_y = pred_lapace[:,1,:,:].contiguous().view_as(pred)
    groundtruth_lapace = get_laplace(groundtruth)
    groundtruth_lapace_x = groundtruth_lapace[:,0,:,:].contiguous().view_as(pred)
    groundtruth_lapace_y = groundtruth_lapace[:,1,:,:].contiguous().view_as(pred)


    loss_photometric = torch.abs(pred_lapace_x) * torch.exp(-a * torch.abs(img_grad_dx)) + torch.abs(pred_lapace_y) * torch.exp(- a * torch.abs(img_grad_dy))
    loss_photometric = loss_photometric.sum().div(img.nelement())

    val_mask = groundtruth > 0
    loss_l1 = F.smooth_l1_loss(pred[val_mask], groundtruth[val_mask], reduction='mean')

    loss_l2 = (input-pred) * (input-pred)
    loss_l2 = torch.sqrt_(loss_l2.sum()).div(pred.nelement())

    loss = loss_dx + loss_dy + loss_norm + loss_photometric +  loss_l2 + loss_l1

    return  loss



def train():
    for epoch in range(cfg.epoch):
        model.train()

    logger.write('Model: {}'.format(model), 'green')
    logger.write('Optimizer: {}'.format(optim), 'green')
    all_loss = []


    for epoch in range(cfg.epoch):
        print('This is %d-th epoch' % (epoch))
        losses = []
        for idx, data in enumerate(tqdm(train_loader)):
            for k in data.keys():
                data[k] = data[k].cuda()

            inputs = dict()
            inputs['left_img'] = data['left_img']
            inputs['right_img'] = data['right_img']
            inputs['left_slidar'] = data['left_slidar']
            inputs['right_slidar'] = data['right_slidar']
            inputs['left_disp'] = data['left_disp']
            inputs['stereobit_pred'] = data['stereobit_pre']

            optimizer.zero_grad()

            inp = torch.cat((inputs['stereobit_pred'], inputs['left_img'],  inputs['left_slidar']), 1)
            pred = model(inp)
            loss = get_loss(pred, inputs['left_img'], inputs['left_disp'], inputs['stereobit_pred'])
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        epoch_scheduler.step()
        all_loss.append(np.array(losses).mean())
        logger.write("epoches: %5d,  loss: %4f" % (
            epoch, np.array(losses).mean()))
        print('epoch %d total training loss = %.3f' % (epoch, all_loss[epoch].mean()))
        writer.add_scalar('train_loss', loss.item(), epoch)

        if epoch%cfg.save_epoch ==0:
            torch.save(model.state_dict(), "{}/{}_{}_{}.pth".format(workspace.ckpt ,cfg.dataset, cfg.network, str(epoch)))
            logger.write('Save checkpoint to {}/{}_{}_{}.pth'.format(workspace.ckpt, cfg.dataset, cfg.network, str(epoch)), 'magenta')




if __name__ == '__main__':
    train()
