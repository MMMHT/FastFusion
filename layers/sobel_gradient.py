# Helpful Tips: https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys

np.set_printoptions(sys.maxsize)
from math import exp
# import cv2
# import matplotlib.pyplot as plt


def sobel(window_size):
    assert (window_size % 2 != 0)
    ind = window_size // 2
    matx = []
    maty = []
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gx_ij = 0
            else:
                gx_ij = i / float(i * i + j * j)
            row.append(gx_ij)
        matx.append(row)
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gy_ij = 0
            else:
                gy_ij = j / float(i * i + j * j)
            row.append(gy_ij)
        maty.append(row)

    # matx=[[-3, 0,+3],
    # 	  [-10, 0 ,+10],
    # 	  [-3, 0,+3]]
    # maty=[[-3, -10,-3],
    # 	  [0, 0 ,0],
    # 	  [3, 10,3]]
    if window_size == 3:
        mult = 2
    elif window_size == 5:
        mult = 20
    elif window_size == 7:
        mult = 780

    matx = np.array(matx) * mult
    maty = np.array(maty) * mult

    return torch.Tensor(matx), torch.Tensor(maty)


def create_window(window_size, channel):
    windowx, windowy = sobel(window_size)
    windowx, windowy = windowx.unsqueeze(0).unsqueeze(0), windowy.unsqueeze(0).unsqueeze(0)
    windowx = torch.Tensor(windowx.expand(channel, 1, window_size, window_size))
    windowy = torch.Tensor(windowy.expand(channel, 1, window_size, window_size))
    # print windowx
    # print windowy

    return windowx, windowy


def gradient(img, windowx, windowy, window_size, padding, channel):
    if channel > 1:  # do convolutions on each channel separately and then concatenate
        gradx = torch.ones(img.shape)
        grady = torch.ones(img.shape)
        for i in range(channel):
            gradx[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(0), windowx, padding=padding, groups=1).squeeze(
                0)  # fix the padding according to the kernel size
            grady[:, i, :, :] = F.conv2d(img[:, i, :, :].unsqueeze(0), windowy, padding=padding, groups=1).squeeze(0)

    else:
        gradx = F.conv2d(img, windowx, padding=padding, groups=1)
        grady = F.conv2d(img, windowy, padding=padding, groups=1)

    return gradx, grady


class SobelGrad(torch.nn.Module):
    def __init__(self, window_size=3, padding=1):
        super(SobelGrad, self).__init__()
        self.window_size = window_size
        self.padding = padding
        self.channel = 1  # out channel
        self.windowx, self.windowy = create_window(window_size, self.channel)

    def forward(self, pred):
        (batch_size, channel, _, _) = pred.size()
        if pred.is_cuda:
            self.windowx = self.windowx.cuda(pred.get_device())
            self.windowx = self.windowx.type_as(pred)
            self.windowy = self.windowy.cuda(pred.get_device())
            self.windowy = self.windowy.type_as(pred)

        pred_gradx, pred_grady = gradient(pred, self.windowx, self.windowy, self.window_size, self.padding, channel)
        # label_gradx, label_grady = gradient(label, self.windowx, self.windowy, self.window_size, self.padding, channel)

        # return pred_gradx, pred_grady, label_gradx, label_grady
        return pred_gradx, pred_grady

# # For testing
# if __name__ == '__main__':
# 	img1_path="1_1_2-cp_Page_0654-XKI0001.exr" # image 1
# 	img2_path="1_1_1-tc_Page_065-YGB0001.exr"  # image 2
# 	img1=cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# 	# print(img1.shape)
# 	img2=cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# 	# OpenCV sobel gradient for to check correctness
# 	sobelx1 = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=5)
# 	sobely1 = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
# 	sobelx2 = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
# 	sobely2 = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=5)

# 	img1=np.array(img1,dtype=np.float).transpose(2,0,1)
# 	img2=np.array(img2,dtype=np.float).transpose(2,0,1)
# 	img1=torch.from_numpy(img1).float().unsqueeze(0)
# 	img2=torch.from_numpy(img2).float().unsqueeze(0)
# 	sgrad=SobelGrad(window_size=5,padding=2)

# 	pred_gradx,pred_grady,label_gradx,label_grady=sgrad(img1,img2)
# 	gradx1=np.array(pred_gradx[0]).transpose(1,2,0)
# 	grady1=np.array(pred_grady[0]).transpose(1,2,0)
# 	gradx2=np.array(label_gradx[0]).transpose(1,2,0)
# 	grady2=np.array(label_grady[0]).transpose(1,2,0)

# 	f, axarr = plt.subplots(2, 4)
# 	axarr[0][0].imshow(sobelx1)
# 	axarr[0][1].imshow(sobely1)
# 	axarr[0][2].imshow(sobelx2)
# 	axarr[0][3].imshow(sobely2)
# 	axarr[1][0].imshow(gradx1)
# 	axarr[1][1].imshow(grady1)
# 	axarr[1][2].imshow(gradx2)
# 	axarr[1][3].imshow(grady2)
# 	plt.show()



class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out

class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
        edge_ky = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out
