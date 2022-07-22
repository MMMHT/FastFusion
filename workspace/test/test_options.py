from easydict import EasyDict
import torch
import torch.nn as nn

def get_config():
    cfg = EasyDict()
    cfg.workspace = './workspace/test'

    cfg.cuda = True  # Use GPU or not
    cfg.gpu = 0# gpu id
    cfg.multigpu = False
    cfg.multi_id = [0, 1]
    cfg.seed = 42  # random seed
    cfg.debug = True  # set to true to print debug infomations and dump intermediate results
    cfg.threads = 8  # number of threads
    cfg.batchsize = 2

    cfg.epoch = 400
    cfg.device = "cuda:{}".format(str(cfg.gpu))
    cfg.alpha = 0
    cfg.beta = 1
    cfg.width = 1216
    cfg.height = 128

    cfg.action = ""  # what to do
    cfg.dataset = "2015"  # version of kitti
    cfg.datapath = '/data/haitao/data/'
    cfg.checkpoint = ''
    cfg.eva_model = ""

    cfg.save_epoch = 15


    cfg.network = "fbnn" #or fnn
    cfg.net = ""  # full path to trained model
    cfg.disp_max = 128  # max disparity
    cfg.err_thres = 3  # kitti error pixel
    cfg.out = "./disp"  # predict image
    cfg.train_metric_field = ['err_3px', 'err_2px', 'err_1px', 'rmse', 'mre', 'mae']

    # Training Options
    cfg.dataset_pos_low = 1
    cfg.dataset_neg_low = 4
    cfg.dataset_neg_high = 10

    # Optimization Options
    cfg.optimization = "adam"
    cfg.lr = 0.0002  # learning rate
    cfg.m = 0.2 # margin
    cfg.pow = 1  # pow

    # dataset hyperparameters
    cfg.hflip = 0
    cfg.vflip = 0
    cfg.rotate = 7
    cfg.hscale = 0.9
    cfg.scale = 1
    cfg.trans = 0
    cfg.hshear = 0.1
    cfg.brightness = 0.7
    cfg.contrast = 1.3



    return cfg
