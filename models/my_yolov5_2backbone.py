from curses import ncurses_version
import torch
import cv2
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
# print('downloaded model')
# # Images
# img = cv2.imread('./data/images/zidane.jpg')  # or file, Path, PIL, OpenCV, numpy, list
# print('downloaded image')
# img = cv2.COLOR_Convet
# # Inference
# results = model(img)
# print('inference done')
# # Results
# # results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
# print(model.eval())

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

hyp = '/home/swbaelab/Documents/yolov5_cycleGAN_no_letterbox/data/hyps/hyp.scratch-low.yaml'
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False  # export mode

    def __init__(self, nc=1, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

class build_yolov5s_2backbone(nn.Module):
    def __init__(self, cfg='models/yolov5s.yaml', ch=6, nc=1):
        super().__init__()
        
        self.yaml = cfg
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict
                
        ch_RGB = 3   # number of channels of RGB images
        ch_IR = 3    # number of channels of IR images
        LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
        nc = 1       # number of classes(categories)
        anchors = [[10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326]]
        gd = 0.33
        gw = 0.50
        na = (len(anchors[0]) // 2)
        no = na * (nc + 5)   # number of outputs = anchors * (classes + 5)
        # print('number of output no: ', no)
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]
        
        
                  # [from, number, module, args]
        backbone_RGB = [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
                        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
                        [-1, 3, C3, [128]],          # 2
                        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
                        [-1, 6, C3, [256]],          # 4
                        [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
                        [-1, 9, C3, [512]],          # 6
                        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
                        [-1, 3, C3, [1024]],         # 8
                        [-1, 1, SPPF, [1024, 5]],  # 9
                        ]
        backbone_IR =  [[-1, 1, Conv, [64, 6, 2, 2]],  # 10-P1/2
                        [-1, 1, Conv, [128, 3, 2]],  # 11-P2/4
                        [-1, 3, C3, [128]],          # 12
                        [-1, 1, Conv, [256, 3, 2]],  # 13-P3/8
                        [-1, 6, C3, [256]],          # 14
                        [-1, 1, Conv, [512, 3, 2]],  # 15-P4/16
                        [-1, 9, C3, [512]],          # 16
                        [-1, 1, Conv, [1024, 3, 2]],  # 17-P5/32
                        [-1, 3, C3, [1024]],         # 18
                        [-1, 1, SPPF, [1024, 5]],  # 19
                        ]
        head = [[[9, 19], 1, Concat, [1]],   # 20
                [-1, 1, Conv, [512, 1, 1]],  # 21
                [-1, 1, nn.Upsample, [None, 2, 'nearest']],  # 22
                [[-1, 6, 16], 1, Concat, [1]],  # cat backbone P4  # 23
                [-1, 3, C3, [512, False]],  # 24

                [-1, 1, Conv, [256, 1, 1]],  # 25
                [-1, 1, nn.Upsample, [None, 2, 'nearest']],  # 26
                [[-1, 4, 14], 1, Concat, [1]],  # cat backbone P3  # 27
                [-1, 3, C3, [256, False]],  # 28 (P3/8-small)

                [-1, 1, Conv, [256, 3, 2]], # 29
                [[-1, 25], 1, Concat, [1]],  # cat head P4  # 30
                [-1, 3, C3, [512, False]],  # 31 (P4/16-medium)

                [-1, 1, Conv, [512, 3, 2]],  # 32
                [[-1, 21], 1, Concat, [1]],  # cat head P5  # 33
                [-1, 3, C3, [1024, False]],  # 34 (P5/32-large)

                [[28, 31, 34], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)  # 35
                ]
        backbone_head = backbone_RGB + backbone_IR + head
        # print('backbone_head: ', backbone_head)
        
        ############ backbone_RGB #######
        ############### 0 ###############
        i, f, n, m, args = 0, backbone_head[0][0], backbone_head[0][1], backbone_head[0][2], backbone_head[0][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch_RGB, args[0]     # input channel: 3, output channel: 64
        c2 = make_divisible(c2*gw, 8)      # output channel:32 = c2:64 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
        
        ############### 1 ###############
        i, f, n, m, args = 1, backbone_head[1][0], backbone_head[1][1], backbone_head[1][2], backbone_head[1][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 32, output channel: 128
        c2 = make_divisible(c2*gw, 8)      # output channel:64 = c2:128 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 2 ###############
        i, f, n, m, args = 2, backbone_head[2][0], backbone_head[2][1], backbone_head[2][2], backbone_head[2][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 64, output channel: 128
        c2 = make_divisible(c2*gw, 8)      # output channel:64 = c2:128 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 3 ###############
        i, f, n, m, args = 3, backbone_head[3][0], backbone_head[3][1], backbone_head[3][2], backbone_head[3][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 64, output channel: 256
        c2 = make_divisible(c2*gw, 8)      # output channel:128 = c2:256 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 4 ###############
        i, f, n, m, args = 4, backbone_head[4][0], backbone_head[4][1], backbone_head[4][2], backbone_head[4][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 128, output channel: 256
        c2 = make_divisible(c2*gw, 8)      # output channel:128 = c2:256 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 5 ###############
        i, f, n, m, args = 5, backbone_head[5][0], backbone_head[5][1], backbone_head[5][2], backbone_head[5][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 128, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 6 ###############
        i, f, n, m, args = 6, backbone_head[6][0], backbone_head[6][1], backbone_head[6][2], backbone_head[6][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 7 ###############
        i, f, n, m, args = 7, backbone_head[7][0], backbone_head[7][1], backbone_head[7][2], backbone_head[7][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 1024
        c2 = make_divisible(c2*gw, 8)      # output channel:512 = c2:1024 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 8 ###############
        i, f, n, m, args = 8, backbone_head[8][0], backbone_head[8][1], backbone_head[8][2], backbone_head[8][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 512, output channel: 1024
        c2 = make_divisible(c2*gw, 8)      # output channel:512 = c2:1024 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 9 ###############
        i, f, n, m, args = 9, backbone_head[9][0], backbone_head[9][1], backbone_head[9][2], backbone_head[9][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 512, output channel: 1024
        c2 = make_divisible(c2*gw, 8)      # output channel:512 = c2:1024 * gw:0.50
        args = [c1, c2, *args[1:]]
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   SPPF
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############ backbone_IR #######
        ############### 10 ###############
        i, f, n, m, args = 10, backbone_head[10][0], backbone_head[10][1], backbone_head[10][2], backbone_head[10][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch_IR, args[0]     # input channel: 3, output channel: 64
        c2 = make_divisible(c2*gw, 8)      # output channel:32 = c2:64 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 11 ###############
        i, f, n, m, args = 11, backbone_head[11][0], backbone_head[11][1], backbone_head[11][2], backbone_head[11][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        
        c1, c2 = ch[f], args[0]     # input channel: 32, output channel: 128
        c2 = make_divisible(c2*gw, 8)      # output channel:64 = c2:128 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 12 ###############
        i, f, n, m, args = 12, backbone_head[12][0], backbone_head[12][1], backbone_head[12][2], backbone_head[12][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 64, output channel: 128
        c2 = make_divisible(c2*gw, 8)      # output channel:64 = c2:128 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 13 ###############
        i, f, n, m, args = 13, backbone_head[13][0], backbone_head[13][1], backbone_head[13][2], backbone_head[13][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 64, output channel: 256
        c2 = make_divisible(c2*gw, 8)      # output channel:128 = c2:256 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 14 ###############
        i, f, n, m, args = 14, backbone_head[14][0], backbone_head[14][1], backbone_head[14][2], backbone_head[14][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 128, output channel: 256
        c2 = make_divisible(c2*gw, 8)      # output channel:128 = c2:256 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 15 ###############
        i, f, n, m, args = 15, backbone_head[15][0], backbone_head[15][1], backbone_head[15][2], backbone_head[15][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 128, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 16 ###############
        i, f, n, m, args = 16, backbone_head[16][0], backbone_head[16][1], backbone_head[16][2], backbone_head[16][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 17 ###############
        i, f, n, m, args = 17, backbone_head[17][0], backbone_head[17][1], backbone_head[17][2], backbone_head[17][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 1024
        c2 = make_divisible(c2*gw, 8)      # output channel:512 = c2:1024 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 18 ###############
        i, f, n, m, args = 18, backbone_head[18][0], backbone_head[18][1], backbone_head[18][2], backbone_head[18][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 512, output channel: 1024
        c2 = make_divisible(c2*gw, 8)      # output channel:512 = c2:1024 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 19 ###############
        i, f, n, m, args = 19, backbone_head[19][0], backbone_head[19][1], backbone_head[19][2], backbone_head[19][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 512, output channel: 1024
        c2 = make_divisible(c2*gw, 8)      # output channel:512 = c2:1024 * gw:0.50
        args = [c1, c2, *args[1:]]
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   SPPF
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        
        ############## HEAD #############
        ############### 20 ###############
        i, f, n, m, args = 20, backbone_head[20][0], backbone_head[20][1], backbone_head[20][2], backbone_head[20][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c2 = sum([ch[x] for x in f])
        m_ = m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: Concat
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 21 ###############
        i, f, n, m, args = 21, backbone_head[21][0], backbone_head[21][1], backbone_head[21][2], backbone_head[21][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 512, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 22 ###############
        i, f, n, m, args = 22, backbone_head[22][0], backbone_head[22][1], backbone_head[22][2], backbone_head[22][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c2 = ch[f]
        m_ = m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # module type: nn.Upsample
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 23 ###############
        i, f, n, m, args = 23, backbone_head[23][0], backbone_head[23][1], backbone_head[23][2], backbone_head[23][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c2 = sum([ch[x] for x in f])
        m_ = m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: Concat
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 24 ###############
        i, f, n, m, args = 24, backbone_head[24][0], backbone_head[24][1], backbone_head[24][2], backbone_head[24][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 25 ###############
        i, f, n, m, args = 25, backbone_head[25][0], backbone_head[25][1], backbone_head[25][2], backbone_head[25][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 256
        c2 = make_divisible(c2*gw, 8)      # output channel:128 = c2:256 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 26 ###############
        i, f, n, m, args = 26, backbone_head[26][0], backbone_head[26][1], backbone_head[26][2], backbone_head[26][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c2 = ch[f]
        m_ = m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # module type: nn.Upsample
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 27 ###############
        i, f, n, m, args = 27, backbone_head[27][0], backbone_head[27][1], backbone_head[27][2], backbone_head[27][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c2 = sum([ch[x] for x in f])
        m_ = m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: Concat
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 28 ###############
        i, f, n, m, args = 28, backbone_head[28][0], backbone_head[28][1], backbone_head[28][2], backbone_head[28][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 128, output channel: 256
        c2 = make_divisible(c2*gw, 8)      # output channel:128 = c2:256 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params   
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 29 ###############
        i, f, n, m, args = 29, backbone_head[29][0], backbone_head[29][1], backbone_head[29][2], backbone_head[29][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 128, output channel: 256
        c2 = make_divisible(c2*gw, 8)      # output channel:128 = c2:256 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 30 ###############
        i, f, n, m, args = 30, backbone_head[30][0], backbone_head[30][1], backbone_head[30][2], backbone_head[30][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c2 = sum([ch[x] for x in f])
        m_ = m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: Concat
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 31 ###############
        i, f, n, m, args = 31, backbone_head[31][0], backbone_head[31][1], backbone_head[31][2], backbone_head[31][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params   
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 32 ###############
        i, f, n, m, args = 32, backbone_head[32][0], backbone_head[32][1], backbone_head[32][2], backbone_head[32][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 256, output channel: 512
        c2 = make_divisible(c2*gw, 8)      # output channel:256 = c2:512 * gw:0.50
        args = [c1, c2, *args[1:]]
        #print('args: ', args)
        m_ = m(*args)
        #print(type(m_))
        t = str(m)[8:-2].replace('__main__.', '')  # module type:   Conv
        np = sum(x.numel() for x in m_.parameters())   # number of parameters
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  ??????
        #print('save: ', save)
        layers.append(m_)
        ch.append(c2)
        
        ############### 33 ###############
        i, f, n, m, args = 33, backbone_head[33][0], backbone_head[33][1], backbone_head[33][2], backbone_head[33][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c2 = sum([ch[x] for x in f])
        m_ = m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: Concat
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 34 ###############
        i, f, n, m, args = 34, backbone_head[34][0], backbone_head[34][1], backbone_head[34][2], backbone_head[34][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        c1, c2 = ch[f], args[0]     # input channel: 512, output channel: 1024
        c2 = make_divisible(c2*gw, 8)      # output channel:512 = c2:1024 * gw:0.50
        args = [c1, c2, *args[1:]]
        args.insert(2, n)
        n = 1
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type: C3
        np = sum(x.numel() for x in m_.parameters())  # number params   
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        ############### 35 ###############
        i, f, n, m, args = 35, backbone_head[35][0], backbone_head[35][1], backbone_head[35][2], backbone_head[35][3]
        n = n_ = max(round(n*gd), 1) if n > 1 else n
        args.append([ch[x] for x in f])
        if isinstance(args[1], int):  # number of anchors
            args[1] = [list(range(args[1] * 2))] * len(f)
            print('args: ', args[1])
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
        
        #####################################
        self.model, self.save = nn.Sequential(*layers), sorted(save)    ### return
        self.names = [str(i) for i in range(nc)]  # default names
        print('self.names: ', self.names)
        self.inplace = self.yaml.get('inplace', True)
        print('self.inplace: ', self.inplace)
        ch = 6
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')
        
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
        
        
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        x_RGB = x[:,:3,:,:]
        x_IR = x[:,3:,:,:]
        # print('shape of x_RGB, x_IR: ', x_RGB.shape, x_IR.shape)
        
        # RGB input
        m = self.model[0]
        # print('m: ', m)
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x_RGB)
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        for m in self.model[1:10]:
            # print('m: ', m)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                # print(m)
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        
        # IR input
        m = self.model[10]
        # print('m: ', m)
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x_IR)
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        for m in self.model[11:20]:
            # print(m)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            # print('pass')
        
        # Head
        m = self.model[20]
        # print('m: ', m)
        if profile:
            self._profile_one_layer(m, x, dt)
        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        # print('pass')
        y.append(x if m.i in self.save else None)  # save output
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        
        for m in self.model[21:]:
            # print(m)
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            # print('pass')
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        

        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self
        
        

if __name__ == "__main__":
    Model = build_yolov5s_2backbone()