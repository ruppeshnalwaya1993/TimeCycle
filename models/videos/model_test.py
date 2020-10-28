# from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
# from correlation_package.modules.corr import Correlation
import math
import copy
import numpy as np
from . import resnet_res4s1
from . import inflated_resnet18 as inflated_resnet
# import i3res_res3
import torchvision

import torch.nn.functional as F
from geotnf.transformation import GeometricTnf,GeometricTnfAffine
from geotnf.loss import TransformedGridLoss, WeakInlierCountPool
from utils.torch_util import expand_dim

import random
import utils.imutils2


import time
import sys

class CycleTime(nn.Module):

    def __init__(self, class_num=8, dim_in=2048, trans_param_num=6, detach_network=False, pretrained=True, temporal_out=4, T=None):
        super(CycleTime, self).__init__()

        dim = 512
        print(pretrained)
        self.pretrained = pretrained
        resnet = resnet_res4s1.resnet18(pretrained=pretrained)
        self.encoderVideo = inflated_resnet.InflatedResNet(copy.deepcopy(resnet))
        self.detach_network = detach_network

        self.div_num = 512
        self.T = self.div_num**-.5 if T is None else T

        print('self.T:', self.T)

        # self.encoderVideo = resnet3d.resnet50(pretrained=False)

        self.afterconv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)

        self.spatial_out1 = 30
        self.spatial_out2 = 10
        self.temporal_out = temporal_out

        self.afterconv3_trans = nn.Conv2d(self.spatial_out1 * self.spatial_out1, 128, kernel_size=4, padding=0, bias=False)
        self.afterconv4_trans = nn.Conv2d(128, 64, kernel_size=4, padding=0, bias=False)

        corrdim = 64 * 4 * 4
        corrdim_trans = 64 * 4 * 4

        self.linear2 = nn.Linear(corrdim_trans, trans_param_num)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.avgpool3d = nn.AvgPool3d((4, 2, 2), stride=(1, 2, 2))
        self.maxpool2d = nn.MaxPool2d(2, stride=2)


        # initialization

        nn.init.kaiming_normal_(self.afterconv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.afterconv3_trans.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.afterconv4_trans.weight, mode='fan_out', nonlinearity='relu')

        # assuming no fc pre-training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # transformation
        self.geometricTnf = GeometricTnfAffine(geometric_model='affine',
                                         tps_grid_size=3,
                                         tps_reg_factor=0.2,
                                         out_h=self.spatial_out2, out_w=self.spatial_out2,
                                         offset_factor=227/210)

        self.geometricTnf_img = GeometricTnfAffine(geometric_model='affine',
                                         tps_grid_size=3,
                                         tps_reg_factor=0.2,
                                         out_h=80, out_w=80,
                                         offset_factor=227/210)


        xs = np.linspace(-1,1,80)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        self.xs = xs


    def compute_corr_softmax(self, patch_feat1, r50_feat2, search_radius=12):
        # brussell: Updated this function
        # patch_feat1 - [batch_size, feature_dim, height, width]
        # r50_feat2 - [batch_size, feature_dim, num_context_frames, height, width]

        #np.savez('/home/code-base/compute_corr_softmax_inputs.npz', patch_feat1=patch_feat1.cpu().detach().numpy(), r50_feat2=r50_feat2.cpu().detach().numpy())
        patch_size = 2*search_radius+1
        batch_size, feature_dim, num_context_frames, height, width = r50_feat2.size()
        assert(height==patch_feat1.size(2))
        assert(width==patch_feat1.size(3))

        r50_feat2_pad = F.pad(r50_feat2, (search_radius, search_radius, search_radius, search_radius))
        r50_feat2_pad = r50_feat2_pad.contiguous()
        corrfeat = torch.zeros(batch_size, num_context_frames, patch_size, patch_size, height, width).cuda()
        valid = torch.ones(batch_size, num_context_frames, patch_size, patch_size, height, width).cuda()
        for c in range(num_context_frames):
            for h in range(-search_radius, search_radius+1):
                hmin = search_radius + h
                hmax = height + search_radius + h
                for w in range(-search_radius, search_radius+1):
                    wmin = search_radius + w
                    wmax = width + search_radius + w
                    corrfeat[:, c, hmin, wmin, :, :] = \
                        torch.sum(r50_feat2_pad[:, :, c, hmin:hmax, wmin:wmax] * patch_feat1, axis=1)  # 10s
                    if h < 0:
                        valid[:, c, hmin, wmin, :-h, :] = 0
                    if w < 0:
                        valid[:, c, hmin, wmin, :, :-w] = 0
                    if h > 0:
                        valid[:, c, hmin, wmin, -h:, :] = 0
                    if w > 0:
                        valid[:, c, hmin, wmin, :, -w:] = 0

        corrfeat = corrfeat.view(batch_size, num_context_frames*patch_size**2, -1)

        # if self.use_l2norm is False:
        corrfeat = torch.div(corrfeat, self.div_num**-.5)  ### brussell: This line should be removed in final code.
        corrfeat  = corrfeat.view(batch_size, num_context_frames, patch_size**2, height, width)
        corrfeat  = F.softmax(corrfeat/self.T, dim=2)
        corrfeat  = corrfeat.view(batch_size, num_context_frames, patch_size, patch_size, height, width)
        corrfeat[valid<1] = -1.0
        corrfeat  = corrfeat.contiguous()

        #np.savez('/home/code-base/compute_corr_softmax_output.npz', corrfeat=corrfeat.cpu().detach().numpy())
        return corrfeat


    def forward(self, ximg1, img2, retfeats=False):

        bs = ximg1.size(0)
        finput_num = ximg1.size(1)

        ximg1_images = ximg1.view(ximg1.size(0) * ximg1.size(1), ximg1.size(2), ximg1.size(3), ximg1.size(4)).clone()

        videoclip1  = ximg1

        # video feature clip1
        videoclip1 = videoclip1.transpose(1, 2)
        r50_feat1 = self.encoderVideo(videoclip1)
        if self.detach_network is True:
            r50_feat1 = r50_feat1.detach()

        if not self.pretrained:
            r50_feat1 = self.afterconv1(r50_feat1)
        r50_feat1_relu = self.relu(r50_feat1)
        # if self.use_softmax is False or self.use_l2norm is True:
        r50_feat1_norm = F.normalize(r50_feat1_relu, p=2, dim=1)


        # target image feature
        img2 = img2.transpose(1, 2)
        img_feat2_pre = self.encoderVideo(img2)

        if not self.pretrained:
            img_feat2 = self.afterconv1(img_feat2_pre)
        else:
            img_feat2 = img_feat2_pre
        img_feat2 = self.relu(img_feat2)
        img_feat2 = img_feat2.contiguous()
        img_feat2 = img_feat2.view(img_feat2.size(0), img_feat2.size(1), img_feat2.size(3), img_feat2.size(4))
        img_feat2_norm = F.normalize(img_feat2, p=2, dim=1)

        spatial_out1 = img_feat2.size(2)
        spatial_out2 = img_feat2.size(3)

        # brussell: Updated the next line:
        corrfeat_trans_matrix_target  = self.compute_corr_softmax(img_feat2_norm, r50_feat1_norm)

        return corrfeat_trans_matrix_target
