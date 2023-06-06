# -*- coding: utf-8 -*- 


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_normal_, constant_
from spkProc.optical_flow.SCFlow0.util import predict_flow, conv_s, conv, deconv
from spkProc.optical_flow.SCFlow0.corr import corr
from spkProc.optical_flow.SCFlow0.utils import flow_warp

__all__ = ['barepwc_d']


class FeatureEncoder(nn.Module):
    def __init__(self, num_chs, batchNorm=True):
        super(FeatureEncoder, self).__init__()
        self.batchNorm = batchNorm
        self.num_chs = num_chs
        self.conv_list = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            if l == 0:
                layer = nn.Sequential(
                    conv(self.batchNorm, ch_in, ch_out, kernel_size=3, stride=1),
                    conv(self.batchNorm, ch_out, ch_out, kernel_size=3, stride=1)
                )
            else:
                layer = nn.Sequential(
                    conv(self.batchNorm, ch_in, ch_out, kernel_size=3, stride=2),
                    conv(self.batchNorm, ch_out, ch_out, kernel_size=3, stride=1)
                )
            self.conv_list.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv_module in self.conv_list:
            x = conv_module(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimator(nn.Module):
    def __init__(self, ch_in, batchNorm=True):
        super(FlowEstimator, self).__init__()
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, ch_in, 96, kernel_size=3, stride=1)
        self.conv2 = conv(self.batchNorm, ch_in + 96, 64, kernel_size=3, stride=1)
        self.conv3 = conv(self.batchNorm, ch_in + 96 + 64, 32, kernel_size=3, stride=1)
        self.conv4 = conv_s(self.batchNorm, ch_in + 96 + 64 + 32, 2, kernel_size=3, stride=1)


    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x_out = self.conv4(x3)
        return x_out


class vfnet(nn.Module):
    def __init__(self, batchNorm):
        super(vfnet, self).__init__()
        self.batchNorm = batchNorm
        self.search_range = 4
        self.num_chs = [25, 32, 64, 96, 128]
        self.output_level = 4
        self.leakyReLU = nn.LeakyReLU(0.1, inplace=True)

        # self.spike_representation = SpikeRepresentation(data_length=41, batchNorm=self.batchNorm)
        self.feature_encoder = FeatureEncoder(num_chs=self.num_chs, batchNorm=self.batchNorm)
        # self.corr = Correlation()
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 32 + self.dim_corr + 2
        self.flow_estimators = FlowEstimator(self.num_ch_in)
        # self.num_context_in = 2 + self.num_ch_in+96+64+32
        # self.context_network = ContextNetwork(ch_in=self.num_context_in)

        self.conv_1x1 = nn.ModuleList([conv_s(False, 128, 32, kernel_size=1, stride=1),
                                       conv_s(False, 96, 32, kernel_size=1, stride=1),
                                       conv_s(False, 64, 32, kernel_size=1, stride=1),
                                       conv_s(False, 32, 32, kernel_size=1, stride=1)])

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def norm_feature(self, x):
        # mean_x = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # var_x = x.var(dim=2, keepdim=True).var(dim=3, keepdim=True)
        # return (x - mean_x) / (var_x + 1e-6)
        return x

    def forward(self, seq1, seq2):
        res_dict = {}

        flows = []

        # x1_repre = self.spike_representation(seq1)
        # x2_repre = self.spike_representation(seq2)

        # res_dict['x1_repre'] = x1_repre.mean(dim=1, keepdim=True)
        # res_dict['x2_repre'] = x2_repre.mean(dim=1, keepdim=True)

        x1_pym = self.feature_encoder(seq1)
        x2_pym = self.feature_encoder(seq2)

        b, c, h, w = x1_pym[0].shape
        init_dtype = x1_pym[0].dtype
        init_device = x1_pym[0].device
        flow = torch.zeros(b, 2, h, w, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pym, x2_pym)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2, mode='bilinear', align_corners=True)
                x2_warp = flow_warp(x2, flow)

            # correlation
            # out_corr = self.corr(x1, x2_warp)
            # out_corr_relu = self.leakyReLU(out_corr)

            x1_norm = self.norm_feature(x1)
            x2_warp_norm = self.norm_feature(x2_warp)
            out_corr = corr(x1_norm, x2_warp_norm)

            # flow estimating
            x1_1x1 = self.conv_1x1[l](x1)
            flow_res = self.flow_estimators(torch.cat([out_corr, x1_1x1, flow], dim=1))
            flow = flow + flow_res

            # flow_context = self.context_network(torch.cat([conved_feature, flow], dim=1))
            # flow = flow + flow_context

            flows.append(flow)

        # return flows[::-1], res_dict
        return flow

def barepwc_d(data=None, batchNorm=True):
    model = vfnet(batchNorm=batchNorm)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    else:
        model.init_weights()
    return model

