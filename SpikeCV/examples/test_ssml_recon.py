# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn
import numpy as np
import sys, time
sys.path.append("..")
from spkProc.reconstruction.SSML_Recon.ssml_model import SSML_ReconNet
import cv2, os
from spkData.load_dat import data_parameter_dict
from spkData.load_dat import SpikeStream
from utils import path
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict

def load_network(load_path, network, strict=True, nameOfPatialLoad=None):
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module

    if nameOfPatialLoad:
        network0 = network.bsn
    else:
        network0 = network

    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    # print(load_net_clean.keys())
    if nameOfPatialLoad:
        load_net_clean_partial = OrderedDict()
        for k, v in load_net_clean.items():
            if k.startswith(nameOfPatialLoad):
                load_net_clean_partial[k[len(nameOfPatialLoad):]] = v
    else:
        load_net_clean_partial = load_net_clean

    network0.load_state_dict(load_net_clean_partial, strict=strict)
    return network0

if __name__ == '__main__':
    model = SSML_ReconNet()
    model_path = os.path.join("..", "spkProc", "reconstruction", "SSML_Recon", "pretrained", "pretrained_ssml_recon.pt")
    # model.load_state_dict(torch.load(model_path))
    model = load_network(model_path, model)
    model = model.cuda()

    # Specifies the data sequence and task type
    data_filename = "recVidarReal2019/classA/car-100kmh"
    label_type = 'raw'

    # Load dataset attribute dictionary
    paraDict = data_parameter_dict(data_filename, label_type)

    # load spike dataset
    vidarSpikes = SpikeStream(**paraDict)
    block_len = 41
    spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len) # T H W numpy

    spikes = torch.from_numpy(spikes.astype(np.float32)).unsqueeze(0).cuda()

    st = time.time()
    res = model(spikes, train=False)
    ed = time.time()
    print('shape: ', res.shape, 'time: {:.6f}'.format(ed - st))
    res = res[0].detach().cpu().permute(1,2,0).numpy()*255

    filename = path.split_path_into_pieces(data_filename)
    if not os.path.exists('results'):
        os.makedirs('results')

    res_path = os.path.join('results', filename[-1] + '_ssml_recon_res.png')
    cv2.imwrite(res_path,res)

    print("done.")