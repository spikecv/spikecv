# -*- coding: utf-8 -*- 
# @Time : 2023/3/7 14:40 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : test_realtime_usb.py
import os
import sys
import ctypes
import cv2
import time
from threading import Thread

sys.path.append('..')
from spkData.load_dat_jy import device_parameters, SpikeStream
from visualization.get_video import obtain_spike_video
from sdk import spikelinkapi as link
import numpy as np
from spkProc.reconstruction.SSML_Recon.ssml_model import SSML_ReconNet, Blind_UNet
from spkProc.recognition import svm
from spkProc.recognition.rpsnet import RPSNet
from spkProc.filters.fir_filter import MeanFilter
from spkProc.optical_flow.SCFlow0.scflow0 import vfnet
from spkProc.optical_flow.SCFlow.utils import InputPadder
from spkProc.reconstruction.tfi import TFI
from visualization.optical_flow_visualization import flow_visualization

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from queue import Queue

# 全局变量 最先初始化的配置
params = link.SpikeLinkInitParams()
params_camera = link.SpikeLinkQSFPInitParams()
params_dummy = link.SpikeLinkDummyInitParams()
usbParams = link.SpikeLinkUSB3InitParams()

DEBUG_OUT = False
count = 0
pool_len = 200
input_c = link.spikelinkInput("../device/spikevision/m1k40/sdk/lib/Release/spikelinkapi.dll")
input_c.linkinputlib.ReleaseFrame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
framepool = link.spikeframepool()

type = 0  # dummy: 0, online camera pcie: 1 online camera usb: 2
# filename = 'F:\\datasets\\B-left-45-jy.dat'
# filename = 'F:\\datasets\\S-left-90-jy.dat'
# filename = 'F:\\datasets\\val-3-jy.dat'
# filename = "F:\\datasets\\rps_datasets\\p-4.dat"
# filename = "F:\\datasets\\rps_datasets\\r-6.dat"
# filename = "F:\\datasets\\rps_datasets\\s-13.dat"
filename = "F:\\datasets\\rps_datasets\\val-r-1.dat"
# filename = "F:\datasets\\1013spikedata\\cross3.dat"
spike_height = 250
spike_width = 400
cusum = 271  # 连续取一段的脉冲长度
paraDict = {'spike_width': spike_width, 'spike_height': spike_height}

process_done = True
pred_0 = np.asarray(cv2.imread('0.png'))
pred_1 = np.asarray(cv2.imread('1.png'))
pred_2 = np.asarray(cv2.imread('2.png'))

recog_res_icon = np.zeros([3, pred_0.shape[0], pred_0.shape[1], pred_0.shape[2]])
recog_res_icon[0] = pred_0
recog_res_icon[1] = pred_1
recog_res_icon[2] = pred_2
cv2.namedWindow('reconstruction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('reconstruction', spike_width, spike_height)
cv2.moveWindow("reconstruction", 200, 200)

cv2.namedWindow('recognition', cv2.WINDOW_NORMAL)
cv2.resizeWindow('recognition', spike_height, spike_height)
cv2.moveWindow("recognition", 700, 200)

cv2.namedWindow('optical flow', cv2.WINDOW_NORMAL)
cv2.resizeWindow('optical flow', spike_width, spike_height)
cv2.moveWindow('optical flow', 1000, 200)
spkTenQueue = Queue(maxsize=0)
spkNpyQueue = Queue(maxsize=0)
process_done = True

def inputcallback(frame):
    # if not vidarSpikes.brunning:
    #     input_c.releaseFrame(frame)
    #     return

    frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
    # if frame2.contents.pts < pool_len:
    # print('frame pts: ', frame2.contents.pts)
    if framepool.size() == 0:
        framepool.push(frame)
    else:
        input_c.releaseFrame(frame)
    vidarSpikes.count += 1

    if DEBUG_OUT:
        print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height,
              frame2.contents.pts)
    if vidarSpikes.count % 10 == 0:
        print("index:", frame2.contents.pts)

    # input_c.releaseFrame(frame)


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


def recon_thread(spikes, model):
    spike_len = spikes.shape[1]
    batch_size = spikes.shape[0]
    mini_batch_size = 12
    batch_num = int(batch_size/mini_batch_size)
    for i_batch in range(batch_num):
        begin_idx = i_batch * mini_batch_size
        end_idx = begin_idx + mini_batch_size
        # recon_spikes = torch.from_numpy(spikes[begin_idx:end_idx, spike_len - recon_input_win:, :, :].astype(np.float32)).to('cuda')
        recon_spikes = spikes[begin_idx:end_idx, spike_len - recon_input_win:, :, :]
        # real-time algorithm perform
        with torch.no_grad():
            res = model(recon_spikes, train=False)
            res = res.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
        # res = (res - res.min()) / (res.max() - res.min())
        # res = res ** (1 / 2.2) * 255
        # res = res / 0.6 * 255
        res = np.squeeze(res)
        recon_res[begin_idx:end_idx] = res
    # res = reconstructor.spikes2frame(spikes.astype(np.float32), int(cusum/2), int(cusum/2)-1)
    # recon_res[index, :, :] = res / 0.6

    return


def recog_thread(model):
    spikes = spkNpyQueue.get()
    spike_len = spikes.shape[1]
    batch_size = spikes.shape[0]

    recog_spikes = spikes[:, spike_len - recog_input_win:, :, :]
    # recog_spikes = recog_spikes.reshape(1, recog_input_win, spike_height, spike_width)

    # spk_data = torch.from_numpy(recog_spikes).cuda().unsqueeze(0)
    # model.eval()
    # out = model(spk_data)
    # recog_pred = out.argmax(1)
    # recog_res[index] = recog_pred.item()

    # real-time algorithm perform
    recog_res = model.predict(recog_spikes)
    # print('recognition results: ', recog_pred[0])

    return


def of_thread(spikes, model):
    # spike1_begin = cusum - 2 * of_input_win - 20
    spike1_begin = 0
    spike1_end = spike1_begin + of_input_win
    spike_len = spikes.shape[1]
    batch_size = spikes.shape[0]
    mini_batch_size = 4
    batch_num = int(batch_size / mini_batch_size)
    for i_batch in range(batch_num):
        begin_idx = i_batch * mini_batch_size
        end_idx = begin_idx + mini_batch_size
        # of_spikes_1 = torch.from_numpy(spikes[begin_idx:end_idx, spike1_begin:spike1_end, :, :].astype(np.float32)).to('cuda')
        # of_spikes_2 = torch.from_numpy(spikes[begin_idx:end_idx, spike_len - of_input_win:, :, :].astype(np.float32)).to('cuda')
        of_spikes_1 = spikes[begin_idx:end_idx, spike1_begin:spike1_end, :, :]
        of_spikes_2 = spikes[begin_idx:end_idx, spike_len - of_input_win:, :, :]
        of_spikes_1, of_spikes_2 = padder.pad(of_spikes_1, of_spikes_2)
        with torch.no_grad():
            of_pred = model(of_spikes_1, of_spikes_2)
            of_pred = padder.unpad(of_pred)
        of_res[begin_idx:end_idx] = of_pred.detach().permute([0, 2, 3, 1]).cpu().numpy()
    # tmp_of_res = of_pred[0].detach().permute([1, 2, 0]).cpu().numpy()
    # tmp_of_res = of_pred[0].detach().permute([1, 2, 0]).numpy()
    # of_res[index] = tmp_of_res

    return


def vis_thread():
    # num_res = recog_res.shape[0]
    fps = evaluate_seq_len
    # for index in range(num_res):
    show_count = 0
    while 1:
        index = show_count % fps
        tmp_recog = int(recog_res[index])
        cv2.imshow('recognition', np.squeeze(recog_res_icon[tmp_recog] / 255))

        tmp_recon = np.squeeze(recon_res[index, :, :])
        cv2.imshow('reconstruction', tmp_recon / 255)

        of_vis = flow_visualization(of_res[index], mode='normal', use_cv2=True)
        cv2.imshow("optical flow", of_vis)

        # cv2.waitKey(int(1000/evaluate_seq_len))

        # if cv2.waitKey(int(1000/fps)):
        if cv2.waitKey(int(33)):
            break
        show_count += 1

    return


def data_thread(spikeStream):
    with torch.cuda.stream(torch.cuda.Stream()):
        # spk_tensor = spikeStream.to_tensor()
        spikes = spikeStream.get_device_matrix(_input=input_c, _framepool=framepool, cusum=cusum)

        crop_spikes = np.zeros([evaluate_seq_len, crop_block_len, spike_height, spike_width])
        begin_spike_ind = np.arange(0, cusum - crop_block_len + 1, stride_step)
        end_spike_ind = begin_spike_ind + crop_block_len

        for index in range(evaluate_seq_len):
            crop_spikes[index] = spikes[begin_spike_ind[index]:end_spike_ind[index], :, :].astype(np.float32)

        spkNpyQueue.put(crop_spikes)
        spike_tensor = torch.Tensor(crop_spikes).to('cuda')
        spike_tensor = spike_tensor.half()
        # spike_tensor = spike_tensor.to('cuda')

        spkTenQueue.put(spike_tensor)


# algorithms
# 1. reconstruction model loading
# recon_model = SSML_ReconNet()
recon_model = Blind_UNet(41, 1, False, False)
# model_path = os.path.join("..", "spkProc", "reconstruction", "SSML_Recon", "pretrained", "pretrained_ssml_recon.pt")

model_path = os.path.join("..", "spkProc", "reconstruction", "SSML_Recon", "pretrained", "nbsn_83000_simple.pth")
# if isinstance(recon_model, nn.DataParallel) or isinstance(recon_model, DistributedDataParallel):
#     recon_model = recon_model.module
# recon_model.load_state_dict(torch.load(model_path))
recon_model = load_network(model_path, recon_model)
recon_model = recon_model.half()
recon_model = recon_model.to('cuda')
reconstructor = TFI(spike_height, spike_width, torch.device('cuda'))
recon_input_win = 41
# 2. recognition model loading
recog_input_win = 20
filter_svm = svm.TemporalFilteringSVM(filter=MeanFilter(win=recog_input_win), dual=False)
model_path = os.path.join("..", "spkProc", "recognition", "rps_model", "svm-rps.pkl")
filter_svm.load_model(model_path)
# rpsnet = RPSNet().to(torch.device('cuda'))
# model_path = os.path.join("..", "spkProc", "recognition", "rps_model", "rpsnet.pkl")
# rpsnet.load_state_dict(torch.load(model_path))
# rpsnet.eval()
# 3. optical flow model loading
of_input_win = 25
optical_flow_pretrained_path = os.path.join("..", "spkProc", "optical_flow", "SCFlow", "pretrained", "barepwc_e70.pth")
optical_flow_model = vfnet(batchNorm=False)
# optical_flow_model = load_network(optical_flow_pretrained_path, optical_flow_model)
# optical_flow_model = optical_flow_model.cuda()
optical_flow_network_data = torch.load(optical_flow_pretrained_path)

optical_flow_model = torch.nn.DataParallel(optical_flow_model)
optical_flow_model.load_state_dict(optical_flow_network_data)
optical_flow_model = optical_flow_model.half()
optical_flow_model = optical_flow_model.to('cuda')

input_callback = link.LinkInputCallBack(inputcallback)

if type == 0:
    device_params = device_parameters(params, params_dummy, type, filename, spike_width, spike_height, cusum)
else:
    device_params = device_parameters(params, usbParams, type, filename, spike_width, spike_height, cusum)

input_c.init(ctypes.byref(device_params))
input_c.open()

input_c.setcallback(input_callback)
# paraDict['params'] = device_params
vidarSpikes = SpikeStream(offline=False, camera_type='USB', **paraDict)

sim_times = 60
read_spikes_time = 0
crop_block_len = 41
stride_step = 10
padder = InputPadder(dims=(250, 400))

evaluate_seq_len = 24

input_c.start()
for it in range(sim_times):
    # spikes = vidarSpikes.get_device_matrix(_input=input_c, _framepool=framepool, block_len=pool_len, cusum=cusum)
    # total_spikes = spikes
    # evaluate_seq_len = 30
    recon_res = np.zeros([evaluate_seq_len, spike_height, spike_width])
    of_res = np.zeros([evaluate_seq_len, spike_height, spike_width, 2])
    recog_res = np.zeros([evaluate_seq_len, ])

    # spikes = vidarSpikes.get_device_matrix(_input=input_c, _framepool=framepool, cusum=cusum)
    total_time = 0

    st = time.time()
    # for i_seq in range(evaluate_seq_len):
    # st = time.time()
    # read_spikes_time += time.time() - st
    # tmp_spike = crop_spikes[i_seq]

    data_th = Thread(data_thread(vidarSpikes))
    data_th.start()

    crop_spikes = spkTenQueue.get()
    rc_th = Thread(recon_thread(crop_spikes, recon_model))
    rcog_th = Thread(recog_thread(filter_svm))
    of_th = Thread(of_thread(crop_spikes, optical_flow_model))
    vis_th = Thread(vis_thread())

    process_done = False

    of_th.start()
    rcog_th.start()
    rc_th.start()
    vis_th.start()

    data_th.join()
    of_th.join()
    rcog_th.join()
    rc_th.join()
    vis_th.join()

    # recon_thread(crop_spikes, recon_model)
    # recog_thread(crop_spikes, filter_svm)
    # of_thread(crop_spikes, optical_flow_model)
    # vis_thread()

    process_done = True

    # total_spikes = np.concatenate((total_spikes, spikes), axis=0)

    total_time += (time.time() - st)
    # print('decode spikes take: %.3f seconds' % read_spikes_time)
    print('All algorithms took: %.3f seconds for %d block spikes' %
          (total_time, evaluate_seq_len))

vidarSpikes.brunning = False
input_c.stop()
input_c.close()

spike_filename = os.path.join('results', 'test_device_usb.avi')
save_paraDict = {'spike_h': spike_height, 'spike_w': spike_width}

# save results
if not os.path.exists('./results'):
    os.makedirs('results')
res_dir = os.path.join('results', 'real_system')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
recon_res_filepath = os.path.join(res_dir, 'recon_')
of_res_filepath = os.path.join(res_dir, 'of_')
recog_res_filepath = os.path.join(res_dir, 'recog_')
print('recognition results: ', recog_res)


for i_seq in range(evaluate_seq_len):
    tmp_recon = np.squeeze(recon_res[i_seq, :, :])
    recon_filename = recon_res_filepath + str(i_seq) + '.png'
    cv2.imwrite(recon_filename, tmp_recon)

    # cv2.imshow('reconstruction', tmp_recon / 255)
    of_filename = of_res_filepath + str(i_seq) + '.png'
    of_vis = flow_visualization(of_res[i_seq], mode='normal', use_cv2=True)
    cv2.imwrite(of_filename, of_vis)

    # tmp_recog = int(recog_res[i_seq])
    # cv2.imshow('recognition', np.squeeze(recog_res_icon[tmp_recog] / 255))

    # cv2.imshow("optical flow", of_vis)
    # cv2.waitKey()
    # if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
    #     continue

cv2.destroyAllWindows()
# obtain_spike_video(total_spikes, spike_filename, **save_paraDict)
