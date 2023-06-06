# -*- coding: utf-8 -*-
# @File : test_device.py
import os
import sys
import ctypes
sys.path.append('..')
from spkData.load_dat_jy import device_parameters, SpikeStream
from visualization.get_video import obtain_spike_video, obtain_mot_video
from sdk import spikelinkapi as link
import numpy as np
from spkProc.tracking.spike_sort import SpikeSORT
import torch

params = link.SpikeLinkInitParams()
params_camera = link.SpikeLinkQSFPInitParams()
params_dummy = link.SpikeLinkDummyInitParams()

DEBUG_OUT = False
brunning = False
count = 0
pool_len = 5
input_c = link.spikelinkInput("../device/spikevision/m1k40/sdk/lib/Release/spikelinkapi.dll")
input_c.linkinputlib.ReleaseFrame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
framepool = link.spikeframepool()
cusum = 500 # for obtain a continuous spike stream


def inputcallback(frame):

    if not vidarSpikes.brunning or vidarSpikes.count > pool_len:
        input_c.releaseFrame(frame)
        return

    frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
    if frame2.contents.pts < pool_len:
        print('frame pts: ', frame2.contents.pts)
        framepool.push(frame)
        vidarSpikes.count += 1

    if DEBUG_OUT:
        print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height,
                frame2.contents.pts)
    if vidarSpikes.count % 10 == 0:
        print("index:", frame2.contents.pts)

    # input_c.releaseFrame(frame)


input_callback = link.LinkInputCallBack(inputcallback)
type = 0 # dummy: 0, online camera: 1
filename = "F:\\datasets\\0824wave.bin"
decode_width = 1024
spike_width = 1000
height = 1000

paraDict = {'decode_width': decode_width, 'spike_width': spike_width, 'height': height}

device_params = device_parameters(params, params_dummy, type, filename, decode_width, height, cusum)
input_c.init(ctypes.byref(device_params))   
input_c.open()
input_c.setcallback(input_callback)
# paraDict['params'] = device_params

vidarSpikes = SpikeStream(offline=False, camera_type='PCIE', **paraDict)

input_c.start()
spikes = vidarSpikes.get_device_matrix(_input=input_c, _framepool=framepool, cusum=cusum)
total_spikes = spikes
evaluate_seq_len = 3
for i_seq in range(evaluate_seq_len):

    spikes = vidarSpikes.get_device_matrix(_input=input_c, _framepool=framepool, cusum=cusum)
    total_spikes = np.concatenate((total_spikes, spikes), axis=0)

print(total_spikes.shape)

vidarSpikes.brunning = False
input_c.stop()
input_c.close()

if not os.path.exists('./results'):
    os.makedirs('results')
spike_filename = os.path.join('results', 'test_device_jy.avi')
save_paraDict = {'spike_h': height, 'spike_w': spike_width}

result_filename = 'sps100_test.txt'
tracking_file = os.path.join('results', result_filename)

device = torch.device('cuda')

calibration_time = 150
spike_tracker = SpikeSORT(height, spike_width, device)
spike_tracker.calibrate_motion(total_spikes, calibration_time)
spike_tracker.get_results(total_spikes[calibration_time:], tracking_file)
obtain_spike_video(np.array(spike_tracker.filterd_spikes), spike_filename, **save_paraDict)
# obtain_spike_video(total_spikes, spike_filename, **save_paraDict)
# video_filename = os.path.join('results', filename[-1] + '_mot.avi')
# obtain_mot_video(np.array(spike_tracker.filterd_spikes), video_filename, tracking_file, **save_paraDict)
