# -*- coding: utf-8 -*- 
import os
import torch
import sys
sys.path.append("..")
from pprint import pprint

from spkData.load_dat import data_parameter_dict
from spkData.load_dat import SpikeStream
from spkProc.tracking.spike_sort import SpikeSORT
from utils import path

from metrics.tracking_mot import TrackingMetrics
from visualization.get_video import obtain_mot_video

data_filename = "motVidarReal2020/spike59"
label_type = 'tracking'

# TODO: download the dataset to the ../spkData/datasets and unzip

paraDict = data_parameter_dict(data_filename, label_type)
pprint(paraDict)

# initial SpikeStream object for format input data
vidarSpikes = SpikeStream(**paraDict)

block_len = 1000
spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len)

device = torch.device('cuda')

calibration_time = 150
filename = path.split_path_into_pieces(data_filename)
result_filename = filename[-1] + '_spikeSort.txt'
if not os.path.exists('results'):
    os.makedirs('results')
tracking_file = os.path.join('results', result_filename)
spike_tracker = SpikeSORT(spikes, paraDict.get('spike_h'), paraDict.get('spike_w'), device)

# using stp filter to filter out static spikes
spike_tracker.calibrate_motion(calibration_time)
# start tracking
spike_tracker.get_results(tracking_file)

# measure the multi-object tracking performance
metrics = TrackingMetrics(tracking_file, **paraDict)
metrics.get_results()

# visualize the tracking results to a video
video_filename = os.path.join('results', filename[-1] + '_mot.avi')
#obtain_mot_video(spike_tracker.filterd_spikes, video_filename, tracking_file, **paraDict)
obtain_mot_video(spikes, video_filename, tracking_file, **paraDict)

