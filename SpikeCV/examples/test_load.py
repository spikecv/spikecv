# -*- encoding: utf-8 -*-
import sys
sys.path.append("..")

import numpy as np
from spkData.load_dat import data_parameter_dict, SpikeStream
from utils import path
from pprint import pprint

data_filename1 = 'motVidarReal2020/spike59/'
label_type = 'tracking'
para_dict = data_parameter_dict(data_filename1, label_type)
pprint(para_dict)
# RESULT：
# {'filepath': '..\\spkData\\datasets\\motVidarReal2020\\spike59\\spikes.dat',
#  'labeled_data_dir': '..\\spkData\\datasets\\motVidarReal2020\\spike59\\spikes_gt.txt',
#  'labeled_data_suffix': 'txt',
#  'labeled_data_type': [4, 5],
#  'spike_h': 250,
#  'spike_w': 400}

data_filename = "recVidarReal2019/classA/car-100kmh"
label_type = 'raw'
paraDict = data_parameter_dict(data_filename, label_type)
pprint(paraDict)
# {'filepath': '..\\spkData\\datasets\\recVidarReal2019\\classA\\car-100kmh',
#  'spike_h': 250,
#  'spike_w': 400}
# initial VidarSpike object for format input data
vidarSpikes = SpikeStream(**paraDict)
# loading total spikes from dat file -- spatial resolution: 400 x 250, begin index: 500 total timestamp: 1500
block_len = 1500
spikes = vidarSpikes.get_block_spikes(begin_idx=500, block_len=block_len)

data_filename2 = 'Spike-Stero/indoor/left/0000/0000/'
label_type = 'stero_depth_estimation'
para_dict = data_parameter_dict(data_filename2, label_type)
pprint(para_dict)
# RESULT：
# {'filepath': '..\\spkData\\datasets\\Spike-Stero\\indoor\\left\\0000\\0000\\0000.dat',
#  'labeled_data_dir': '..\\spkData\\datasets\\Spike-Stero\\indoor\\left\\0000\\0000\\0000_gt.npy',
#  'labeled_data_suffix': 'npy',
#  'labeled_data_type': [3.2],
#  'spike_h': 250,
#  'spike_w': 400}

data_filename3 = 'PKU-Vidar-DVS/train/Vidar/00152_driving_outdoor3/1.dat'
label_type = 'detection'
para_dict = data_parameter_dict(data_filename3, label_type)
pprint(para_dict)
# RESULT：
# {'filepath': '..\\spkData\\datasets\\PKU-Vidar-DVS\\train\\Vidar\\00152_driving_outdoor3\\1.dat',
#  'labeled_data_dir': '..\\spkData\\datasets\\PKU-Vidar-DVS\\train\\labels\\00152_driving_outdoor3\\1.txt',
#  'labeled_data_suffix': 'txt',
#  'labeled_data_type': [4],
#  'spike_h': 250,
#  'spike_w': 400}
vidarSpikes = SpikeStream(**para_dict)
# loading total spikes from dat file -- spatial resolution: 400 x 250, total timestamp: 400
spikes = vidarSpikes.get_spike_matrix()
block_len = spikes.shape[2]
pprint(spikes.shape)

data_filename3 = 'Spike-Stero/indoor/left/0000/0000/spikes.dat'
pieces = path.split_path_into_pieces(data_filename3)
print(pieces)
# RESULT：
# ['Spike-Stero', 'indoor', 'left', '0000', '0000', 'spikes.dat']

data_filename3 = r'Spike-Stero\indoor\left\0000\0000\spikes.dat'
pieces = path.split_path_into_pieces(data_filename3)
print(pieces)
# RESULT：
# ['Spike-Stero', 'indoor', 'left', '0000', '0000', 'spikes.dat']

data_filename3 = 'Spike-Stero\\indoor\\left\\0000\\0000\\spikes.dat'
pieces = path.split_path_into_pieces(data_filename3)
print(pieces)
# RESULT：
# ['Spike-Stero', 'indoor', 'left', '0000', '0000', 'spikes.dat']