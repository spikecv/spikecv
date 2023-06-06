# -*- coding: utf-8 -*- 
import os, sys
sys.path.append("..")
from visualization.get_video import obtain_detection_video, vis_trajectory

filename = 'Vidar'
trajectories_filename = os.path.join('results', filename + '.json')
result_filename = filename + '_spikeSort.txt'
tracking_file = os.path.join('results', result_filename)

para_dict = {
    'spike_h': 250,
    'spike_w': 400
}

res_filename = 'vidar.png'
vis_trajectory(tracking_file, trajectories_filename, res_filename, **para_dict)


