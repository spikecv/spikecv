# -*- coding: utf-8 -*- 

import numpy as np
import torch
import cv2


class TFP:
    def __init__(self, spike_h, spike_w, device):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device


    def spikes2images(self, spikes, half_win_length):
        '''
        Texture From Playback (TFP) algorithm
        Convert the spikes as a whole into an image reconstructed by the TFP algorithm

        input：
        spikes: T x H x W numpy array, Data type: either integer or floating point
        max_search_half_window: For the time point to be converted into an image, the maximum number of spike frames
        for the left and right references, beyond this number will not be searched

        output：
        ImageMatrix: T' x H x W numpy array, where T' = T - (2 x max_search_half_window)
        data type: uint8, ranges: 0 ~ 255
        '''

        T = spikes.shape[0]
        T_im = T - 2*half_win_length

        if T_im < 0:
            raise ValueError('The length of spike stream {:d} is not enough for half window length {:d}'.format(T, half_win_length))
        
        spikes = torch.from_numpy(spikes).to(self.device).float()
        ImageMatrix = torch.zeros([T_im, self.spike_h, self.spike_w]).to(self.device)

        for ts in range(half_win_length, T-half_win_length):
            ImageMatrix[ts - half_win_length] = spikes[ts-half_win_length : ts+half_win_length+1].mean(dim=0) * 255

        ImageMatrix = ImageMatrix.cpu().detach().numpy().astype(np.uint8)

        return ImageMatrix


    def spikes2frame(self, spikes, key_ts, half_win_length):
        '''
        Texture From Playback (TFP) algorithm
        Get a frame of TFP image from spikes
        
        input：
        spikes: T x H x W numpy array, Data type: either integer or floating point
        key_ts: The image timestamp to reconstruct
        max_search_half_window: For the time point to be converted into an image, the maximum number of spike frames
        for the left and right references, beyond this number will not be searched

        output：
        Image: H x W numpy array, data type: uint8, ranges: 0 ~ 255
        '''

        T = spikes.shape[0]

        if (key_ts - half_win_length < 0) or (key_ts + half_win_length > T):
            raise ValueError('The length of spike stream {:d} is not enough for half window length {:d} at key time stamp {:d}'.format(T, half_win_length, key_ts))
        
        spikes = spikes[key_ts - half_win_length : key_ts + half_win_length + 1]
        spikes = torch.from_numpy(spikes).to(self.device).float()

        Image = spikes.mean(dim=0) * 255
        Image = Image.cpu().detach().numpy().astype(np.uint8)

        return Image
