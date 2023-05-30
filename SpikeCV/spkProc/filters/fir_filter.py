# -*- coding: utf-8 -*-

import torch
import numpy as np

def mean_filter(spk_data, win, stride=1, padding=0):
    '''
    Temporal averaging filter
    :param spk_data :[ ..., T, H, W]
    :param win: win size
    :param stride: stride
    :param padding: size of padding
    :return: filtered spike data
    '''
    if isinstance(spk_data, torch.Tensor):
        T = spk_data.shape[-3]
        H = spk_data.shape[-2]
        W = spk_data.shape[-1]
        if padding != 0:
            spk_data_padding = torch.zeros((*spk_data.shape[:-3], T + 2 * padding, H, W))
            spk_data_padding[..., padding:T+padding,:,:] = spk_data
        else:
            spk_data_padding = spk_data
        filtered_spk_data = torch.zeros((*spk_data.shape[:-3], (T + 2 * padding - win) // stride + 1, H, W))

        for i in range((T + 2 * padding - win) // stride + 1):
            filtered_spk_data[..., i,:,:] = spk_data_padding[..., i * stride:i * stride+win,:,:].mean(dim=-3)
    elif isinstance(spk_data, np.ndarray):
        T = spk_data.shape[-3]
        H = spk_data.shape[-2]
        W = spk_data.shape[-1]
        if padding != 0:
            spk_data_padding = np.zeros((*spk_data.shape[:-3], T + 2 * padding, H, W))
            spk_data_padding[..., padding:T + padding, :, :] = spk_data
        else:
            spk_data_padding = spk_data
        filtered_spk_data = np.zeros((*spk_data.shape[:-3], (T + 2 * padding - win) // stride + 1, H, W))

        for i in range((T + 2 * padding - win) // stride + 1):
            filtered_spk_data[..., i, :, :] = spk_data_padding[..., i * stride:i * stride + win, :, :].mean(axis=-3)
    else:
        raise TypeError('Expected torch.Tensor' +
                        ' but got {0}'.format(type(spk_data)))
    return filtered_spk_data


class MeanFilter(object):
    '''
    Temporal averaging filter
    '''
    def __init__(self, win, stride=1, padding=0):
        '''

        :param win: filter window
        :param stride: stride length
        :param padding: padding size
        '''
        self.win = win
        self.stride = stride
        self.padding = padding

    def __call__(self, spk_data):
        return mean_filter(spk_data, win=self.win, stride=self.stride, padding=self.padding)

    def __repr__(self):
        return self.__class__.__name__ + '(win={},stride={},padding={})'.format(self.win, self.stride, self.padding)