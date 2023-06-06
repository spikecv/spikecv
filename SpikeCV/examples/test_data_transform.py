# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')
import SpikeCV.spkData.data_transform as transform
import numpy as np
import torch

ndarray_spike_matrix = np.random.randint(2, size=(100, 32, 32))

# np.ndarray -> torch.tensor
tensor_spike_matrix = transform.ToTorchTensor(type=torch.FloatTensor)(ndarray_spike_matrix)
print(tensor_spike_matrix.shape, type(tensor_spike_matrix))

# torch.tensor -> np.ndarray
ndarray_spike_matrix = transform.ToNPYArray(type=np.float)(tensor_spike_matrix)
print(ndarray_spike_matrix.shape, type(ndarray_spike_matrix))
