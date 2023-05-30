# -*- coding: utf-8 -*- 

import torch
import numpy as np

def compute_aee(flow_gt, flow_pred):
    EE = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    EE = torch.from_numpy(EE)

    if torch.sum(EE) == 0:
        AEE = 0
    else:
        AEE = torch.mean(EE)

    return AEE