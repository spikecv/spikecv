import numpy as np


def img_to_spike(img, gain_amp=0.5, v_th=1.0, n_timestep=100):
    '''
    Spike Simulator: Image to Spikes

    :param img: image numpy.ndarray size：h x w
    :param gain_amp: gain coefficient
    :param v_th: threshold for generate spikes
    :param n_timestep: timestamps
    :return: spike streams numpy.ndarray
    '''

    h, w = img.shape
    if img.max() > 1:
        img = img / 255.
    assert img.max() <= 1.0 and img.min() >= 0.0
    mem = np.zeros_like(img)
    spks = np.zeros((n_timestep, h, w))
    for t in range(n_timestep):
        mem += img * gain_amp
        spk = (mem >= v_th)
        mem = mem * (1 - spk)
        spks[t, :, :] = spk
    return spks.astype(np.float)

