
import numpy as np
from config import *

def local_diffusion(x, strength):
    kernel = np.array([[0,1,0],[1,4,1],[0,1,0]], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(x, 1, mode='edge')
    out = np.zeros_like(x)
    for i in range(H):
        for j in range(W):
            out[i,j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    return (1-strength)*x + strength*out

def depth_params(l):
    leak = BASE_LEAK + LEAK_K * (l / (LAYERS-1))
    gain = BASE_GAIN * np.exp(-l / GAIN_TAU)
    diff = DIFF_START + (DIFF_END - DIFF_START) * (l / (LAYERS-1))
    return leak, gain, diff
