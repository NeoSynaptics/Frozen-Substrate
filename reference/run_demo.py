
import numpy as np
import matplotlib.pyplot as plt
from substrate import *
from config import *

A = np.zeros((LAYERS,H,W))
baseline = np.zeros_like(A)

def inject(t):
    x = np.zeros((H,W))
    cx = int(H/2 + 4*np.sin(t/30))
    cy = int(W/2 + 4*np.cos(t/30))
    x[cx-1:cx+2, cy-1:cy+2] = 1.0
    return x

depth_map = []

for t in range(TIMESTEPS):
    A[0] = inject(t)
    for l in range(LAYERS):
        leak, gain, diff = depth_params(l)
        A[l] = local_diffusion(A[l], diff)
        A[l] *= (1 - leak)
        if l < LAYERS-1:
            A[l+1] += gain * A[l]
        baseline[l] = (1-BASELINE_ALPHA)*baseline[l] + BASELINE_ALPHA*A[l]
    R = np.abs(A - baseline)
    depth_map.append(R.mean(axis=(1,2)))
    if t < TIMESTEPS-1:
        A[1:] *= 0

plt.imshow(np.array(depth_map).T, aspect='auto')
plt.title("Depth Map (Mean Residual)")
plt.xlabel("Time")
plt.ylabel("Layer")
plt.colorbar()
plt.show()
