
import numpy as np

def depth_centroid(depth_map):
    layers = np.arange(depth_map.shape[1])
    return (depth_map * layers).sum(axis=1) / (depth_map.sum(axis=1) + 1e-8)

def depth_spread(depth_map):
    layers = np.arange(depth_map.shape[1])
    centroid = depth_centroid(depth_map)
    return np.sqrt(((layers-centroid[:,None])**2 * depth_map).sum(axis=1))

def persistence_score(depth_map):
    return depth_map.sum(axis=1)

