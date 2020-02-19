import numpy as np

def quantize(x):
    bins = list(range(0, 256, 32)) + [255]
    return np.digitize(x, bins, False) - 1