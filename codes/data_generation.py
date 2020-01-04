import numpy as np
import matplotlib.pyplot as plt
import pickle

def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

x = np.arange(2 * 365 + 1)
series = autocorrelation(x, 10, seed=42)

noise_level = 2
noise = white_noise(x, noise_level, seed=42)
series += noise

with open('../data/generated_data.pickel', 'wb') as handle:
    pickle.dump(series, handle, protocol=pickle.HIGHEST_PROTOCOL)