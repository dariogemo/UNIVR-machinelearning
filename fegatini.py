import numpy as np

def rounded_mean(array):
    mean = np.mean(array)
    return round(mean, 2)