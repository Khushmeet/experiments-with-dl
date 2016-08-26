import numpy as np

def xor_dataset():
    x_xor = np.random.randn(300,2)
    y_xor = np.logical_xor(x_xor[:,0] > 0, x_xor[:,1] > 0)
    y_xor = np.where(y_xor,1,-1)
    return x_xor, y_xor