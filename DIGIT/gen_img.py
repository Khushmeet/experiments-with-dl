import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

mnist = pd.read_csv('test.csv')
mnist = np.array(mnist, dtype='uint8')
for i in range(15):
    img = mnist[i, :]
    ex = img.reshape((28, 28))
    plt.title('Image')
    plt.imshow(ex, cmap='gray')
    plt.show()
