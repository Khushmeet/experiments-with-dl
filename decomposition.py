from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
print(cancer.data.shape)

# using principal component analysis
k_pca = PCA(n_components=2)
xkpca = k_pca.fit_transform(cancer.data)
print(xkpca.shape)

# visualizing data
plt.figure()
plt.scatter(xkpca[:,0], xkpca[:,1])
plt.show()