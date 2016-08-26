from sklearn.svm import SVC
from mlxtend.evaluate import plot_decision_regions
import matplotlib.pyplot as plt
from helpers.xor_dataset import xor_dataset

# creating xor datasets
x_xor, y_xor = xor_dataset()
print(y_xor)

# loading the SVM
# gamma controls overfitting
svm = SVC(C=10.0, random_state=0, gamma=0.2, kernel='rbf')
svm.fit(x_xor, y_xor)

# visualizing the output
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=x_xor, y=y_xor, clf=svm, legend=2)
plt.title('RBF SVM')
plt.legend(loc='upper left')
plt.show()