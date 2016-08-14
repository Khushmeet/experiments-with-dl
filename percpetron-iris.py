from sklearn.linear_model import Perceptron
from mlxtend.evaluate import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# loading iris datasets
iris = load_iris()
X = iris.data[:, [2,3]]    # features
Y = iris.target            # labels
print(X)
print(Y)

# splitting data and labels into train and test sets for measuring performance of the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# loading the perceptron
pcp = Perceptron(n_iter=30, eta0=0.01, random_state=0)
pcp.fit(X_train_std, Y_train)

# predicting the outputs
Y_pred = pcp.predict(X_test_std)

# measuring the accuracy
print('Accuracy', accuracy_score(Y_test, Y_pred))

# visualizing the output
X_combined = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X_combined, y=Y_combined, clf=pcp, legend=2)
plt.title('Perceptron')
plt.show()