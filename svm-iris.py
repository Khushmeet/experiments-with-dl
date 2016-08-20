from sklearn.linear_model import LogisticRegression
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

# splitting data and labels into train and test sets for measuring performance of the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# loading the perceptron
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, Y_train)

# predicting the outputs
Y_pred = lr.predict(X_test_std)

# measuring the accuracy
print('Accuracy', accuracy_score(Y_test, Y_pred))

# we can predict the probabilities for test set using lr.predict_proba()

# visualizing the output
X_combined = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))
print(X_combined)
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X_combined, y=Y_combined, clf=lr, legend=2)
plt.title('Logistic Regression')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()