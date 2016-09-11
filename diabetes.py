from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# using linear regression
diabetes = load_diabetes()
print(diabetes.data, diabetes.target)

x = diabetes.data[:, np.newaxis, 2]

xtrain, xtest, ytrain, ytest = train_test_split(x, diabetes.target , test_size=0.1)

lrg = LinearRegression()
lrg.fit(xtrain, ytrain)

plt.figure()
plt.scatter(xtrain, ytrain, c='r')
plt.plot(xtest, lrg.predict(xtest), c='b', marker='o')
plt.show()

print('Accuracy:', accuracy_score(ytest, lrg.predict(xtest)))

