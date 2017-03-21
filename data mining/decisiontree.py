from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# loading dataset
iris = load_iris()
x = iris.data[:,[2,3]]
y = iris.target

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)

# using entropy for impurity
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(xtrain, ytrain)

predict = dtree.predict(iris.data[:5,[2,3]])
print('Predicted', predict)


# tree visualization using graphviz
export_graphviz(dtree, out_file='dtree.dot', filled=True, rounded=True, feature_names=['petal length', 'petal width'])
# convert .dot to png using
# dot -Tpng dtree.dot -o dtree.png