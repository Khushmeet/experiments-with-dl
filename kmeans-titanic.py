import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
style.use('ggplot')

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

titanic_df = pd.read_excel('data/titanic.xls')
titanic_df.drop(['body'], 1, inplace=True)
titanic_df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df
titanic_df = handle_non_numerical_data(titanic_df)


fig = plt.figure()
p1 = plt.subplot2grid((2,2),(0,0))
titanic_df.survived.value_counts().plot(kind='bar', alpha=0.5)
p1.set_xlim(-1, 2)
plt.title("Distribution of Survival, (1 = Survived)")

p2 = plt.subplot2grid((2,2),(0,1))
plt.scatter(titanic_df.age, titanic_df.survived, alpha=0.2)
plt.ylabel("Age")
plt.grid(b=True, which='major', axis='y')
plt.title("Survival by Age,  (1 = Survived)")

plt.subplot2grid((2,2),(1,0), colspan=2)
titanic_df.age[titanic_df.pclass == 1].plot(kind='kde')
titanic_df.age[titanic_df.pclass == 2].plot(kind='kde')
titanic_df.age[titanic_df.pclass == 3].plot(kind='kde')
plt.xlabel("Age")
plt.title("Age Distribution within classes")
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')
plt.show()

X = np.array(titanic_df.drop(['survived'],1).astype(float))
Y = np.array(titanic_df['survived'])
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

X = scale(X)

k_means = KMeans(n_clusters=2)
k_means.fit(x_train)

print(titanic_df.head())

# k-means clusters
# visualization
clusters = k_means.cluster_centers_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_train_shuffle =shuffle(x_train)
ax.scatter(x_train_shuffle[:,3], x_train_shuffle[:,7], x_train_shuffle[:,9], c='r')
ax.scatter(clusters[0], clusters[1], c='b', marker='o')
plt.show()

correct = 0
for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    prediction = k_means.predict(predict)
    if prediction[0] == Y[i]:
        correct += 1

print('Accuracy:', correct/len(X))

# prediction
predicted = k_means.predict(x_test)
for i in range(len(x_test)):
    print('Predicted value:', predicted[i], '\t', 'Actual vaue:', y_test[i])
