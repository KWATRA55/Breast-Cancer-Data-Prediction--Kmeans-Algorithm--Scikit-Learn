import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score


bc= datasets.load_breast_cancer()

x= bc.data
y= bc.target

model= KMeans(n_clusters=2, random_state=0)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

model.fit(x_train)

predictions= model.predict(x_test)
labels= model.labels_
print(predictions)
print("accuracy : ", accuracy_score(y_test, predictions))
print(y_test)

print(pd.crosstab(y_train,labels))