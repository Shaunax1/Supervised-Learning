import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("column_2C_weka.csv")

data.head(15)

data.rename(columns={"class": "normality"}, inplace=True)

data.normality = [1 if each == "Normal" else 0 for each in data.normality]

y = data.normality.values
x_data = data.drop(["normality"],axis=1)
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())    

from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=42)

#%% KNN MODEL
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("{} score :  {}" .format(3,knn.score(x_test,y_test)))

score_list = []
for each in range(1,150):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,150),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

#%% Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

print("accuracy of nb algorithym : " , nb.score(x_test,y_test))

#%% Super Vector Machine Algorithm
from sklearn.svm import SVC
svm = SVC(random_state = 1 )
svm.fit(x_train,y_train)

print("accuracy of svm algorithym : " , svm.score(x_test,y_test))

#%% Confusion Matrix

y_pred = svm.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

import seaborn as sns
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True, linewidths=0.5, linecolor="red",fmt= ".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

