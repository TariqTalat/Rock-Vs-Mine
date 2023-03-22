# importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#importing the Data
#getting the Dataset
dataset = pd.read_csv('Copy of sonar data.csv',header=None)
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:, -1].values
describe=dataset.describe()
counts_of60th=dataset[60].value_counts()
#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,stratify=y, random_state = 0)

#the model prediction
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Model evalution accuracy
model_prdicition=classifier.predict(X_train)
data_accuracy=accuracy_score(model_prdicition, y_train)
#predicting
y_pred = classifier.predict(X_test)
test_data=accuracy_score(y_pred, y_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

