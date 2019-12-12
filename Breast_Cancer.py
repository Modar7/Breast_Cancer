#Classification of cancer dignosis
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r'D:\4\Datasets\Breast_Cancer2.csv',header=None)
X1 = dataset.iloc[:, 0].values
X2 = dataset.iloc[:, 2:32].values

X = np.column_stack((X1, X2))

Y = dataset.iloc[:, 1].values

dataset.head()

print("Cancer data set dimensions : {}".format(dataset.shape))


dataset.isnull().sum()
dataset.isna().sum()

dataframe = pd.DataFrame(Y)
#Encoding categorical data values 
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


#predicting the Test set results
Y_pred = classifier.predict(X_test)

#Creating the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
c = print(cm[0, 0] + cm[1, 1])

print(cm)


from sklearn.metrics import classification_report
from sklearn.metrics import  accuracy_score


print(classification_report(Y_test,Y_pred))
print("Accuracy = ", accuracy_score(Y_test, Y_pred))











