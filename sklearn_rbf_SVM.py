import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  


# Import Dataset
#colnames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','Class']
colnames = ['AF3','AF4','F3','F4','Class']
data = pd.read_csv('Training_data/db.csv',sep='\t' ,header=None, names=colnames)
print(data.shape)
print(data.head())
#X = data.values[:, :4]
#y = data.values[:, 4]
X = data.values[:, :4]
y = data.values[:, 4]
x_vals = np.array(X)
y_vals = np.array(y)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.svm import SVC  
#svclassifier = SVC(kernel='linear')  
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  