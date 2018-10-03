#!/usr/bin/python3  

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.svm import SVC
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))
#(training_data, training_labels),(testing_data,testing_labels) = tf.keras.datasets.mnist.load_data()

# Import Dataset
data = pd.read_csv('Training_data/full_data.csv',sep='\t' ,header=None)
X = data.values[:, :8]
y = data.values[:, 8]

# A function to draw hyperplane and the margin of SVM classifier
def draw_svm(X, y, C=1.0):
	clf = SVC(kernel='linear', C=C)
	clf_fit = clf.fit(X, y)
	return clf_fit

clf_arr = []
#clf_arr.append(draw_svm(X, y, 0.0001))
#clf_arr.append(draw_svm(X, y, 0.001))
clf_arr.append(draw_svm(X, y, 1))
#clf_arr.append(draw_svm(X, y, 10))

for i, clf in enumerate(clf_arr):
    # Accuracy Score
    print(clf.score(X, y))
    #pred = clf.predict([(12, 32), (-250, 32), (120, 43)])
    #print(pred)
