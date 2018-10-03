import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.contrib.learn.python.learn.estimators import svm 


model_dir = '../svm_model_dir'

FEATURE_COLUMNS = [
    # age, age_buckets, class_of_worker, detailed_industry_recode,
    'AF3',
    'AF4',
    'F3',
    'F4'
]

LABEL_COLUMN = 'Class'

CONTINUOUS_COLUMNS = [0,1,2,3]

CATEGORICAL_COLUMNS = []

def input_fn(df):
    # continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    continuous_cols = {
        k: tf.expand_dims(tf.constant(df[k].values), 1)
        for k in CONTINUOUS_COLUMNS
    }
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS
    }
    feature_cols = dict(continuous_cols.items())
    feature_cols['example_id'] = tf.constant(
        [str(i + 1) for i in range(df['AF3'].size)])
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label

def train_input_fn(data):
    return input_fn(data)


def eval_input_fn():
	return input_fn(data)

# Import Dataset
#colnames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','Class']
colnames = ['AF3','AF4','F3','F4','Class']
data = pd.read_csv('Training_data/db.csv',sep='\t' ,header=None, names=colnames)
print(data.shape)
print(data.head())
#X = data.values[:, :8]
#y = data.values[:, 8]
X = data.values[:, :4]
y = data.values[:, 4]
#x_vals = np.array(X)
#y_vals = np.array(y)
data1 = np.array(data)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

# example_id = numpy.array(['%d' % i for i in range(len(y_train))])

# x_column_name = 'x'
# example_id_column_name = 'example_id'

# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x=X_train,
#     y=y_train,
#     num_epochs=None,
#     shuffle=True)

# svm = tf.contrib.learn.SVM(
#     example_id_column=example_id_column_name,
#     feature_columns=(tf.contrib.layers.real_valued_column(
#         column_name=x_column_name, dimension=128),),
#     l2_regularization=0.1)

# svm.fit(input_fn=train_input_fn, steps=10)

# # Use the trained model to predict on the test data
# predictions = list(svm.predict(X_test, as_iterable=True))
# score = metrics.accuracy_score(y_test, predictions)

model = svm.SVM(example_id_column='example_id',
                feature_columns=FEATURE_COLUMNS,
                model_dir=model_dir)
model.fit(input_fn=train_input_fn(data1), steps=10)
#results = model.evaluate(input_fn=eval_input_fn, steps=1)

predictions = list(model.predict(X_test, as_iterable=True))
score = metrics.accuracy_score(y_test, predictions)