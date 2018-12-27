# Nonlinear SVM 
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

# PEP 484:
# https://www.python.org/dev/peps/pep-0484/
import pandas as pd, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split 

def main():

    ops.reset_default_graph()

    FEATURE_COLUMNS = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','edad','literalidad','escolaridad','trabaja','acomp_psico']
    LABEL_COLUMN = ['Class']
    CONTINUOUS_COLUMNS = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6',]
    CATEGORICAL_COLUMNS = ['edad','literalidad','escolaridad','trabaja','acomp_psico']
    # Declare main variables
    batch_size = 500
    #gamma_value = -25.050
    gamma_value = -2.750
    optimizer_value = 0.01
    """batch_size (int): Size of data frame for SVM model (preventing memory allocation error).
       gamma_value (float): Constant for gaussian rbf kernel.
       optimizer_value (vloat): Constant for loss function optimizer.
    """
    train_iterations = 5000
    test_iterations = 2500
    dataset_path = 'Training_data_psico/db.csv'
    
    dataset_data = load_dataset_data(dataset_path, db=True)
    tensors, variables, loss, accuracy, train_step, prediction = build_model(batch_size, gamma_value, optimizer_value)
    print("[!] Creating new Tensorflow(r) session...")
    with tf.Session() as sess:
        print("[ ] Initializing variables...")
        init = tf.global_variables_initializer()
        sess.run(init)
        train(dataset_data, tensors, train_step, batch_size, sess, loss, accuracy, train_iterations)
        test(dataset_data, tensors, batch_size, sess, accuracy, test_iterations)

        entry_file = load_dataset_data("Cross_validation_psico/E002.csv", entry_file=True)
        predict(entry_file, dataset_data, tensors, batch_size, sess, prediction)


def load_dataset_data(path, db=False, test_file=False, entry_file=False):
    """Funtion for loading data from csv files into numpy arrays.

    Args:
        path (String): path to csv file.
        db (bool): set True if loading a whole trainig dataset.
        test_file (bool): set True if loading a test file with attributes and labels.
        entry_file (bool): set True if laoding a entry file with just attributes (for prediction).

    Returns:
        np.array, np.array, np.array, np.array: (if db) test and train split of whole dataset. 
        np.array, np.array: (if test_file) attributes and labes of a test file.
        np.array: (if entry_file) just attributes of a entry file, meant for prediction. 
    """
    print("[ ] Importing dataset from {}".format(path))
    if db or test_file:
        colnames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','edad','literalidad','escolaridad','trabaja','acomp_psico','Class']
    elif entry_file:
        colnames = ['AF3','AF4','F7','F3','F4','F8','FC5','FC6','edad','literalidad','escolaridad','trabaja','acomp_psico']
    else:
        return -1

    # Reading the dataset csv
    data = pd.read_csv(path, sep='\t' ,header=None, names=colnames)
    print("[+] Dataset shape: {}".format(data.shape))
    print("[+] Dataset head:")
    # Show first rows of dataset
    print(data.head())
    
    if db or test_file:
        X = data.values[:, :13]
        y = data.values[:, 13]
    elif entry_file:
        return np.array(data)
    
    if db:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)  
        # print(X_train.shape)
        # print(X_test.shape)
        return X_train, X_test, y_train, y_test
    elif test_file:
        X_vals = np.array(X)
        y_vals = np.array(y)
        return X_vals, y_vals



def build_model(batch_size, gamma_value, optimizer_value):
    """SVM model builder.

    Args:
        batch_size (int): Size of data frame for SVM model (preventing memory allocation error).
        gamma_value (float): Constant for gaussian rbf kernel.
        optimizer_value (vloat): Constant for loss function optimizer.

    Returns:
        [tensor, tensor, tensor]: placeholders for model variables.
        [tf variable}: SVM rbf variables generated.
        tf function: loss tensorflow function
        tf function: accuracy tensorflow function
        tf function: train tensorflow function
        tf function: prediction tensorflow function
    """

    # Initialize placeholders
    print("[ ] Initializing placeholders")
    x_data = tf.placeholder( dtype=tf.float32)
    y_target = tf.placeholder( dtype=tf.float32)
    prediction_grid = tf.placeholder(dtype=tf.float32)

    # Create variables for svm
    b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

    # Gaussian (RBF) kernel
    print("[-] Setting gamma constant to {}".format(gamma_value))
    gamma = tf.constant(gamma_value)
    sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Compute SVM Model
    print("[-] Computing SVM model...")
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
    loss = tf.negative(tf.subtract(first_term, second_term))

    # Gaussian (RBF) prediction kernel
    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

    prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
    prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

    # Declare optimizer
    print("[-] Setting optimizer value to {}".format(optimizer_value))
    my_opt = tf.train.GradientDescentOptimizer(optimizer_value)
    train_step = my_opt.minimize(loss)
    return [x_data, y_target, prediction_grid], [b], loss, accuracy, train_step, prediction


def train(data, tensors, train_step, batch_size, sess, loss, accuracy, train_iterations, steps=500):
    X_train = data[0]
    y_train = data[2] 
    x_data, y_target, prediction_grid = tensors
    loss_vec = []
    batch_accuracy = []
    print("[!] +++++++++++++++++++ Starting trainig phase +++++++++++++++++++")
    for i in range(train_iterations):
        # get a sampĺe from the trainig data the size of batch_size, so memory allocation error is avoided
        rand_index = np.random.choice(len(X_train), size=batch_size)
        rX = X_train[rand_index]
        rY = np.transpose([y_train[rand_index]])
        
        # train the model
        sess.run(train_step, feed_dict={x_data: rX, y_target: rY})
        
        # get loss fuction value
        temp_loss = sess.run(loss, feed_dict={x_data: rX, y_target: rY})
        loss_vec.append(temp_loss)
        
        # get accuracy from current train
        acc_temp = sess.run(accuracy, feed_dict={x_data: rX, y_target: rY, prediction_grid: rX})
        batch_accuracy.append(acc_temp)
        
        if (i + 1) % steps == 0:
            print("[ ] Step #{} -> accuracy: {}".format(i + 1, acc_temp))
    
    print("[+] Final accuracy: {}; Average accuracy: {}".format(batch_accuracy[-1], np.mean(batch_accuracy)))
    plot_accuracy_graph(batch_accuracy)


def test(data, tensors, batch_size, sess, accuracy, test_iterations, steps=500):
    X_test = data[1]
    y_test = data[3]
    x_data, y_target, prediction_grid = tensors
    test_accuracy = []
    print("[!] +++++++++++++++++++ Starting testing phase +++++++++++++++++++")
    for i in range(test_iterations):
        # get a sampĺe from the trainig data the size of batch_size, so memory allocation error is avoided
        rand_index = np.random.choice(len(X_test), size=batch_size)
        rX = X_test[rand_index]
        rY = np.transpose([y_test[rand_index]])
        
        # get accuracy from current train
        acc_temp = sess.run(accuracy, feed_dict={x_data: rX, y_target: rY, prediction_grid: rX})
        test_accuracy.append(acc_temp)
        
        if (i + 1) % steps == 0:
            print("[ ] Step #{} -> accuracy: {}".format(i + 1, acc_temp))
    
    print("[+] Final accuracy: {}; Average accuracy: {}".format(test_accuracy[-1], np.mean(test_accuracy)))


def predict(data, train_data, tensors, batch_size, sess, prediction, iterations=5000):
    accumulation = []
    print("[!] ---------------- Predicting (may take a while)...")
    for i in range(iterations):
        X_train = train_data[0]
        y_train = train_data[2] 
        x_data, y_target, prediction_grid = tensors

        rand_index = np.random.choice(len(X_train), size=batch_size)
        rX = X_train[rand_index]
        rY = np.transpose([y_train[rand_index]])

        rand_index = np.random.choice(len(data), size=batch_size)
        rZ = data[rand_index]
        #[grid_predictions] = sess.run(prediction, feed_dict={x_data: rX, y_target: rY, prediction_grid: data[0]})
        [grid_predictions] = sess.run(prediction, feed_dict={x_data: rX, y_target:rY, prediction_grid:rZ})
        accumulation.append(int(np.ceil(np.abs(np.mean(grid_predictions)))))
    print("[+] Average prediction (0 for Truth, 1 for Lie): {}".format(int(np.mean(accumulation))))


def plot_accuracy_graph(_array):
    plt.plot(_array, 'k-', label='Accuracy')
    plt.title('Batch Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def random_state_generator(iterations):
    return [i for x in range(iterations)]
        

if __name__ == "__main__":
    main()
