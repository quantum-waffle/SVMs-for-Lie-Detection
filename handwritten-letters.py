import math, random, numpy as np, tensorflow as tf 

"""
ACCURACY ON TEST: 0.9801
"""

# generar los sets de datos a partir de la libreria mnist
(training_data, training_labels),(testing_data,testing_labels) = tf.keras.datasets.mnist.load_data()

# construir los vectores de input de 28x28 pixels
training_data = training_data.reshape(60000, 784)
testing_data = testing_data.reshape(10000, 784)

# el tipo de dato da utilizar debe de ser float 
training_data = training_data.astype('float32')
testing_data = testing_data.astype('float32')

# normalizar la data de pixeles
training_data /= 255
testing_data /= 255

# KERAS (modelo de red neural)
print("BUILD MODEL")
model = tf.keras.models.Sequential()
# el espacio de input es de 512, 256*2. 784 es el shape del input, 28*28
model.add(tf.keras.layers.Dense(512, activation="relu", input_shape=(784, ))) # 20 por el log2 de 10000000
model.add(tf.keras.layers.Dense(512, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax")) # pudo haber sido solo 1, da lo mismo
model.add(tf.keras.layers.Flatten()) # reduce la dimensionalidad de la caapa de salida

print("COMPILING MODEL")
model.compile(
	loss=tf.keras.losses.sparse_categorical_crossentropy,
	optimizer=tf.keras.optimizers.Adadelta(), # mejora los pesos de adelante a atras
	#optimizer=tf.keras.optimizers.Adam(),
	metrics=["accuracy"] # output grafico
)

print("TRAINING MODEL")
model.fit(
	training_data,
	training_labels,
	epochs=8, 	
	verbose=1
)

print("EVALUAR")
(loss, accuracy) = model.evaluate(testing_data, testing_labels)
print("Test loss", loss)
print("Test accuracy", accuracy)