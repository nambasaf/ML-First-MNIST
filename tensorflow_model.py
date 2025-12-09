import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# step 1: always load the data first
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model_tf = models.Sequential([
    layers.Flatten(input_shape=(28, 28)), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(10)
])

model_tf.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model_tf.fit(x_train, y_train, epochs=3)
model_tf.evaluate(x_test, y_test, verbose=2)

