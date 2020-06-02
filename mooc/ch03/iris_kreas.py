import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

# 告知要喂入网络的训练集和测试集是什么
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
np.random.seed(116)

model = tf.keras.Sequential([tf.keras.layers.Dense(3, activation="softmax",
                                                   kernel_regularizer=tf.keras.regularizers.l2())])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
model.summary()