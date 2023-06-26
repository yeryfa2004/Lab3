import numpy as np

import tensorflow

from keras.layers import Input, Dense
from keras import Model
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam

import pandas as pd

train = pd.read_csv("mnist_train.csv").values
Y_train = train[:, 0]
X_train = train[:, 1:]

test = pd.read_csv("mnist_test.csv").values
Y_test = train[:, 0]
X_test = train[:, 1:]

X_train, X_test = (X_train / 255.0), (X_test / 255.0)

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

x = Input(shape=(784,))
h1 = Dense(64, activation="relu")(x)
h2 = Dense(64, activation="relu")(h1)
h3 = Dense(64, activation="relu")(h2)
output = Dense(10, activation="softmax")(h3)
model = Model(inputs=x, outputs=output)

opt = Adam(learning_rate=0.001)

model.compile(optimizer=opt,
              loss=sparse_categorical_crossentropy,
              metrics=[sparse_categorical_accuracy, sparse_categorical_crossentropy],
              )

bs = 64
n_epoch = 6

model.fit(X_train, Y_train, batch_size=bs, epochs=n_epoch, validation_data=(X_test, Y_test))

pdc = model.predict(X_test)

for real, predicted in zip(Y_test, pdc):
    max_index = np.argmax(predicted)

    print("Value {} was predicted as {}".format(real, max_index))
