#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

## params
NUM_EPOCHS =  5

## callback class for each epoch of training
class myCallback(tf.keras.callbacks.Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        if logs.get('loss') < 0.1 :
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


## load data
fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()

## normalization
train_imgs = train_imgs / 255.0
test_imgs  = test_imgs  / 255.0

## show images
img_idx = 0
img_idx %= len(train_imgs)
display = False
if display :
    plt.imshow(train_imgs[img_idx])
    plt.show()
    print train_imgs[img_idx]
    print train_labels[img_idx]

## building model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), ## flatten 28 * 28 pixels imgs
    keras.layers.Dense(128, activation=tf.nn.relu),  ## hidden layer
    keras.layers.Dense(10, activation=tf.nn.softmax) ## 10 labels in y
    ])


## training
callback = myCallback()
model.compile(optimizer = "adam",
        loss="sparse_categorical_crossentropy", metrics= ["accuracy"])
model.fit(train_imgs, train_labels, epochs=NUM_EPOCHS, callbacks=[callback])


## evaluate
model.evaluate(test_imgs, test_labels)

## predict
classifications = model.predict(test_imgs)
test_img_idx = 0
print classifications[test_img_idx]

