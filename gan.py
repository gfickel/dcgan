import random

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Dense, BatchNormalization, Conv2DTranspose,
        Reshape, Conv2D, Flatten, LeakyReLU, ReLU)

seed = 42
num_dim = 2
project_dim = 64

# Generator
inputs_gen = keras.Input(shape=(num_dim,))
x = Dense(project_dim*7*7, activation='tanh')(inputs_gen)
x = BatchNormalization()(x)
x = Reshape((7,7,project_dim))(x)
x = Conv2DTranspose(project_dim/2, kernel_size=5, strides=(2,2), padding='same')(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(1, kernel_size=5, strides=(2,2), padding='same', activation='tanh')(x)
out_gen = keras.layers.Reshape((28,28))(x)
gen = keras.Model(inputs=inputs_gen, outputs=out_gen, name='generator')

# Discriminator
inputs_disc = keras.Input(shape=(28,28,1))
x = Conv2D(16, (5,5))(inputs_disc)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3), strides=2, padding='same')(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), strides=2, padding='same')(x)
x = LeakyReLU()(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)
disc = keras.Model(inputs=inputs_disc, outputs=x, name='discriminant')
disc.compile(
        optimizer=keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999),
        loss='binary_crossentropy',
    )

# GAN
for layer in disc.layers:
    layer.trainable = False

input_vec = Input((num_dim,), name='input')
output = disc(gen(input_vec))
gan = Model(input_vec, outputs=output, name='gan')
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, beta_1=0.5, beta_2=0.999))

print(gen.summary())
print(disc.summary())
print(gan.summary())


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32")/127.5-1

steps = 10000
batch_size = 128
gan_loss = 0

for i in range(steps):
    idxs = random.sample(range(0,len(x_train)), batch_size)
    images_train = x_train[np.array(idxs), :, :]

    noise = np.random.uniform(-1.0, 1.0, size=(batch_size, num_dim))
    gan_imgs = gen.predict(noise).reshape((batch_size,28,28))
    
    x = np.concatenate((images_train, gan_imgs))
    y = np.zeros((batch_size*2, 1))
    y[:batch_size] = 1
    y[batch_size:] = 0

    disc_loss = disc.train_on_batch(x, y)

    y = np.ones((batch_size, 1))
    noise = np.random.uniform(-1.0, 1.0, size=(batch_size, num_dim))
    gan_loss = gan.train_on_batch(noise, y)
    print('{} - {:.06f} - {:.06f}'.format(i, disc_loss, gan_loss))

    example = (gan_imgs[0,:,:]+1)*127.5
    example = example.astype(np.uint8)
    if i%100 == 0:
        cv2.imwrite('out_{:06d}.png'.format(i), example)
