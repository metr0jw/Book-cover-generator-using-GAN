import tensorflow as tf
import keras as k
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=False)
print("Shape of the training image dataset is " + str(mnist.train.images.shape))
print("Shape of the training image label is " + str(mnist.train.images.shape))

index = np.random.choice(mnist.train.images.shape[0], 1)
random_image = mnist.train.images[index]
random_label = mnist.train.labels[index]
random_image = random_image.reshape([28, 28])

plt.gray()
plt.imshow(random_image)
plt.show()