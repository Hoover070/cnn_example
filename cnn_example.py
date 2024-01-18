import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Model
from keras.models import Sequential
from keras.initializers import HeNormal
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# globals

SEED = 13
BATCH_SIZE = 32

# set random seed
tf.random.set_seed(SEED)
np.random.seed(SEED)

# plotting function
def plot_imgs(img):
    """plot 8x8 matrix of img
    
    Args: 
        img (RGB image): images from dataset
    """
    fig, ax = plt.subplots(8,8)
    k=0

    for i in range(8):
        for j in range(8):
            ax[i][j].set_axis_off()
            ax[i][j].imshow(img[k], aspect='auto')
            k+=1
    plt.show()


def plot_iter(obj, n=4):
    """plot generator images
    
    Args: 
        obj (iter): image iter
        n (int, optional): plot matrix size . Defaults to 4.
    
    """
    fig, ax, = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            batch = obj.next()
            image = batch[0].astype('float')
            ax[i][j].set_axis_off()
            ax[i][j].imshow(image, aspect='auto')
    plt.show()


def plot_scores(history):
    """plot accuracy and loss
    
    Args: 
        history(dict): contains info from training
    """

    # plot loss
    plt.subplot(211)
    plt.title('cross entropy loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()


# Load data
    

# preprocess data scale/flatten
    

# define model


