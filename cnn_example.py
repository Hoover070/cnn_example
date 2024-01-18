import numpy as numpy
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

