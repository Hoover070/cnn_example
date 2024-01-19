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
STRIDES = (2,2)


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

def load_cifar():
    # load cifar10 dataset
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return xtrain, ytrain, xtest, ytest

# preprocess data scale/flatten
def preprocess(xtrain, ytrain, xtest, ytest):
    xtrain, xtest = xtrain / 255.00, xtest / 255.00
    ytrain, ytest = ytrain.flatten(), ytest.flatten()
    return xtrain, ytrain, xtest, ytest

# define model
def define_model(xtrain, ytrain):
    # number of classes
    k = len(set(ytrain))
    print('number of classes', k)

    # build network structure
    # input
    i = Input(shape=xtrain[0].shape)
    # conv_1 (went from 32 to 64 neurons)
    x = Conv2D(64, (3,3), strides=1,  activation='relu', padding='same')(i)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), strides=STRIDES, activation='relu', padding='same')(x) # changed from MaxPooling2D to Conv2D
    x = BatchNormalization()(x) # added another BatchNormalization because thats what I understood needed to happen when using Conv2D instead of MaxPooling2D from the lecture
    
    # conv_2 (went from 64 to 128 neurons)
    x = Conv2D(128, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), strides=STRIDES, activation='relu', padding='same')(x) # changed from MaxPooling2D to Conv2D
    x = BatchNormalization()(x) # added another BatchNormalization because thats what I understood needed to happen when using Conv2D instead of MaxPooling2D from the lecture

    # conv_3 (went from 128 to 256 neurons)
    x = Conv2D(256, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), strides=1, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), strides=STRIDES, activation='relu', padding='same')(x) # changed from MaxPooling2D to Conv2D
    x = BatchNormalization()(x) # added another BatchNormalization because thats what I understood needed to happen when using Conv2D instead of MaxPooling2D from the lecture

    # Dense Layer (FC)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x) # changed from 1024 to 512
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x) # added another Dense layer at 256 neurons
    x = Dropout(0.2)(x)
    # softmax
    x = Dense(10, activation='softmax')(x) # changed from k to 10 neurons

    model = Model(i, x)

    return model

# compile
def compile_model(model):
    # compile model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# augment training data
def data_aug():
    data_generator = ImageDataGenerator(width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=[0.90, 1.1],
                                        rotation_range=10,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
    return data_generator


def main():
    
    # harness
    X_train, y_train, X_test, y_test = load_cifar()
    X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)
    model = define_model(X_train, y_train)
    compile_model(model)
    #model.summary()


    train_gen = data_aug()
    '''data = np.expand_dims(X_train[60], axis=0)
    print(data.shape)
    train_generator = train_gen.flow(data, batch_size=1)
    plot_iter(train_generator)
    plot_imgs(X_train)'''

    # fit model
    train_generator = train_gen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    steps_per_epoch = X_train.shape[0] // BATCH_SIZE
    history = model.fit(train_generator, validation_data=[X_test, y_test],
                        steps_per_epoch=steps_per_epoch,
                        epochs=50) # change to 50 after verfiying code works

    # plot learning curves
    plot_scores(history)


    # test accuracy
    _, acc = model.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()
