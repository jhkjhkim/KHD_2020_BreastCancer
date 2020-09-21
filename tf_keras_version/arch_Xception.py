import tensorflow as tf # Tensorflow 2
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import csv
import sys
import itertools

def build_xception():
    base_model = Xception(include_top = False, weights='imagenet', input_shape = (299,299,3))
    base_model.trainable = True
    
    base_model.outputs = [base_model.layers[-1].output]
    last = base_model.outputs[0]
    x = GlobalAveragePooling2D()(last)
    preds = Dense(1, activation='sigmoid')(x)
    
    model = Model(base_model.input, preds)

    return model

def build_train_generator(train_dir, target_image_size, train_batch_size):
    datagen = ImageDataGenerator(horizontal_flip=True,
                                vertical_flip=True,
                                rescale = 1./255)
    
    train_generator = datagen.flow_from_directory(train_dir,
                                                  target_size = target_image_size, 
                                                  batch_size = train_batch_size,
                                                 class_mode = 'binary')
    
    
    return train_generator