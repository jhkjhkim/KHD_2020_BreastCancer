import tensorflow as tf # Tensorflow 2
from tensorflow.keras.applications import Xception

def Xception():
    conv_base = Xception(weights='imagenet',
                         include_top = False)
    model = tf.keras.models.Sequential()
    model.add(conv_base)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model