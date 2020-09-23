import tensorflow as tf  # Tensorflow 2.2.
import tensorflow.keras.backend as K


def cnn():
    base_model = tf.keras.applications.Xception(include_top=False, weights=None, input_shape=(299, 299, 3))
    base_model.trainable = True

    model = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.Resizing(299, 299),
        # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, interpolation='nearest'),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, kernel_initializer='he_normal', activation='sigmoid')])
    return model

def cnn_base():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5),
                                    activation='relu',
                                    kernel_initializer='he_normal',
                                    input_shape=(299, 299, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(rate = 0.2))

    model.add(tf.keras.layers.Conv2D(16, (3, 3),
                                    kernel_initializer='he_normal',
                                    activation='relu'))

    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(64,
    #                                kernel_initializer='he_normal',
    #                                activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    return model


def build_resnet50():
    base_model = tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(299, 299, 3))
    base_model.trainable = True

    model = tf.keras.Sequential([
        # tf.keras.layers.experimental.preprocessing.Resizing(299, 299),
        # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, interpolation='nearest'),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')])
    return model


def build_xception():
    base_model = tf.keras.applications.Xception(include_top=False, weights=None, input_shape=(299, 299, 3))
    base_model.trainable = True

    model = tf.keras.Sequential([
      #  tf.keras.layers.experimental.preprocessing.Resizing(299, 299),
      #  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      #  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, interpolation='nearest'),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')])
    print("---------xceptionnet is loaded---------")
    return model

def build_inception():
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(299, 299, 3))
    base_model.trainable = True

    model = tf.keras.Sequential([
      #  tf.keras.layers.experimental.preprocessing.Resizing(299, 299),
      #  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      #  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, interpolation='nearest'),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')])
    print("---------inception is loaded---------")
    return model


def recall(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_true = K.cast(y_true, 'uint8')
    y_pred = K.cast(K.argmax(y_pred), 'uint8')
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    r = tp / (tp + fn + K.epsilon())
    r = tf.where(tf.math.is_nan(r), tf.zeros_like(r), r)
    return K.mean(r)


def precision(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_true = K.cast(y_true, 'uint8')
    y_pred = K.cast(K.argmax(y_pred), 'uint8')
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    p = tf.where(tf.math.is_nan(p), tf.zeros_like(p), p)

    return K.mean(p)


def f1(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_true = K.cast(y_true, 'uint8')
    y_pred = K.cast(K.argmax(y_pred), 'uint8')
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)


def sp(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_true = K.cast(y_true, 'uint8')
    y_pred = K.cast(K.argmax(y_pred), 'uint8')
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    sp = tn / (tn + fp + K.epsilon())
    sp = tf.where(tf.math.is_nan(sp), tf.zeros_like(sp), sp)

    return K.mean(sp)


def ntv(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_true = K.cast(y_true, 'uint8')
    y_pred = K.cast(K.argmax(y_pred), 'uint8')
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    ntv = tn / (tn + fn + K.epsilon())
    ntv = tf.where(tf.math.is_nan(ntv), tf.zeros_like(ntv), ntv)

    return K.mean(ntv)


def custom(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_true = K.cast(y_true, 'uint8')
    y_pred = K.cast(K.argmax(y_pred), 'uint8')
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    a = (tp + tn) / (tp + fn + fp + tn + K.epsilon())
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    sp = tn / (tn + fp + K.epsilon())
    ntv = tn / (tn + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())

    a = tf.where(tf.math.is_nan(a), tf.zeros_like(a), a)
    p = tf.where(tf.math.is_nan(p), tf.zeros_like(p), p)
    r = tf.where(tf.math.is_nan(r), tf.zeros_like(r), r)
    sp = tf.where(tf.math.is_nan(sp), tf.zeros_like(sp), sp)
    ntv = tf.where(tf.math.is_nan(ntv), tf.zeros_like(ntv), ntv)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean((f1 + a + p + r + sp + ntv) / 6)


def cust_loss_function(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    a = (tp + tn) / (tp + fn + fp + tn + K.epsilon())
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    sp = tn / (tn + fp + K.epsilon())
    ntv = tn / (tn + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())

    a = tf.where(tf.math.is_nan(a), tf.zeros_like(a), a)
    p = tf.where(tf.math.is_nan(p), tf.zeros_like(p), p)
    r = tf.where(tf.math.is_nan(r), tf.zeros_like(r), r)
    sp = tf.where(tf.math.is_nan(sp), tf.zeros_like(sp), sp)
    ntv = tf.where(tf.math.is_nan(ntv), tf.zeros_like(ntv), ntv)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean((f1 + a + p + r + sp + ntv) / 6)