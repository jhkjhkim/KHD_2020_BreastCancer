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


def build_xception():
    base_model = tf.keras.applications.Xception(include_top=False, weights=None, input_shape=(299, 299, 3))
    base_model.trainable = True

    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(299, 299),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, interpolation='nearest'),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, kernel_initializer='he_normal', activation='sigmoid')])
    return model


def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    r = tp / (tp + fn + K.epsilon())
    r = tf.where(tf.math.is_nan(r), tf.zeros_like(r), r)
    return K.mean(r)


def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    p = tf.where(tf.math.is_nan(p), tf.zeros_like(p), p)

    return K.mean(p)


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
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
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    sp = tn / (tn + fp + K.epsilon())
    sp = tf.where(tf.math.is_nan(sp), tf.zeros_like(sp), sp)

    return K.mean(sp)


def ntv(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    ntv = tn / (tn + fn + K.epsilon())
    ntv = tf.where(tf.math.is_nan(ntv), tf.zeros_like(ntv), ntv)

    return K.mean(ntv)


def custom(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    a = (tp + tn) / (tp + tn + fp + tn + K.epsilon())
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

    a = (tp + tn) / (tp + tn + fp + tn + K.epsilon())
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