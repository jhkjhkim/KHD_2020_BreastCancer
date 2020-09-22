# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:41:32 2020

@author: user
"""

import tensorflow as tf
import keras.backend as K

def custom_loss(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    
    a = (tp + tn) / (tp+tn+fp+tn+K.epsilon())
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    sp = tn / (tn + fp + K.epsilon())
    ntv = tn / (tn + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    
    a  = tf.where(tf.is_nan(a), tf.zeros_like(a), a )
    p = tf.where(tf.is_nan(p), tf.zeros_like(p), p)
    r = tf.where(tf.is_nan(r), tf.zeros_like(r), r)
    sp = tf.where(tf.is_nan(sp), tf.zeros_like(sp), sp)
    ntv = tf.where(tf.is_nan(ntv), tf.zeros_like(ntv), ntv)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    
    return K.mean((f1+a+p+r+sp+ntv/6)), a, p, r, sp, ntv, f1

def custom_loss_function(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    a = (tp + tn) / (tp+tn+fp+tn+K.epsilon())
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    sp = tn / (tn + fp + K.epsilon())
    ntv = tn / (tn + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    
    a  = tf.where(tf.is_nan(a), tf.zeros_like(a), a )
    p = tf.where(tf.is_nan(p), tf.zeros_like(p), p)
    r = tf.where(tf.is_nan(r), tf.zeros_like(r), r)
    sp = tf.where(tf.is_nan(sp), tf.zeros_like(sp), sp)
    ntv = tf.where(tf.is_nan(ntv), tf.zeros_like(ntv), ntv)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 6 - K.mean((f1+a+p+r+sp+ntv/6))

