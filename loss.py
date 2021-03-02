import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error


# Ref: https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html
alpha = .25
gamma = 2
def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

def focal_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

def regress_loss(y_true, y_pred): # Regression Loss
    # mask = y_true[:,:,:, 0][:,:,:,np.newaxis]
    # regr = y_true[:,:,:,1:]
    regr = y_true[:,:,:, -2:]

    regr_loss = mean_squared_error(regr, y_pred)
    loss = regr_loss

    return loss

def conf_loss(y_true, y_pred): # Heatmap Loss
    # mask = y_true[:,:,:, 0:-2]#[:,:,:,np.newaxis]
    mask = y_true
    prediction = y_pred

    # Binary mask loss
    # pred_mask = tf.sigmoid(prediction[:,:,:, 0])[:,:,:,np.newaxis]
    pred_mask = tf.sigmoid(prediction)
    mask_loss = focal_loss(mask, pred_mask)
    mask_loss = tf.reduce_mean(mask_loss)

    loss = mask_loss
    return loss