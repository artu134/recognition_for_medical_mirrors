import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ConvLSTMcell import ConvLSTMCell

from ConvGRUcell import ConvGRUCell

#################
# normal layers #
#################

def conv2d(inputs, nfilters , ksize=3, padding="valid", regularizer=None, drate=0.0, activation=tf.nn.elu, bn=False, verbose=False):
    conv = layers.Conv2D(nfilters, ksize, padding=padding, kernel_initializer=tf.keras.initializers.GlorotUniform(), bias_initializer=tf.keras.initializers.Constant(0.01), activation=activation, kernel_regularizer=regularizer)(inputs)
    if bn:
        conv = layers.BatchNormalization()(conv)
    drop = layers.Dropout(rate=drate)(conv)
    if bn:
        drop = layers.BatchNormalization()(drop)
    return drop

def conv2d_from_3d(inputs, nfilters , ksize=3, padding="valid", regularizer=None, drate=0.0, activation=tf.nn.elu, bn=False, verbose=False):
    shape = inputs.shape
    reshape1 = tf.reshape(inputs, [-1, shape[2], shape[3], shape[4]])
    conv1 = tf.keras.layers.Conv2D(nfilters, ksize, padding=padding, kernel_regularizer=regularizer, activation=activation)(reshape1)
    if bn: conv1 = tf.keras.layers.BatchNormalization()(conv1)
    reshape2 = tf.reshape(conv1, [-1, shape[1], shape[2], shape[3], nfilters])
    return reshape2


def max_pool2d(inputs, size=2, strides=2, verbose=False):
    return layers.MaxPooling2D(pool_size=size, strides=strides)(inputs)

def max_pool2d_from_3d(inputs, size=2, strides=2, verbose=False):
    shape_before = inputs.shape
    reshape1 = tf.reshape(inputs, [-1, shape_before[2], shape_before[3], shape_before[4]])
    pool1 = max_pool2d(reshape1, size=size, strides=strides, verbose=verbose)
    shape_after = pool1.shape
    reshape2 = tf.reshape(pool1, [-1, shape_after[0], shape_after[1], shape_after[2], shape_after[3]])
    return reshape2


def deconv2d(inputs, nfilters, ksize=2, strides=2, padding="valid", regularizer=None, activation=tf.nn.elu, verbose=False):
    return layers.Conv2DTranspose(nfilters, ksize, strides=strides, padding=padding, kernel_initializer=tf.keras.initializers.GlorotUniform(), bias_initializer=tf.keras.initializers.Constant(0.01), activation=activation, kernel_regularizer=regularizer)(inputs)

def deconv2d_from_3d(inputs, nfilters, ksize=2, strides=2, padding="valid", regularizer=None, activation=tf.nn.elu, verbose=False):
    shape_before = inputs.shape
    reshape1 = tf.reshape(inputs, [-1, shape_before[2], shape_before[3], shape_before[4]])
    deconv1 = deconv2d(reshape1, nfilters, ksize=ksize, strides=strides, padding=padding, regularizer=regularizer, activation=activation, verbose=verbose)
    shape_after = deconv1.shape
    reshape2 = tf.reshape(deconv1, [-1, shape_after[0], shape_after[1], shape_after[2], nfilters])
    return reshape2

def crop_concat(inputs_1, inputs_2, verbose=False):
    cur_size = inputs_1.shape
    target_size = inputs_2.shape
    offset = [0, int((cur_size[1] - target_size[1]) // 2), int((cur_size[2] - target_size[2]) // 2), 0]
    size = (-1, int(target_size[1]), int(target_size[2]), -1)
    crop = tf.slice(inputs_1, offset, size) 
    concat = tf.concat([crop, inputs_2], 3)
    return concat

def crop_concat_from_3d(inputs_1, inputs_2, verbose=False):
    shape_inputs1 = inputs_1.shape
    reshape_inputs1_1 = tf.reshape(inputs_1, [-1, shape_inputs1[2], shape_inputs1[3], shape_inputs1[4]])
    shape_inputs2 = inputs_2.shape
    reshape_inputs2_1 = tf.reshape(inputs_2, [-1, shape_inputs2[2], shape_inputs2[3], shape_inputs2[4]])
    _cc = crop_concat(reshape_inputs1_1, reshape_inputs2_1, verbose=verbose)
    reshape2 = tf.reshape(_cc, [-1, shape_inputs2[1], shape_inputs2[2], shape_inputs2[3], shape_inputs2[4]*2])
    return reshape2

def bclstm(inputs, nfilters=64, ksize=3, verbose=False):
    cell_fw = ConvLSTMCell(output_channels=nfilters//2, kernel_shape=[ksize, ksize])
    cell_bw = ConvLSTMCell(output_channels=nfilters//2, kernel_shape=[ksize, ksize])

    initial_state_fw = cell_fw.get_initial_state(inputs)
    initial_state_bw = cell_bw.get_initial_state(inputs)

    output_fw, _ = tf.keras.layers.RNN(cell_fw, return_sequences=True, return_state=True, go_backwards=False)(inputs, initial_state=initial_state_fw)
    output_bw, _ = tf.keras.layers.RNN(cell_bw, return_sequences=True, return_state=True, go_backwards=True)(inputs, initial_state=initial_state_bw)

    concated_outputs = tf.concat([output_fw, output_bw], -1)
    if verbose: 
        print(concated_outputs.shape)
    return concated_outputs

def bcgru(inputs, nfilters=64, ksize=3, verbose=False):
    cell_fw = ConvGRUCell(units=nfilters//2, kernel=[ksize, ksize], filters=nfilters//2)
    cell_bw = ConvGRUCell(units=nfilters//2, kernel=[ksize, ksize], filters=nfilters//2)

    state_size = cell_fw.state_size
    initial_state_fw = [tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], state_size])]
    initial_state_bw = [tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], state_size])]

    output_fw, _ = tf.keras.layers.RNN(cell_fw, return_sequences=True, return_state=True, go_backwards=False)(inputs, initial_state=initial_state_fw)
    output_bw, _ = tf.keras.layers.RNN(cell_bw, return_sequences=True, return_state=True, go_backwards=True)(inputs, initial_state=initial_state_bw)

    concated_outputs = tf.concat([output_fw, output_bw], -1)
    if verbose: 
        print(concated_outputs.shape)
    return concated_outputs

#######################
# acc/loss functions #
#######################
def dice(predictions, labels, nclasses, smooth=1e-5, class_index=None):

    predictions = tf.round(predictions)
    labels = tf.round(labels)

    if class_index == None:
        return _dice(predictions, labels, nclasses, smooth)

    else:

        with tf.name_scope("dice_class_" + str(class_index)):

            predictions = predictions[..., class_index]
            labels = labels[..., class_index]

            tp = tf.reduce_sum(predictions * labels)
            
            npixel_pred = tf.reduce_sum(predictions)
            npixel_label = tf.reduce_sum(labels)

            dice = (2. * tp + smooth) / (npixel_pred + npixel_label + smooth)

            return dice


def _dice(predictions, labels, nclasses, smooth=1e-5):
    dices = []
    for i in range(nclasses):
        p = predictions[..., i]
        l = labels[..., i]
        tp = tf.reduce_sum(p * l)
        npixel_pred = tf.reduce_sum(p)
        npixel_label = tf.reduce_sum(l)
        dice = (2. * tp + smooth) / (npixel_pred + npixel_label + smooth)
        dices.append(dice)
    return tf.reduce_mean(dices)

def metric_mean_precision(predictions, labels, nclasses):
    # Precision = true_positive / (true_positive + false_positive)
    with tf.name_scope("metrics_mean_precision"):
        predictions = tf.round(predictions)
        
        ones_like_labels = tf.ones_like(labels[..., 0])
        zeros_like_labels = tf.zeros_like(labels[..., 0])
        
        metrics = []
        for i in range(nclasses):
            per_class_pred = predictions[..., i]
            per_class_label = labels[..., i]
            
            tp = tf.reduce_sum(tf.cast(tf.logical_and(
                                            tf.equal(per_class_label, ones_like_labels), 
                                            tf.equal(per_class_pred, ones_like_labels)), "float"))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(
                                            tf.equal(per_class_label, zeros_like_labels), 
                                            tf.equal(per_class_pred, ones_like_labels)), "float"))

            m = tp / (tp + fp)
            m = tf.where(tf.is_nan(m), 1., m)
            metrics.append(m)

        return tf.reduce_mean(metrics)

def metric_mean_recall(predictions, labels, nclasses):
    # Recall = true_positive / (true_positive + false_negative)
    with tf.name_scope("metrics_mean_recall"):
        predictions = tf.round(predictions)
        
        ones_like_labels = tf.ones_like(labels[..., 0])
        zeros_like_predictions = tf.zeros_like(predictions[..., 0])
        
        metrics = []
        for i in range(nclasses):
            per_class_pred = predictions[..., i]
            per_class_label = labels[..., i]
            
            tp = tf.reduce_sum(tf.cast(tf.logical_and(
                                            tf.equal(per_class_label, ones_like_labels), 
                                            tf.equal(per_class_pred, ones_like_labels)), "float"))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(
                                            tf.equal(per_class_label, ones_like_labels), 
                                            tf.equal(per_class_pred, zeros_like_predictions)), "float"))

            m = tp / (tp + fn)
            m = tf.where(tf.is_nan(m), 1., m)
            metrics.append(m)

        return tf.reduce_mean(metrics)


def sdc(predictions, labels, nclasses=3, smooth=1e-5):
    avg_dice = []
    for i in range(nclasses):
        per_class_pred = predictions[..., i]
        per_class_label = labels[..., i]

        tp = tf.reduce_sum(per_class_pred * per_class_label)
        npixel_pred = tf.reduce_sum(per_class_pred)
        npixel_label = tf.reduce_sum(per_class_label)

        dice = (2. * tp + smooth) / (npixel_pred + npixel_label + smooth)
        avg_dice.append(dice)

    return avg_dice

def metric_pixel_accuraccy(predictions, labels):
    predictions = tf.round(predictions)

    npixel_correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
    npixel_all = tf.cast(tf.size(labels), dtype=tf.float32)

    return npixel_correct / npixel_all

def metric_pixel_error(predictions, labels):
    return 1 - metric_pixel_accuraccy(predictions, labels)

def metric_mean_pa(predictions, labels, nclasses):
    predictions = tf.round(predictions)

    metrics = []
    for i in range(nclasses):
        per_class_pred = predictions[..., i]
        per_class_label = labels[..., i]

        tp = tf.reduce_sum(per_class_pred * per_class_label)
        npixel_per_class = tf.reduce_sum(per_class_label)

        m = tp / npixel_per_class
        m = tf.where(tf.math.is_nan(m), 1., m)
        metrics.append(m)

    return tf.reduce_mean(metrics)

# def metric_mean_pa(predictions, labels, nclasses):
#     predictions = tf.round(predictions)

#     metrics = []
#     for i in range(nclasses):
#         per_class_pred = predictions[..., i]
#         per_class_label = labels[..., i]

#         tp = tf.reduce_sum(tf.cast(tf.equal(per_class_pred, per_class_label), tf.float32))
#         npixel_all = tf.reduce_sum(tf.cast(tf.equal(per_class_label, 1), tf.float32))

#         pa = tp / npixel_all
#         metrics.append(pa)

#     return tf.reduce_mean(metrics)

def metric_mean_iou(predictions, labels, nclasses):
    # IoU = true_positive / (true_positive + false_positive + false_negative)
    predictions = tf.round(predictions)

    ones_like_labels = tf.ones_like(labels[..., 0])
    zeros_like_labels = tf.zeros_like(labels[..., 0])
    ones_like_predictions = tf.ones_like(predictions[..., 0])
    zeros_like_predictions = tf.zeros_like(predictions[..., 0])

    metrics = []
    for i in range(nclasses):
        per_class_pred = predictions[..., i]
        per_class_label = labels[..., i]

        tp = tf.reduce_sum(tf.cast(tf.logical_and(
                                        tf.equal(per_class_label, ones_like_labels),
                                        tf.equal(per_class_pred, ones_like_predictions)), "float"))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(
                                        tf.equal(per_class_label, zeros_like_labels),
                                        tf.equal(per_class_pred, ones_like_predictions)), "float"))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(
                                        tf.equal(per_class_label, ones_like_labels),
                                        tf.equal(per_class_pred, zeros_like_predictions)), "float"))

        m = tp / (tp + fp + fn)
        m = tf.where(tf.math.is_nan(m), 1., m)
        metrics.append(m)

    return tf.reduce_mean(metrics)

def metric_mean_f1(predictions, labels, nclasses):
    # F1 score = 2 * (precision * recall) / (precision + recall)
    with tf.name_scope("metrics_mean_f1"):
        precision = metric_mean_precision(predictions, labels, nclasses)
        recall = metric_mean_recall(predictions, labels, nclasses)

        return 2 * (precision * recall) / (precision + recall)
    


# def conv2d(inputs, nfilters , ksize=3, padding="valid", regularizer=None, drate=0.0, activation=tf.nn.elu, bn=False, verbose=False):
#     conv = layers.Conv2D(nfilters, ksize, padding=padding, 
#                          kernel_initializer=tf.keras.initializers.GlorotUniform(), 
#                          bias_initializer=tf.keras.initializers.Constant(0.01),
#                            activation=activation, kernel_regularizer=regularizer)(inputs)
#     if bn:
#         conv = layers.BatchNormalization()(conv)
#     drop = layers.Dropout(rate=drate)(conv)
#     if bn:
#         drop = layers.BatchNormalization()(drop)
#     return drop

# def deconv2d(inputs, nfilters, ksize=2, strides=2, padding="valid", regularizer=None, activation=tf.nn.elu, bn=False, verbose=False):
#     deconv = layers.Conv2DTranspose(nfilters, ksize, strides=strides, padding=padding, kernel_initializer=tf.keras.initializers.GlorotUniform(), bias_initializer=tf.keras.initializers.Constant(0.01), activation=activation, kernel_regularizer=regularizer)(inputs)
#     if bn:
#         deconv = layers.BatchNormalization()(deconv)
#     drop = layers.Dropout(rate=drate)(deconv)
#     if bn:
#         drop = layers.BatchNormalization()(drop)
#     return drop

# def bclstm(inputs, nfilters=64, ksize=3, bn=False, drate=0.0, verbose=False):
#     cell_fw = ConvLSTMCell(filters=nfilters//2, kernel_size=[ksize, ksize])
#     cell_bw = ConvLSTMCell(filters=nfilters//2, kernel_size=[ksize, ksize])
#     output_fw, _ = cell_fw(inputs, training=training)
#     output_bw, _ = cell_bw(inputs, training=training)
#     output_fw = layers.Dropout(rate=drate)(output_fw)
#     output_bw = layers.Dropout(rate=drate)(output_bw)
#     if bn:
#         output_fw = layers.BatchNormalization()(output_fw)
#         output_bw = layers.BatchNormalization()(output_bw)
#     concated_outputs = tf.concat([output_fw, output_bw], -1)

#     if verbose: 
#         print(concated_outputs.shape)
#     return concated_outputs

# def bcgru(inputs, nfilters=64, ksize=3, bn=False, drate=0.0, verbose=False):
#     cell_fw = ConvGRUCell(filters=nfilters//2, kernel=[ksize, ksize])
#     cell_bw = ConvGRUCell(filters=nfilters//2, kernel=[ksize, ksize])
#     output_fw, _ = cell_fw(inputs, training=training)
#     output_bw, _ = cell_bw(inputs, training=training)
#     output_fw = layers.Dropout(rate=drate)(output_fw)
#     output_bw = layers.Dropout(rate=drate)(output_bw)
#     if bn:
#         output_fw = layers.BatchNormalization()(output_fw)
#         output_bw = layers.BatchNormalization()(output_bw)
#     concated_outputs = tf.concat([output_fw, output_bw], -1)

#     if verbose: 
#         print(concated_outputs.shape)
#     return concated_outputs