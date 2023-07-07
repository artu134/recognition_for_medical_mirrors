import tensorflow as tf
from tensorflow.keras.layers import GRUCell

# class ConvGRUCell(GRUCell):
#   def __init__(self, units, kernel, **kwargs):
#     self.units = units
#     self.kernel = kernel
#     super(ConvGRUCell, self).__init__(units, **kwargs)

#   def build(self, input_shape):
#     self.kernel_shape = tf.TensorShape((self.kernel[0], self.kernel[1], input_shape[-1] + self.units, 2 * self.units))
#     self.kernel = self.add_weight(shape=self.kernel_shape, initializer='uniform', name='kernel')
#     self.recurrent_kernel = self.add_weight(shape=self.kernel_shape, initializer='uniform', name='recurrent_kernel')
#     self.built = True

#   def call(self, inputs, states, training=None):
#     h_tm1 = states[0]  # previous memory state
#     x_h = tf.concat([inputs, h_tm1], axis=-1)
#     [z, r] = tf.split(self.kernel * x_h, 2, axis=-1)
#     z = tf.keras.activations.sigmoid(z)
#     r = tf.keras.activations.sigmoid(r)
#     hh = self.recurrent_kernel * tf.concat([inputs, r * h_tm1], axis=-1)
#     hh = tf.keras.activations.tanh(hh)
#     h = z * h_tm1 + (1 - z) * hh
#     return h, [h]

class ConvGRUCell(GRUCell):
    def __init__(self, units, kernel, filters, **kwargs):
        self.units = units
        self.kernel = kernel
        self.filters = filters
        super(ConvGRUCell, self).__init__(units, **kwargs)
        self.state_size = [self.units]
    
    def build(self, input_shape):
        self.kernel_shape = tf.TensorShape((self.kernel[0], self.kernel[1], input_shape[-1] + self.units, 2 * self.units))
        self.kernel = self.add_weight(shape=self.kernel_shape, initializer='uniform', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=self.kernel_shape, initializer='uniform', name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        x_h = tf.concat([inputs, h_tm1], axis=-1)  # concat input and previous memory state
        [z, r] = tf.split(self.kernel * x_h, 2, axis=-1)
        z = tf.nn.sigmoid(z)
        r = tf.nn.sigmoid(r)
        hh = self.recurrent_kernel * tf.concat([inputs, r * h_tm1], axis=-1)
        hh = tf.nn.tanh(hh)
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

