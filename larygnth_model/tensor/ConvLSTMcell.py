import tensorflow as tf


class ConvLSTMCell(tf.keras.layers.Layer):
    def __init__(self, output_channels, kernel_shape, **kwargs):
        super(ConvLSTMCell, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.state_size = [self.output_channels, self.output_channels]

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.kernel_shape[0], self.kernel_shape[1], input_shape[-1], self.output_channels * 4),
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.kernel_shape[0], self.kernel_shape[1], self.output_channels, self.output_channels * 4),
                                                initializer='orthogonal',
                                                name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.output_channels * 4,),
                                    initializer='zeros',
                                    name='bias')

    def get_initial_state(self, inputs):
        initial_state = [tf.zeros([tf.shape(inputs)[0], self.output_channels]), 
                         tf.zeros([tf.shape(inputs)[0], self.output_channels])]
        return initial_state
    
    def call(self, inputs, states):
        h, c = states
        inputs = tf.cast(inputs, dtype=tf.float32)  # Convert the inputs to float32
        print('inputs.shape: ', inputs.shape)
        z = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        z = tf.nn.bias_add(z, self.bias)

        # separate all from z
        z0, z1, z2, z3 = tf.split(z, 4, axis=3)

        # gate
        input_gate = tf.nn.sigmoid(z0)
        forget_gate = tf.nn.sigmoid(z1)
        output_gate = tf.nn.sigmoid(z3)

        # new cell info
        cell = tf.nn.tanh(z2)
        new_c = forget_gate * c + input_gate * cell

        new_h = output_gate * tf.nn.tanh(new_c)

        return new_h, new_c


    # def call(self, inputs, states):
    #     h_prev, c_prev = states

    #     # convolution operation
    #     z = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
    #     z += tf.nn.conv2d(h_prev, self.recurrent_kernel, strides=[1, 1, 1, 1], padding='SAME')

    #     # add bias
    #     z = tf.nn.bias_add(z, self.bias)

    #     # split the data to input, forget, and output gates and block input activations vectors
    #     z = tf.split(z, num_or_size_splits=4, axis=3)

    #     # calculate the data for each gate
    #     input_gate = tf.nn.sigmoid(z[0])
    #     forget_gate = tf.nn.sigmoid(z[1])
    #     output_gate = tf.nn.sigmoid(z[2])
    #     block_input = tf.nn.tanh(z[3])

    #     # calculate the new cell state
    #     c = forget_gate * c_prev + input_gate * block_input

    #     # calculate the new hidden state for the cell
    #     h = output_gate * tf.nn.tanh(c)

    #     return h, [h, c]
