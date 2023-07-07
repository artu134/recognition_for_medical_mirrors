import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, losses, optimizers
from layer import * 
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np

class LSTM_SA_Architecture(tf.keras.Model):
    def __init__(self, 
                 xsize=(256, 256),
                 ysize=(256, 256),
                 ndimensions=3,
                 nfilters=64,
                 nsequences=10,
                 nbatches=1,
                 nlayers=6,
                 nclasses=4,
                 loss_function="dice",
                 class_weights=None,
                 learning_rate=0.0001,
                 decay_rate=None,
                 bn=False,
                 reg=None,
                 reg_scale=0.001,
                 image_std=True,
                 crop_concat=True,
                 constant_nfilters=True,
                 name=None,
                 verbose=False,
                 two_gpus=False,
                 gru=False,
                 midseq=False,
                 **kwargs):
        
        super(LSTM_SA_Architecture, self).__init__(**kwargs)
        
        self.xsize = xsize
        self.ysize = ysize
        self.ndimensions = ndimensions
        self.nfilters = nfilters
        self.nsequences = nsequences
        self.nbatches = nbatches
        self.nclasses = nclasses
        self.nlayers = nlayers
        self.loss_function = loss_function
        self.class_weights = class_weights
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.bn = bn
        self.reg = reg
        self.reg_scale = reg_scale
        self.image_std = image_std
        self.crop_concat = crop_concat
        self.constant_nfilters = constant_nfilters
        #self.name = name
        self.verbose = verbose
        self.two_gpus = two_gpus
        self.gru = gru
        self.midseq = midseq

        self.dropout = Dropout(rate=self.reg_scale)
        self.global_step = tf.Variable(0, trainable=False)
        


        if reg is not None:
            if reg == "L2":
                self.regularizer = l2(self.reg_scale)
            else:
                raise Exception("Unknown Regularizer.")  
        else:
            self.regularizer = None


    def call(self, inputs):
        # Down-path
        down_path = []
        x = inputs
        for layer in range(self.nlayers):
            # Down-path layers
            if self.gru:
                x = bcgru(x, self.nfilters)
            else:
                x = bclstm(x, self.nfilters)

            down_path.append(x)
            x = max_pool2d_from_3d(x)
            
        # Middle connection
        if self.gru:
            x = bcgru(x, self.nfilters)
        else:
            x = bclstm(x, self.nfilters)

        # Up-path
        for layer in range(self.nlayers-1, -1, -1):
            # Up-path layers
            x = deconv2d_from_3d(x, self.nfilters, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            x = crop_concat_from_3d(down_path[layer], x)
            
            if self.gru:
                x = bcgru(x, self.nfilters)
            else:
                x = bclstm(x, self.nfilters)

        # Last layer
        x = conv2d_from_3d(x, self.nclasses, ksize=1, strides=1, padding="SAME")

        return x

    def get_loss(self, y_true, y_pred, loss_function="softmax", class_weights=None):
        if self.midseq:
            y_pred = y_pred[:, 4:6, :, :, :]
            y_true = y_true[:, 4:6, :, :, :]

        if loss_function == "softmax":
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)
            
            if class_weights is not None:
                weights = tf.reduce_sum(class_weights * y_true, axis=-1)
                loss *= weights

            return tf.reduce_mean(loss)

        elif loss_function == "dice":
            y_pred = tf.nn.softmax(y_pred)
            loss = 1 - sdc(y_true, y_pred)  # Assuming sdc is defined elsewhere and returns a Dice coefficient
            
            if class_weights is not None:
                loss = [a*b for a,b in zip(loss, class_weights)]
                loss = tf.reduce_mean(loss)
                loss = tf.reduce_mean(class_weights) - loss
            
            return loss

        else:
            raise Exception("Unknown Loss-Function.")


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.get_loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}
    

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_pred = self([x, y], training=False)

        # Updates the metrics tracking the loss
        loss = self.get_loss(y, y_pred, class_weights=self.class_weights)

        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    


class LSTM_SA_Trainer:
    
    def __init__(self, 
                 dataprovider_train, 
                 dataprovider_valid, 
                 log_path, 
                 model_path,
                 drop_rate=0.5,
                 batch_size=32, 
                 epochs=20, 
                 display_step=10,
                 save_model=True,
                 load_model_path=None,
                 skip_val=False):
        
        self.dataprovider_train = dataprovider_train
        self.dataprovider_valid = dataprovider_valid
        self.model_path = model_path
        self.log_path = log_path
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.display_step = display_step
        self.save_model = save_model
        self.load_model_path = load_model_path
        self.skip_val = skip_val

    def compile_model(self, lstm_net):
        # Compile the model, could be a custom compilation
        lstm_net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Load weights if load_model_path is specified
        if self.load_model_path:
            lstm_net.load_weights(self.load_model_path)

        return lstm_net

    def train(self, lstm_net):
        lstm_net = self.compile_model(lstm_net)

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch+1, self.epochs))

            # Training loop
            for step in range(self.dataprovider_train.steps_per_epoch):
                # Get next batch of data
                x_batch_train, y_batch_train = self.dataprovider_train.next_batch()

                # Train on batch and return dict of metrics
                metrics = lstm_net.train_on_batch(x_batch_train, y_batch_train)

                # Print metrics every display_step steps
                if step % self.display_step == 0:
                    print('Training metrics at step {}: {}'.format(step, metrics))

            # Validate
            if not self.skip_val:
                self.validate(lstm_net)

        # Save the model after training
        if self.save_model:
            lstm_net.save_weights(self.model_path)
        print('Training finished.')

    def validate(self, lstm_net):
        # Initialize lists for metrics
        val_accs = []
        val_precisions = []
        val_recalls = []
        val_f1s = []

        # Validation loop
        for step in range(self.dataprovider_valid.steps_per_epoch):
            x_batch_val, y_batch_val = self.dataprovider_valid.next_batch()

            # Predict on batch
            y_pred = lstm_net.predict_on_batch(x_batch_val)
            y_pred = np.argmax(y_pred, axis=1)

            # Calculate metrics and append to lists
            val_accs.append(accuracy_score(y_batch_val, y_pred))
            val_precisions.append(precision_score(y_batch_val, y_pred, average='micro'))
            val_recalls.append(recall_score(y_batch_val, y_pred, average='micro'))
            val_f1s.append(f1_score(y_batch_val, y_pred, average='micro'))

        # Print average metrics
        print('Validation accuracy: {}'.format(np.mean(val_accs)))
        print('Validation precision: {}'.format(np.mean(val_precisions)))
        print('Validation recall: {}'.format(np.mean(val_recalls)))
        print('Validation F1: {}'.format(np.mean(val_f1s)))




    # def call(self, inputs, training=False):
    #     # Down-path
    #     down_path = []
    #     x = inputs
    #     for layer in range(self.nlayers):
    #         # Down-path layers
    #         if self.gru:
    #             x = bcgru(x, self.nfilters, verbose=self.verbose)
    #         else:
    #             x = bclstm(x, self.nfilters, verbose=self.verbose)

    #         down_path.append(x)
    #         x = max_pool2d_from_3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME", scope='max_pool'+str(layer))
            
    #     # Middle connection
    #     if self.gru:
    #         x = bcgru(x, self.nfilters, verbose=self.verbose)
    #     else:
    #         x = bclstm(x, self.nfilters, verbose=self.verbose)

    #     # Up-path
    #     for layer in range(self.nlayers-1, -1, -1):
    #         # Up-path layers
    #         x = deconv2d_from_3d(x, self.nfilters, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], scope='up_sampling'+str(layer))
    #         x = crop_concat(down_path[layer], x, scope='crop_concat'+str(layer))
            
    #         if self.gru:
    #             x = bcgru(x, self.nfilters, verbose=self.verbose)
    #         else:
    #             x = bclstm(x, self.nfilters, verbose=self.verbose)

    #     # Last layer
    #     x = conv2d_from_3d(x, self.nclasses, ksize=1, strides=1, padding="SAME", scope='logits')

    #     return x
    