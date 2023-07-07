import tensorflow as tf
import logging
from basic_model import BasicModel
from layer import *
from working_code.model.util import save_img_prediction

class LaryngthTensorModel(BasicModel):
    def __init__(self, model_path, output_path = None, config = None):
        super().__init__(model_path)
        self.output_path = output_path
        self.config = config #if config else self._load_config()
        self.graph, self.xi, self.yi, self.dr, self.p, self.ppe, self.pmiou, self.ppa, self.pmpa, self.pd = self._create_graph()
        with tf.Session(graph=self.graph) as sess:
            self.model_restore(sess, self.model_path)

    def _create_graph(self):
        g = tf.Graph()
        with g.as_default(): 
            # Assuming that self._net is already defined and has the necessary methods
            logits, x_input, y_input, drop_rate, global_step = self._net.create_net()
            loss = self._net.get_loss(logits, y_input)
            optimizer = self._net.get_optimizer(loss, global_step)

            prediction = tf.nn.softmax(logits)
            prediction = tf.round(prediction)
            
            # Mid.Seq.
            _prediction = tf.slice(prediction, [0, 4, 0, 0, 0], [prediction.shape[0], 2, prediction.shape[2], prediction.shape[3], prediction.shape[4]])
            _y_input = tf.slice(y_input, [0, 4, 0, 0, 0], [y_input.shape[0], 2, y_input.shape[2], y_input.shape[3], y_input.shape[4]])
            
            # Assuming metric_pixel_error, metric_mean_iou, metric_pixel_accuraccy, metric_mean_pa, dice functions are already defined
            performance_pixel_error = metric_pixel_error(_prediction, _y_input)
            performance_mean_iou = metric_mean_iou(_prediction, _y_input, self._net._nclasses)
            performance_pa = metric_pixel_accuraccy(_prediction, _y_input)
            performance_mpa = metric_mean_pa(_prediction, _y_input, self._net._nclasses)
            performance_dice = dice(_prediction, _y_input, self._net._nclasses)
            
            self._performance_pe_list = []
            self._performance_miou_list = []
            self._performance_pa_list = []
            self._performance_mpa_list = []
            self._performance_d_list = []

            return g, x_input, y_input, drop_rate, _prediction, performance_pixel_error, performance_mean_iou, performance_pa, performance_mpa, performance_dice 

    def model_restore(self, sess, path):
        logging.info("Restoring model from file: {} ...".format(path))
        saver = tf.train.Saver()
        saver.restore(sess, path)
        logging.info("Model successfully restored.")

    def predict(self, input_sequences, batch_size=1, save_validate_image=False):
        # initialize lists to hold prediction results and performance metrics
        predictions = []
        performance_metrics = []
        
        with tf.Session(graph=self.graph) as sess:
            # iterate over input sequences in batches
            for i in range(0, len(input_sequences), batch_size):
                x = input_sequences[i:i+batch_size]
                feed_dict = {self.xi: x, self.dr:0.}         
                out = sess.run([self.p, self.ppe, self.pmiou, self.ppa, self.pmpa, self.pd], feed_dict)
                out_p, out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd = out
                
                # add output prediction and performance metrics to the respective lists
                predictions.append(out_p)
                performance_metrics.append([out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd])

                if save_validate_image:
                    output_x = x[:,4:6,...]

                    if i == 0:
                        #init
                        sequence_output_x = output_x
                        sequence_output_p = out_p
                    else:
                        sequence_output_x = np.concatenate([sequence_output_x, output_x], axis=1)
                        sequence_output_p = np.concatenate([sequence_output_p, out_p], axis=1)
                    
            if save_validate_image:
                pass
                #self._save_image(sequence_output_x, sequence_output_p, 'validation')

        return predictions, performance_metrics
    
    def _save_image(self, x, y, p, nr):
        xxsize, xysize, xch = x.shape[2], x.shape[3], x.shape[4]
        yxsize, yysize, ych = y.shape[2], y.shape[3], y.shape[4]
        x = np.reshape(x, [-1, xxsize, xysize, xch])
        y = np.reshape(y, [-1, yxsize, yysize, ych])
        p = np.reshape(p, [-1, yxsize, yysize, ych])
        save_img_prediction(x, y, p, self.output_path, image_name=str(nr), background_mask=self._background_mask_index)