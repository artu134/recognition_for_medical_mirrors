import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from util import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Disable eager execution for compatibility with older TF models
tf.compat.v1.disable_eager_execution()

def load_model(session, save_path):
    """
    Load TensorFlow model from save path.
    """

    saver = tf.compat.v1.train.import_meta_graph(save_path + '.meta')
    saver.restore(session, save_path)
    return saver

def run_inference(image_path, model_path):
    """
    Run inference on the given image with the provided model.
    """
    # Create session
    with tf.compat.v1.Session() as sess:
        # Load model
        saver = load_model(sess, model_path)
        # Get default graph (supply your custom graph if you have one)
        graph = tf.compat.v1.get_default_graph()

        # It's assumed that images are input into the model with the name 'image_pl' and the model outputs probabilities 'probs_op'
        # If these tensor names do not match your model, modify the code accordingly
        image_pl = graph.get_tensor_by_name('image_pl:0')
        probs_op = graph.get_tensor_by_name('probs_op:0')

        # Open and preprocess the image
        image = np.asarray(Image.open(image_path))
        if len(image.shape) == 2:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        # Expand dimensions to match the dimensions the model expects
        image = np.expand_dims(image, axis=0)

        # Run inference
        probabilities = sess.run(probs_op, feed_dict={image_pl: image})

        return probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on a single image.')
    parser.add_argument('--image_path', type=str, help='Path to the image file.')
    parser.add_argument('--model_path', type=str, default='C:\\Users\\Roman\\Desktop\\ocr_for_mirros\\larygnth_model\\model_cnn\\version_1-1\\code\\lstm_save\\model-30000', help='Path to the model file.')

    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model_path

    probabilities = run_inference(image_path, model_path)
    print("Predicted probabilities:", probabilities)
