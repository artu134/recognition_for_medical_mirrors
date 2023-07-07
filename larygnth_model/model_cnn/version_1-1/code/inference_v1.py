# inference_v1.py
import argparse
import yaml
import tensorflow as tf
import cv2
import util  # make sure util.py is in the same directory or properly installed

def load_model(checkpoint_path):
    sess = tf.compat.v1.Session()
    util.model_restore(sess, checkpoint_path)
    graph = tf.get_default_graph()
    return sess, graph

def infer(sess, graph, image_path):
    # Load image
    image = cv2.imread(image_path)

    # Do some preprocessing if needed
    # For example, resizing to the input size expected by your model:
    # image = cv2.resize(image, (width, height))

    # Get input and output tensors
    # Replace 'input:0' and 'output:0' with the names of your actual input and output tensors
    x = graph.get_tensor_by_name('input:0')
    y = graph.get_tensor_by_name('output:0')

    # Run the model
    prediction = sess.run(y, feed_dict={x: image})

    return prediction

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    sess, graph = load_model(config['model']['checkpoint_path'])

    # Perform inference
    prediction = infer(sess, graph, args.image)

    # Now you can use the `prediction` in any way you want.
    # For example, you can save it as an image:
    cv2.imwrite('output.png', prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to the config file.", default="config.yml")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    main(args)
