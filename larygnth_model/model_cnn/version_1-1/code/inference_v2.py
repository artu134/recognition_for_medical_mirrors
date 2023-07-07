import tensorflow as tf
import argparse
import yaml
import cv2

def load_model(checkpoint_dir):
    # Initialize a checkpoint object
    checkpoint = tf.train.Checkpoint()

    # Create a CheckpointManager to manage the checkpoints
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Restore the latest checkpoint
    status = checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return checkpoint

def infer(checkpoint, image_path):
    # Load image
    image = cv2.imread(image_path)

    # Do some preprocessing if needed
    # For example, resizing to the input size expected by your model:
    # image = cv2.resize(image, (width, height))

    # Run the model
    prediction = checkpoint.model(image)

    return prediction

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    checkpoint = load_model(config['model']['checkpoint_path'])

    # Perform inference
    prediction = infer(checkpoint, args.image)

    # Now you can use the `prediction` in any way you want.
    # For example, you can save it as an image:
    cv2.imwrite('output.png', prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to the config file.", default="config.yml")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    main(args)
