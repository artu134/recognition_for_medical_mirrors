# main.py
import argparse
import inject
from module.di import configure_with_args
from working_code.dispatcher.basic_dispatcher import Dispatcher

def main(args):
    # Configure injector with command line arguments
    configure_with_args(args)
    
    # get instance of dispatcher with injected dependencies
    dispatcher = inject.instance(Dispatcher)
    
    # run the dispatcher
    dispatcher.run(args.video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video processing pipeline.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--framerate', type=int, default=30, help='Framerate for the video file.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for video processing.')
    parser.add_argument('--output_path', type=str, default=None, help='Output path for the model.')

    args = parser.parse_args()
    
    main(args)
