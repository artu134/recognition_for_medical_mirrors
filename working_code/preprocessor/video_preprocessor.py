from PIL import Image
import numpy as np  
from basic_preprocessor import BasicPreprocessor

class Preprocessor(BasicPreprocessor):
    def __init__(self, size=(256, 256)):
        super.__init__(self)
        self.size = size

    def preprocess(self, frames):
        return self.resize_and_crop(frames)

    def resize_and_crop(self, frames):
        processed_frames = []
        for frame in frames:
            # Resize
            frame = Image.fromarray(frame)
            frame = frame.resize(self.size, Image.ANTIALIAS)

            # Crop center
            width, height = frame.size   # Get dimensions
            new_width, new_height = self.size

            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            frame = frame.crop((left, top, right, bottom))

            processed_frames.append(np.array(frame))

        return np.array(processed_frames)
