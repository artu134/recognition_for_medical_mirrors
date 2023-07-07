import tensorflow as tf
from PIL import Image
import glob
import numpy as np
import os
from sklearn.utils import shuffle


class SequenceDataProvider(tf.keras.utils.Sequence):
    def __init__(self, path, sequence_length, image_suffix="_rgb.png", 
                 label_suffix="_mask.png", nclasses=3, background=False, 
                 background_mask_index=None, shuffle_data=True, debug=False):
        self._path = path
        self._image_suffix = image_suffix
        self._label_suffix = label_suffix
        self._nclasses = nclasses
        self._background = background
        self._background_mask_index = background_mask_index if background else None
        self._shuffle_data = shuffle_data
        self._sequence_length = sequence_length
        self._debug = debug
        self._filelist = self._get_filelist()


    
    def __len__(self):
        batch_size = 1  # Define your batch size here
        return len(self._filelist) // (self._sequence_length * batch_size)


    # def __getitem__(self, idx):
    #     batch_size = 1  # Define your batch size here
    #     batch_x = []
    #     batch_y = []

    #     for b in range(batch_size):
    #         batch_files = self._filelist[idx * batch_size + b]
    #         batch_y.append([self._preprocess_label(self._load_label(x)) for x in batch_files])
    #         batch_x.append([self._preprocess_image(self._load_image(x)) for x in batch_files])

    #     return np.array(batch_x), np.array(batch_y)

    def __getitem__(self, idx):
        batch_files = self._filelist[idx]
        batch_y = [self._preprocess_label(self._load_label(x)) for x in batch_files]
        batch_x = [self._preprocess_image(self._load_image(x)) for x in batch_files]

        return np.expand_dims(np.array(batch_x), 0), np.expand_dims(np.array(batch_y), 0)


    def _get_filelist(self):
        file_names = [fname for fname in glob.glob(self._path) 
                    if (self._label_suffix is None and self._image_suffix in fname) 
                    or (self._image_suffix in fname and not self._label_suffix in fname)]
        if self._shuffle_data:
            file_names = shuffle(file_names)

        sequences = []
        nsequence = -1
        for i in range(len(file_names)):
            if i % self._sequence_length == 0:
                sequences.append([])
                nsequence += 1
            fname = file_names[i]
            sequences[nsequence].append(fname)

        return sequences  # now sequences of files are returned

    
    def _load_image(self, image_path):
        return np.asarray(Image.open(image_path))
    
    def _load_label(self, image_path):
        label_path = image_path.replace(self._image_suffix, self._label_suffix)
        return np.asarray(Image.open(label_path))
    
    def _preprocess_image(self, image):  
        if len(image.shape) == 2: 
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        return image
    
    def _preprocess_label(self, label):
        nx, ny = label.shape[:2]
        processed_label = np.zeros((nx, ny, self._nclasses))

        if self._background:
            processed_label = np.zeros((nx, ny, self._nclasses + 1))
            processed_label[..., self._nclasses] = (label==0)
        
        for i in range(self._nclasses):
            pixel_value = i + 1
            processed_label[..., i] = (label==pixel_value)

        return processed_label
