from abc import ABC, abstractmethod

class BasicPreprocessor(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def preprocess(self, data):
        pass
