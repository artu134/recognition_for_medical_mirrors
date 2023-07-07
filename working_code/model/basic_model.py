import abc

class BasicModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, model_path):
        self.model_path = model_path

    @abc.abstractmethod
    def _create_graph(self):
        pass

    @abc.abstractmethod
    def model_restore(self, sess, path):
        pass

    @abc.abstractmethod
    def predict(self, input_sequences, batch_size=1, save_validate_image=False, save_path=None):
        pass