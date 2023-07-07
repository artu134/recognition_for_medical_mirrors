from inputter.basic_inputter import BasicInputter
from preprocessor.basic_preprocessor import BasicPreprocessor
from model.basic_model import BasicModel


class Dispatcher:
    def __init__(self, inputter: BasicInputter, preprocessor: BasicPreprocessor, 
                 model: BasicModel, callback):
        self.inputter = inputter
        self.preprocessor = preprocessor
        self.model = model
        self.callback = callback

    def run(self, source):
        # inputting data from source
        for batch in self.inputter.input(source):
            # preprocessing each batch
            preprocessed_batch = self.preprocessor.preprocess(batch)
            
            # predicting using the model
            predictions = self.model.predict(preprocessed_batch)

            # sending the predictions to callback function
            self.callback(predictions)