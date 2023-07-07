# injections.py
import inject
from inject import singleton
from inputter.file_inputter import FileVideoInputter
from preprocessor.video_preprocessor import Preprocessor
from model.laryngth_tensor_model import LaryngthTensorModel
from dispatcher.basic_dispatcher import Dispatcher
from inputter.basic_inputter import BasicInputter
from preprocessor.basic_preprocessor import BasicPreprocessor
from model.basic_model import BasicModel

def callback(predictions):
    # define what happens with predictions here
    print(predictions)

@singleton
def dispatcher_factory(inputter: BasicInputter, preprocessor: BasicPreprocessor, 
                       model: BasicModel) -> Dispatcher:
    return Dispatcher(inputter, preprocessor, model, callback)

def configure(binder, model_path: str, output_path: str,
              framerate: int, batch_size: int):
    binder.bind(BasicInputter, FileVideoInputter(framerate=framerate, batch_size=batch_size))
    binder.bind(BasicPreprocessor, Preprocessor())
    binder.bind(BasicModel, LaryngthTensorModel(model_path=model_path, output_path=output_path))
    binder.bind_to_constructor(Dispatcher, 
        lambda: dispatcher_factory(binder[BasicInputter], binder[BasicPreprocessor], 
                                   binder[BasicModel]))

# This is a modification of the configuration function to accept arguments.
def configure_with_args(args):
    inject.configure(lambda binder: configure(binder, args.model_path, args.output_path, args.framerate, args.batch_size))
