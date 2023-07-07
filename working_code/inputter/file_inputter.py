import cv2
from basic_inputter import BasicInputter

class FileVideoInputter(BasicInputter):
    def __init__(self, config, framerate, batch_size=10):
        super().__init__(config)
        self.framerate = framerate
        self.batch_size = batch_size

    def input(self, source):
        return self.__get_batch__(source)
    
    def __get_batch__(self, source):
        cap = cv2.VideoCapture(source)
        frames = []
        i = 0
        while(cap.isOpened()):
            cap.set(cv2.CAP_PROP_POS_MSEC, (i*self.framerate)) 
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                if len(frames) == self.batch_size:
                    yield frames
                    frames = []
            else:
                break
            i += 1

        # yield any remaining frames
        if frames:
            yield frames
            
        cap.release()