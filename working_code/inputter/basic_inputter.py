from abc import ABC, abstractmethod

class BasicInputter(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def input(self, source) -> bytes:
        pass