from abc import ABC
class BaseExplainer(ABC):
    def __init__(self, model):
        self.model = model
    
    def preprocess(self, inputs):
        pass
    