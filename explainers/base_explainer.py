from abc import ABC
class BaseExplainer():
    def __init__(self, model):
        self.model = model
    
    def preprocess(self, inputs):
        pass
    