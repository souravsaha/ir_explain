from ir_explain.explainers.base_explainer import BaseExplainer


class BaseListwiseExplainer(BaseExplainer):
    def __init__(self, model):
        self.model = model 
        self.document_vector = None
    
    def preprocess(self, inputs):
        """ Preprocess inputs specific to listwise ranking """
        pass 
    
    def explain(self, inputs):
        pass
