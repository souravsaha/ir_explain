from explainers.base_explainer import BaseExplainer
import math

class BasePointwiseExplainer(BaseExplainer):
    def __init__(self, model):
        # super().__init__(model)
        # vs 
        self.model = model

    def preprocess(self, inputs):
        """ Preprocess inputs specific to pointwise ranking"""
        pass

    def explain(self, inputs):
        pass
    
    def get_document_vector(self, doc):
        print(f"{self.indexer_type}")

        if self.indexer_type == "no-indexer":
            tf = self.index_reader.get_document_vector(document_id)
        df = {term: (self.index_reader.get_term_counts(term, analyzer=None))[0] for term in tf.keys()}
        tot_num_docs_idx = self.index_reader.stats()['documents']
        #print("tot_num_docs_idx : ", tot_num_docs_idx)

        N_terms = len(tf.keys())
        idf = {}
        tfidf = {}
        #print("N_terms: ", N_terms)
        for term in tf.keys():
        tf[term] /= N_terms
        idf[term] = 1 + math.log(tot_num_docs_idx/(df[term]+1))
        tfidf[term] = tf[term] * idf[term]
        #print(term," , ", tf[term], " , " , df[term], " , ", idf[term], " , " , tfidf[term])

        return tfidf
        
        elif self.indexer_type == "pyserini":
            pass
        elif self.indexer_type == "pyterrier":
            pass
