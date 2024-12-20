from explainers.base_explainer import BaseExplainer
import math
from rank_bm25 import BM25Okapi

class BasePointwiseExplainer(BaseExplainer):
    def __init__(self, model):
        # super().__init__(model)
        # vs 
        self.model = model
        self.document_vector = None

    def preprocess(self, inputs):
        """ Preprocess inputs specific to pointwise ranking"""
        pass

    def explain(self, inputs):
        pass

    def explanation_to_vector(self, word_list, explain_tuples, doc_vector):
        #print(explain_tuples)
        return dict([ (word_list[entry[0]] ,\
        doc_vector[word_list[entry[0]]]) for entry in explain_tuples])
    
    def get_document_vector(self, doc, corpus):
        print(f"{self.indexer_type}")

        if self.indexer_type == "no-index":
            # print(corpus)
            bm25 = BM25Okapi(corpus)
            doc = doc.split(' ')
            doc = [term.lower() for term in doc]
            doc = list(set(doc))
            # TODO: maybe could remove stopwords and periods
            print(f"after tokenized {doc}")
            print()
            doc_vector = bm25.get_scores(doc)
            
            print(f"{len(doc_vector)}")
            tfidf = {}
            for term, weight in zip(doc, doc_vector):
                tfidf[term] = weight
            self.document_vector = tfidf
            print(tfidf)
            
            return tfidf
            
        elif self.indexer_type == "pyserini":
            pass
        elif self.indexer_type == "pyterrier":
            pass