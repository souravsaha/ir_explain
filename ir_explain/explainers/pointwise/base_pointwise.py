from ir_explain.explainers.base_explainer import BaseExplainer
import math, nltk, json
from rank_bm25 import BM25Okapi
from pyserini.index.lucene import IndexReader

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
            doc = nltk.word_tokenize(doc)
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
            docid = doc
            index_reader = IndexReader(self.corpus_path)
            doc_object = self.searcher.doc(doc)
            doc = json.loads(doc_object.raw())['contents']
            doc = nltk.word_tokenize(doc)
            doc = [term.lower() for term in doc]
            doc = list(set(doc))
            
            tfidf = {}
            for term in doc:
                tfidf[term] = index_reader.compute_bm25_term_weight(docid, term, analyzer=None)
            self.document_vector = tfidf

            return tfidf 

            # index_reader = IndexReader(self.corpus_path)
            # # notice that here, doc is not the document content, rather docid
            # doc_vector = index_reader.get_document_vector(doc)
            # if doc_vector == None:
            #     print(f"Unable to fetch document vector. Document vector is empty")
            #     return {}
            # else:                
            #     bm25_vector = {term: index_reader.compute_bm25_term_weight(doc, term, analyzer=None) for term in doc_vector.keys()}
            #     return bm25_vector
        elif self.indexer_type == "pyterrier":
            pass