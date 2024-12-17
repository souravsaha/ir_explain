from explainers.pointwise.base_pointwise import BasePointwiseExplainer
import numpy as np
from explainers.pointwise.lime import lime_ranker

class LirmePointwiseExplainer(BasePointwiseExplainer):
    def __init__(self, model, corpus_path, indexer_type):
        # super().__init__(model)
        self.model = model
        self.corpus_path = corpus_path
        self.indexer_type = indexer_type
        self.corpus = None

        if self.indexer_type == "no-index":
            self.corpus = self.preprocess(corpus_path)
            print(f"length of corpus : {len(self.corpus)}")
    
    def preprocess(self, corpus_path):
        r"""
        Preprocess the corpus, input-type: docid \t document
        return: a list of documents 
        TODO: can be moved to dataloaders class
        """
        corpus = []
        with open(corpus_path) as corpus_file:
            for line in corpus_file:
                corpus.append(line.strip("\n").split("\t")[1])
                
        return corpus

    def explain(self, query, doc, params):
        print("You wanted to explain with LIRME")
        
        sampling_method = params["sampling_method"]
        num_top_terms = params["top_terms"]
        kernel_range = params["kernel_range"]


        document_vectors = self.get_document_vector(doc)

        
        ranker_explanation = lime_ranker.LimeRankerExplainer(#kernel=uniform_kernel,\
										  kernel_width = np.sqrt(100000) * .80,\
										  random_state=123456, verbose=verbose, \
										  relevance_labels=[0,1,2,3,4])
        """
        sample_generator = PerturbDocument(200)
        if(sampling_method == 'random_words'):
          samples_generated = sample_generator.random_sampler_using_doc_id(doc_id, self.index_reader)[1:]    #first sample is the original doc text
        elif(sampling_method == 'masking'):
          samples_generated = sample_generator.masking_sampler_using_doc_id(doc_id, self.index_reader)[1:]    #first sample is the original doc text
        elif(sampling_method == 'tfidf'):
          samples_generated = sample_generator.tfidf_sampler(doc_id, self.index_reader)[1:]    #first sample is the original doc text
        print("len(samples_generated) : ", len(samples_generated))

        print("example sample:\n", np.random.choice(samples_generated,1))


        sample_scores = sample_generator.score_samples_with_reranker(query, samples_generated, reranker)

        explanation_vectors, ranked_lists = self.get_explanation(query, doc_id, doc_score, samples_generated, sample_scores, kernel_range, document_vectors, ranker_explanation, num_top_terms, "result")

        return (explanation_vectors, ranked_lists)
        """