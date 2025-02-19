from ir_explain.explainers.pointwise.base_pointwise import BasePointwiseExplainer
import numpy as np
from ir_explain.explainers.pointwise.lime import lime_ranker
from ir_explain.utils.perturb import PerturbDocument
import pandas as pd
import operator, json
from pyserini.search.lucene import LuceneSearcher

class LirmePointwiseExplainer(BasePointwiseExplainer):
	def __init__(self, model, corpus_path, indexer_type):
		# super().__init__(model)
		self.model = model
		self.corpus_path = corpus_path
		self.indexer_type = indexer_type
		self.corpus = None
		self.searcher = None

		if self.indexer_type == "no-index":
			self.corpus = self.preprocess(corpus_path)
			print(f"length of corpus : {len(self.corpus)}")
		elif self.indexer_type == "pyserini":
			self.searcher = LuceneSearcher(self.corpus_path)
	
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

	def get_explanation(self, query, doc, doc_score, samples, sample_scores, kernel_range, document_vectors, ranker_explanation, num_top_terms, name):
		r"""
		return explanation terms with their weights
		"""
		# TODO: instead of doc_id replace with doc
		explanation_vectors = []
		consistency_scores = []

		if len(samples) > 100:
			idx = np.random.choice(np.arange(len(samples)), 100, replace=False)
		else:
			idx = np.arange(len(samples))

		# document_dict = self.get_document_vector(doc_id)
		document_dict = document_vectors
		word_list = list(document_dict.keys())

		print(f"document score {doc_score}")

		explain_objects = ranker_explanation.explain_document_label(document_dict,\
								doc_score[0],\
								list(operator.itemgetter(*idx)(samples)),\
								list(operator.itemgetter(*idx)(sample_scores)),\
								num_top_terms,\
								weights_range=kernel_range)

		ranked_lists = []
		pos_ranked_lists = []
		neg_ranked_lists = []

		for eobject, kernel in zip(explain_objects, kernel_range):
			exp_sorted_values = sorted(eobject.local_exp[1], key = lambda x: np.absolute(x[1]),\
								reverse=True)
			#exp_vector is dict of (word, tfids) of words selected in explanation (sorted order)
			exp_vector = self.explanation_to_vector(word_list, exp_sorted_values,\
								document_dict)

			#print("eobject: ", eobject.as_list())
			#print("exp_vector: ", exp_vector)

			term_vector = []
			for exp_score, exp_doc_tfidf in zip(exp_sorted_values, exp_vector.items()):
				#term_vector --> (term, explanation score/weight, tfidf of this term in given doc)
				term_vector.append((exp_doc_tfidf[0], exp_score[1], exp_doc_tfidf[1]))

			# TODO: instead of doc_id replace with doc
			explanation_vectors.append({'doc_id':doc,'top_feat':num_top_terms,\
					'kernel':kernel, \
					#'doc_rel':doc_label ,\
					'doc_score': doc_score,\
					'intercept':eobject.intercept[1],
					'escore':eobject.score, \
					'local_pred': eobject.local_pred,\
					'term_vector': term_vector,\
					'query': query} )
			ranked_lists.append([entry[0] for entry in sorted(eobject.local_exp[1],\
								key = lambda x: np.absolute(x[1]),\
								reverse=True)])

			pos_ranked_lists.append([entry[0] for entry in sorted(eobject.local_exp[1],\
										key = lambda x: x[1],\
										reverse=True) if entry[1] >=0 ])

			neg_ranked_lists.append([entry[0] for entry in sorted(eobject.local_exp[1],\
										key = lambda x: x[1],\
										reverse=True) if entry[1] < 0 ])

		# TODO: instead of doc_id replace with doc

		return (explanation_vectors, ranked_lists)

	def explain(self, query, doc, params):
		print("You wanted to explain with LIRME")
		
		sampling_method = params["sampling_method"]
		num_top_terms = params["top_terms"]
		kernel_range = params["kernel_range"]

		document_vectors = self.get_document_vector(doc, self.corpus)
		
		if self.indexer_type == "pyserini":
			doc_object = self.searcher.doc(doc)
			doc = json.loads(doc_object.raw())['contents']

		ranker_explanation = lime_ranker.LimeRankerExplainer(kernel_width = np.sqrt(100000) * .80,\
										  random_state=123456, verbose=False, relevance_labels=[0, 1, 2, 3])
		
		sample_generator = PerturbDocument(200)

		if(sampling_method == 'random_words'):
			samples_generated = sample_generator.random_sampler(doc)    #first sample is the original doc text
			# samples_generated = sample_generator.random_sampler_using_doc_id(doc_id, self.index_reader)[1:]    #first sample is the original doc text
		
		elif(sampling_method == 'masking'):
			samples_generated = sample_generator.masking_sampler(doc)
			# samples_generated = sample_generator.masking_sampler_using_doc_id(doc_id, self.index_reader)[1:]    #first sample is the original doc text
		elif(sampling_method == 'tfidf'):
			# samples_generated = sample_generator.tfidf_sampler(doc_id, self.index_reader)[1:]    #first sample is the original doc text
			samples_generated = sample_generator.tfidf_sampler(doc, document_vectors) 

		print("len(samples_generated) : ", len(samples_generated))

		print("example sample:\n", np.random.choice(samples_generated,1))

		sample_scores = sample_generator.score_samples_with_reranker(query, samples_generated, self.model)
		doc_score = sample_generator.score_samples_with_reranker(query, [doc], self.model)

		print(f"document score {doc_score}")
		print(f"{len(doc_score)}")

		print(f"size of the samples {len(samples_generated)}")
		print(f"size of the sample scores {len(sample_scores)}")

		# exit(1)
		explanation_vectors, ranked_lists = self.get_explanation(query, doc, doc_score, samples_generated, sample_scores, kernel_range, document_vectors, ranker_explanation, num_top_terms, "result")
		# return "haru"
		return (explanation_vectors, ranked_lists)