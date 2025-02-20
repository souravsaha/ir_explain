from typing import Dict, List, Tuple, Union

import numpy as np
from ir_explain.explainers.pointwise.base_pointwise import \
    BasePointwiseExplainer
from ir_explain.utils.perturb import PerturbDocument
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import (LogisticRegression, SGDClassifier,
                                  SGDRegressor)
from sklearn.pipeline import Pipeline

STOP = stopwords.words('english')
import json

from pyserini.search.lucene import LuceneSearcher


class EXSPointwiseExplainer(BasePointwiseExplainer):
	"""
	EXS pointwise class to explain a single query document pair
	"""
	def __init__(
		self, 
		model, 
		corpus_path: str, 
		indexer_type: str, 
		exs_model : str = 'svm', 
		num_samples : int = 100,
		batch_size: int=10, 
		seed: int=10
		):
		# super().__init__(model)
		self.ranker = model
		self.corpus_path = corpus_path
		self.indexer_type = indexer_type
		self.corpus = None
		self.searcher = None

		self.num_samples = num_samples
		self.batch_size = batch_size
		self.seed = seed
		# this is only different from LIRME
		self.exs_model = exs_model
	
		if self.indexer_type == "no-index":
			print(f"corpus path: {self.corpus_path}")
			self.corpus = self.preprocess(corpus_path)
			print(f"length of corpus : {len(self.corpus)}")
		elif self.indexer_type == "pyserini":
			self.searcher = LuceneSearcher(self.corpus_path)

	def _set_exs_model(self, exs_model: str, method: str, seed: int=10) -> None:
		""" Update the surrogate model."""
		if exs_model == 'lr':
			return LogisticRegression(random_state=seed)
		elif exs_model == 'svm':
			if method == 'topk-bin':
				return SGDClassifier(random_state=seed)
			else:
				return SGDRegressor(random_state=seed)
		else:
			raise NotImplementedError('Only support lr and svm. :(')
	
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

	def build_input_exs(self, query, doc_id_list, rerank_score_list, rank_to_explain, doc_text_at_r):
		# build input for explaining function, use one query as example.
		exp_input = {}
		exp_input[query] = dict([(a, b) for a, b in zip(doc_id_list, rerank_score_list)])
		#for key, value in exp_input[query].items():
		#    print(key, value, '\n')
		# explain the picked doc, use the 1-th ranked doc as the baseline, use topk-bin method.
		# the returned results include {query: (words, weights)}.
		exp_doc = {query: {'rank': rank_to_explain - 1, 'text':doc_text_at_r}}

		return (exp_input, exp_doc)

	def generate_label(self, query: str, rank_orig: int, all_docs: List[str], scores_topk: List[float], method: str='topk-bin') -> np.array:
		""" Generate labels for all perturbed docs, based on three methods described in the paper.
			Args:
				query: raw-text query.
				rank_orig: the rank of the doc to be explained.
				all_docs: a list of perturbed docs based on doc_exp.
				scores_topk: the topk ranking scores in descending order by the ranker.
		"""
		#input_pairs = [[query, doc] for doc in all_docs]
		#rank_scores = self.ranker.predict(input_pairs, batch_size=self.batch_size)

		#using PerturbDoc class
		rank_scores = PerturbDocument().score_samples_with_reranker(query, all_docs, self.ranker)
		score_l = scores_topk[-1]  # lowest score.
		score_h = scores_topk[0]   # highest score.

		""" According to the paper, score_based and rank_based generate a float instead of binary labels."""
		if method == 'topk-bin':
			labels = (rank_scores > score_l).astype(int)
		elif method == 'score':
			labels = 1 - (score_h - rank_scores)/score_h
		elif method == 'rank':
			scores_new = np.tile(scores_topk, (len(rank_scores), 1))
			scores_new[:, rank_orig] = rank_scores
			orders = np.argsort(-scores_new)
			rank_new = orders.argsort()[:, rank_orig]
			labels = 1 - rank_new/len(scores_topk)
			pass
		else:
			raise ValueError('Invalid method.')
		return labels

	def explain_single(self, query: str, doc_h: Tuple[int, str], scores_topk: List[float], method: str, seed: int) -> Tuple[np.array, np.array]:
		""" explain a single doc for a single query.
			Args:
				query: raw-text query. NOT query_id!
				doc_h: the doc to be explained, including the rank and the text of the doc, it should have higher rank than k. raw text!
				scores_topk: the list of topk ranking scores by ranker. In descending order.
				method: three methods for generating labels for perturbed docs to train the surrogate model.

		"""
		rank_orig, doc_orig = doc_h

		#perturb docs using PerturbDoc class
		PerturbDoc = PerturbDocument(self.num_samples)
		print(f"document orig: {doc_orig}")
		print(f"document orig: {type(doc_orig)}")
		docs_perturb = PerturbDoc.random_sampler(doc_orig["contents"])

		#print("Inside def explain_single, docs_perturb:\n",docs_perturb)  ##added by Harsh
		lables_perturb = self.generate_label(query, rank_orig, docs_perturb, scores_topk, method)
		print("lables_perturb unique: ", np.unique(lables_perturb))
		""" Learn a simpler model"""

		print("Inside def explain_single, self.exs_model: ", self.exs_model) ##added by Harsh

		clf = Pipeline([('vect', CountVectorizer()),
						('tfidf', TfidfTransformer()),
						('clf', self._set_exs_model(self.exs_model, method, seed))])

		clf.fit(docs_perturb, lables_perturb)
		coef = clf['clf'].coef_.copy()
		vocabs = np.array(clf['vect'].get_feature_names_out())
		return vocabs, coef

	def explain(self, query, doc_at_r, Method: Union[str, Dict[str, str]], params: Dict, seed: int=10) -> Dict[str, np.array]:
		""" 
		Explain the rank for a group of queries.
		
		Args:
			corpus: the query-doc-rerank datasets. The format should follow:
				{'query1': {'doc1': rel_score, 'doc2': rel_score2,...}, 'query2': {...}, ...}
			docs_exp: the doc to be explained. {query: {'text':which_doc, 'rank':0}}, this doc has to be the raw text.
			topk: integer, the baseline doc's rank which is used to explain the doc in doc_ids

		"""
		doc_ids_array = params["doc_ids"]
		rerank_scores_array = params["rerank_scores"]
		topk = params["rank"]

		inp_exs = self.build_input_exs(query, doc_ids_array, rerank_scores_array, topk, doc_at_r)
		
		corpus = inp_exs[0]
		docs_exp = inp_exs[1]
		
		Results = {}
		for query in corpus:
			if isinstance(topk, dict):  # choose different baseline doc for each query.
				k = topk[query]
			else:
				k = topk
			if isinstance(Method, dict):  # choose different method for each query.
				method = Method[query]
			else:
				method = Method
			doc_rank, doc_exp = docs_exp[query]['rank'], docs_exp[query]['text']   # the doc to be explained.
			assert(doc_rank < k)
			docs_sorted = sorted(corpus[query].items(), key=lambda item: item[1], reverse=True)[:k]
			scores_topk = [d[1] for d in docs_sorted]

			print("Inside def explain, scores_topk:\n",scores_topk)  ##added by Harsh
			print("***********DEBUG****************")
			print("query: ", query)
			print("doc_rank: ", doc_rank)
			print("doc_exp: ", doc_exp)
			print("scores_topk: ", scores_topk)
			print("method: ", method)
			print("seed: ", seed)
			print("***********DEBUG****************")

			vectors, coef = self.explain_single(query, (doc_rank, doc_exp), scores_topk, method, seed)
			if self.indexer_type == "pyserini":
				document_vectors = self.get_document_vector(doc_exp['id'], self.corpus)
			elif self.indexer_type == "no-indexer":	
				document_vectors = self.get_document_vector(doc_exp['contents'], self.corpus)
			word_list = list(document_vectors.keys())

			term_weight_list = []
			for item in zip(vectors, coef[0]):
				print(f"item {item}")
				index = word_list.index(item[0])
				term_weight_list.append(tuple((index, item[1])))
			
			term_weight_list.sort(key=lambda tup: tup[1], reverse=True)
			explanation_vectors = []
			print(f"term_weight_list : {term_weight_list}")
			exp_vector = self.explanation_to_vector(word_list, term_weight_list, document_vectors)

			term_vector, ranked_lists = [], []
			for exp_score, exp_doc_tfidf in zip(term_weight_list, exp_vector.items()):
				term_vector.append((exp_doc_tfidf[0], exp_score[1], exp_doc_tfidf[1]))

			# TODO: we can fill up other stuffs here, but for now it's not required
			explanation_vectors.append({
				'term_vector': term_vector,
				'query': query
			})
			temp_list = []
			for items in term_weight_list:
				temp_list.append(items[0])
			ranked_lists.append(temp_list)
			# Results[query] = results_exp
		return (explanation_vectors, ranked_lists)
