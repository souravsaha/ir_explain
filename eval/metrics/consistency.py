import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy, kendalltau, weightedtau
import ir_datasets
import numpy as np

class PointWiseConsistency():
	""" Pointwise consistency metric class"""
	def __init__(self, pointwise_class):
		self.pointwise_class = pointwise_class

	def consistency(self, explanation_objects, stype='kendall'):
		"""
		  Compares the relative differences in explanations for the same document accross different sampling
		  explanation_objects: list of indexes of explanation terms (in decr sorted order)
		"""
		#kendall_values = {}
		scores = []
		l1 = None
		l2 = None
		for i in range(len(explanation_objects)):
			#kendall_values[i] = {}
			l1 = explanation_objects[i]
			for j in range(i+1,len(explanation_objects)):
				l2 = explanation_objects[j]

				if len(l1) > 3 and len(l2) > 3:
					if len(l1) != len(l2):
						min_len = min(len(l1), len(l2))
						#print(len(l1), len(l2), min_len)
						l1 = explanation_objects[i][:min_len]
						l2 = explanation_objects[j][:min_len]

					if stype == 'kendall':
						kscore = kendalltau(l1,l2)
						if True:       # kscore[1] < 0.05:
							#kendall_values[i][j] = kscore[0]
							scores.append(kscore[0])
					else:
						kscore = weightedtau(l1, l2, False)
						scores.append(kscore[0])

		if len(scores)!=0:
			return np.mean(scores)
		else:
			return "Empty vector!"
	
	def evaluate(self, query_id, doc_id, explanation_vector):
		""" Compute the consistency evaluation of Pointwise approach"""
		print("Evaluation: ")
		# print(explanation_vector)
		print(self.consistency(explanation_objects = explanation_vector))