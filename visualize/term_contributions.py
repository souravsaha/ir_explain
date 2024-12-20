import numpy as np
import matplotlib.pyplot as plt

class TermVisualization():
	"""
	Term visualization class to visualize terms and weights
	"""	
	def __init__(self):
		pass

	def visualize(self, term_vectors, show_top: int=10, show_pos_only=False, saveto: str='pointwise_visualization.pdf'):
		""" Visualize term vectors"""		  
		coef = np.array([x[1]  for x in term_vectors])
		#print("Coef: ", coef)

		vocabs = np.array([x[0]  for x in term_vectors])
		#print("vocab: ", vocabs)

		if len(coef.shape) > 1:  # binary,
			coef = np.squeeze(coef)
		
		sorted_coef = np.sort(coef)
		#print("sorted_coef: ", sorted_coef)
		sorted_idx = np.argsort(coef)
		#print("sorted_idx: ", sorted_idx)
		
		
		pos_y = sorted_coef[-show_top:]
		#print("pos_y: ", pos_y)
		neg_y = sorted_coef[:show_top]
		#print("neg_y: ", neg_y)
		pos_idx = sorted_idx[-show_top:]
		neg_idx = sorted_idx[:show_top]
		
		if(show_pos_only):
			words = vocabs[pos_idx]
			y = pos_y
		else:
			words = np.append(vocabs[pos_idx], vocabs[neg_idx])
			y = np.append(pos_y, neg_y)

		fig, ax = plt.subplots(figsize=(8, 10))
		colors = ['green' if val >0 else 'red' for val in y]
		pos = np.arange(len(y)) #+ .5
		ax.barh(pos, y, align='center', color=colors)
		ax.set_yticks(np.arange(len(y)))
		ax.set_yticklabels(words, fontsize=10)
		
		#change x label scale
		ax.tick_params(axis='x', labelsize=20)
		#define custom range on x-axis
		#plt.xlim(-20,15)  
	
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)
		fig.savefig(saveto)