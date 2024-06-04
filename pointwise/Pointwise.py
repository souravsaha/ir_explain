import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Pointwise():

    def __init__(self, index_path=None):
        self.index_path = index_path


    def get_document_vector(self, index_reader, document_id):
        """
        Returns a dict of terms in the doc with their tfidf scores
        """
        print("get_document_vector inside pw class")
        #index_reader = IndexReader.from_prebuilt_index('robust04')
        tf = index_reader.get_document_vector(document_id)
        df = {term: (index_reader.get_term_counts(term, analyzer=None))[0] for term in tf.keys()}
        tot_num_docs_idx = index_reader.stats()['documents']
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


    def generate_ground_truth_terms(self, relevant_doc_list):
        """
        relevant_doc_list: list of rel doc-ids for the given query

        compute relevant doc vectors for each rel doc: terms with their normalized tfidf values across all rel docs
        """
        rel_doc_vectors = {}

        for doc_id in relevant_doc_list:
            print("doc_id: ", doc_id)
            doc_vector = self.get_document_vector(self.index_reader, doc_id)
            for term , value in doc_vector.items():
                if term not in rel_doc_vectors:
                    rel_doc_vectors[term] = value
                else:
                    rel_doc_vectors[term] += value

        # normalize the vectors.
        for term in rel_doc_vectors.keys():
            rel_doc_vectors[term] /= len(relevant_doc_list)
        
        return rel_doc_vectors


    def divergence_from_truth(self, rel_vector,  explain_vector):
        """
        measure divergence of the explanation vector from the rel/non-rel doc-vector
        """
        # find the common words in all three vectors.
        pos_all_vect = pd.DataFrame({'rel_vector': rel_vector,
                'explain_vector': explain_vector}).fillna(0.0001)

        #neg_all_vect = pd.DataFrame({'non_rel_vector':non_rel_vector,\
        #        'explain_vector': explain_vector}).fillna(0.0001)


        #print(all_vectors[all_vectors['explain_vector'] > 0].head())

        #norm_df = all_vectors.apply(lambda x: x/x.max(), axis=0)

        pos_ent = entropy(pos_all_vect['rel_vector'].values,pos_all_vect['explain_vector'].values )

        #neg_ent = entropy(neg_all_vect['non_rel_vector'].values,neg_all_vect['explain_vector'].values )
        #print(neg_ent, pos_ent, all_vectors.shape)

        '''
        # Compute KL Divergence between relevant and non-relevant
        neg_ent =  cosine_similarity([all_vectors['non_rel_vector'].values],\
            [all_vectors['explain_vector'].values])
        pos_ent =  cosine_similarity([all_vectors['rel_vector'].values], \
                      [all_vectors['explain_vector'].values])
        #print(neg_ent, pos_ent, len(explain_vector), all_vectors.shape)
        '''

        #return  neg_ent, pos_ent, neg_ent/pos_ent
        return  pos_ent

    def divergence_from_truth_cosine(self, rel_vector,  explain_vector):
        """
        measure cosine distance of the explanation vector from the rel/non-rel doc-vector
        """
        # find the common words in all three vectors.
        pos_all_vect = pd.DataFrame({'rel_vector': rel_vector,
                'explain_vector': explain_vector}).fillna(0.0001)

        #neg_all_vect = pd.DataFrame({'non_rel_vector':non_rel_vector,\
        #        'explain_vector': explain_vector}).fillna(0.0001)

        # Compute Cosine between relevant and non-relevant
        #neg_ent =  cosine_similarity([neg_all_vect['non_rel_vector'].values],\
        #    [neg_all_vect['explain_vector'].values])

        pos_ent =  cosine_similarity([pos_all_vect['rel_vector'].values], \
                      [pos_all_vect['explain_vector'].values])

        #return  neg_ent, pos_ent, (1.0- neg_ent)/(1.0 - pos_ent)
        return  pos_ent

    def explanation_to_vector(self, word_list, explain_tuples, doc_vector):
        #print(explain_tuples)
        return dict([ (word_list[entry[0]] ,\
        doc_vector[word_list[entry[0]]]) for entry in explain_tuples])


    def compare_with_ground_truth(self, doc_id, term_weight_list, relevant_doc_list):
        """
        term_weight_list : list of tuples of terms and weights in sorted order
        relevant_doc_dict: dict containing list of rel/non-rel doc-ids for the given query
        or
        relevant_doc_list: list of rel doc-ids for the given query

        generates consistency and correctness metrics with the ground truth
        """

        rel_doc_vector = self.compute_relevant_doc_vectors(relevant_doc_list)

        document_dict = self.get_document_vector(self.index_reader, doc_id)
        word_list = list(document_dict.keys())

        #Consistency : kendall tau

        ##Correctness : divergence measures
        pe = self.divergence_from_truth(rel_doc_vector,\
                        self.explanation_to_vector(word_list,\
                                    eobject.local_exp[1],\
                        document_dict))

        cpe = self.divergence_from_truth_cosine(rel_doc_vector,\
                              self.explanation_to_vector(word_list,\
                                        eobject.local_exp[1],\
                                        document_dict))

        print("pe, cpe : ", pe, cpe)

        return (pe,cpe)


    def consistency_among_explanations(self, term_weight_lists):
         """
        term_weight_lists : list of lists of tuples of terms and weights in sorted order

        generates consistency amongst the different explanations generated
        """

    def visualize(self, term_vectors, show_top: int=10, saveto: str='pointwise_visualization.pdf'):

          coef = np.array([x[1]  for x in term_vectors])
          print("Coef: ", coef)

          vocabs = np.array([x[0]  for x in term_vectors])
          print("vocab: ", vocabs)



          if len(coef.shape) > 1:  # binary,
              coef = np.squeeze(coef)
          sorted_coef = np.sort(coef)
          print("sorted_coef: ", sorted_coef)
          sorted_idx = np.argsort(coef)
          print("sorted_idx: ", sorted_idx)
          pos_y = sorted_coef[-show_top:]
          print("pos_y: ", pos_y)
          neg_y = sorted_coef[:show_top]
          print("neg_y: ", neg_y)
          pos_idx = sorted_idx[-show_top:]
          neg_idx = sorted_idx[:show_top]

          words = np.append(vocabs[pos_idx], vocabs[neg_idx])
          #words = vocabs[pos_idx]
          y = np.append(pos_y, neg_y)
          #y = pos_y

          fig, ax = plt.subplots(figsize=(8, 10))
          colors = ['green' if val >0 else 'red' for val in y]
          pos = np.arange(len(y)) #+ .5
          ax.barh(pos, y, align='center', color=colors)
          ax.set_yticks(np.arange(len(y)))
          ax.set_yticklabels(words, fontsize=10)
          
          #change x label scale
          ax.tick_params(axis='x', labelsize=50)
          #define custom range on x-axis
          #plt.xlim(-20,15)  
        
          ax.spines['top'].set_visible(False)
          ax.spines['right'].set_visible(False)
          ax.spines['bottom'].set_visible(False)
          ax.spines['left'].set_visible(False)
          fig.savefig(saveto)