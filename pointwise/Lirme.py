import math
import operator
from scipy.stats import entropy, kendalltau,weightedtau
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PerturbDocument import PerturbDocument
from Pointwise import *
from pyserini.index.lucene import IndexReader
import ir_datasets
import lime
#upload lime_Ranker,lime_base file from local to installed lime location: /usr/local/lib/python3.10/dist-packages/lime
#import lime.lime_ranker
from lime import lime_ranker
import pandas as pd


class Lirme(Pointwise):
    def __init__(self, index_path):
        #self.index_path = index_path
        super().__init__(index_path)
        
        #Initialize IndexReader
        #self.index_reader = IndexReader(self.index_path)
        
        #harcoding msmarco IndexReader for demo
        self.index_reader = IndexReader.from_prebuilt_index('msmarco-v1-passage-full')  #added
        print("index_reader : ", self.index_reader)

        self.sampling_method = "random_words"
        self.top_terms = 10
        self.kernel = [5]

    def consistency(self, explanation_objects, stype='kendall'):
        '''
          Compares the relative differences in explanations for the same document accross different sampling
          explanation_objects: list of indexes of explanation terms (in decr sorted order)
        '''
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

        return np.mean(scores)

    """divergence_from_truth
    
    def divergence_from_truth(self, rel_vector, explain_vector):
        '''
        #measure divergence of the explanation vector from the rel/non-rel doc-vector
        '''
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

        return  pos_ent
    """
    
    """ divergence_from_truth_cosine
    def divergence_from_truth_cosine(self, rel_vector, explain_vector):
        '''
        measure cosine distance of the explanation vector from the rel/non-rel doc-vector
        '''
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

        return  pos_ent
    """

    def correctness(self, doc_id, exp_term_vec, rel_doc_vector):
        '''
        exp_term_vec: list of (term, exp_score, tfidf_score)
        '''
        ##Correctness : divergence measures
        exp_term_vec_tfidf = {}
        for vec in exp_term_vec:
            exp_term_vec_tfidf[vec[0]] = vec[2]

        #document_dict = self.get_document_vector(doc_id)

        pe = self.divergence_from_truth(rel_doc_vector,\
                        exp_term_vec_tfidf)

        cpe = self.divergence_from_truth_cosine(rel_doc_vector,\
                              exp_term_vec_tfidf)

        print("pe, cpe : ", pe, cpe)

        return (pe,cpe)


    #Not being used as of now
    """ compute_query_document_vectors
    def compute_query_document_vectors(self, qrel_path, index_reader):
        # Read the qrel file
        qrel_object = QRel()

        query_vectors = {}
        doc_counts = {}

        #ADDED FOR TESTING -- NEED TO CHANGE AND CONTROL ACCORDINGLY
        num_iterations_for_testing = 0
        for line in open(qrel_path, 'r'):
          num_iterations_for_testing+=1
          if(num_iterations_for_testing >102):
            break

          split = line.split(' ')
          query_id = split[0]
          doc_id = split[2]
          rel_label = int(split[3])
          if query_id not in query_vectors:
            query_vectors[query_id] = {}
            doc_counts[query_id] = {}

          if rel_label not in query_vectors[query_id]:
            query_vectors[query_id][rel_label] = {}
            doc_counts[query_id][rel_label] = 0.0

          qrel_object.set_rel(query_id, doc_id, rel_label)

          print("query_id, doc_id, rel_label: ", query_id, doc_id, rel_label)
          if doc_counts[query_id][rel_label] < 2:
              print("doc_counts less than 2, adding term tfids")
              doc_vector = get_document_vector(index_reader, doc_id)
              for entry , value in doc_vector.items():
                  if entry not in query_vectors[query_id][rel_label]:
                      query_vectors[query_id][rel_label][entry] = value
                  else:
                      query_vectors[query_id][rel_label][entry] += value

              doc_counts[query_id][rel_label] += 1.0

          # normalize the vectors.
        for query_id in query_vectors.keys():
            for rel_label in query_vectors[query_id].keys():
                for word in query_vectors[query_id][rel_label].keys():
                    query_vectors[query_id][rel_label][word]/=doc_counts[query_id][rel_label]


        return qrel_object, query_vectors
    """

    def get_document_vector(self, document_id):
        """
        Returns a dict of terms in the doc with their tfidf scores
        """
        #index_reader = IndexReader.from_prebuilt_index('robust04')
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


    def explanation_to_vector(self, word_list, explain_tuples, doc_vector):
        #print(explain_tuples)
        return dict([ (word_list[entry[0]] ,\
        doc_vector[word_list[entry[0]]]) for entry in explain_tuples])


    def get_explanation(self, query, doc_id, doc_score, samples, sample_scores, kernel_range, document_vectors, ranker_explanation, num_top_terms, name):
        """
        return explanation terms with their weights
        """
        explanation_vectors = []
        consistency_scores = []

        if len(samples) > 100:
          idx = np.random.choice(np.arange(len(samples)), 100, replace=False)
        else:
          idx = np.arange(len(samples))

        document_dict = self.get_document_vector(doc_id)
        word_list = list(document_dict.keys())

        explain_objects = ranker_explanation.explain_document_label(document_dict,\
                              doc_score,\
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

            print("eobject: ", eobject.as_list())
            print("exp_vector: ", exp_vector)

            term_vector = []
            for exp_score, exp_doc_tfidf in zip(exp_sorted_values, exp_vector.items()):
                #term_vector --> (term, explanation score/weight, tfidf of this term in given doc)
                term_vector.append((exp_doc_tfidf[0], exp_score[1], exp_doc_tfidf[1]))

            explanation_vectors.append({'doc_id':doc_id,'top_feat':num_top_terms,\
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

        consistency_scores.append({'doc_id':doc_id, 'query': query,\
                   'ktau': self.consistency(ranked_lists,stype='kendall'),\
                   'wktau': self.consistency(ranked_lists,stype='weighted'),\
                   'pktau': self.consistency(pos_ranked_lists,stype='kendall'),\
                   'nktau': self.consistency(neg_ranked_lists,stype='kendall'),\
                   'doc_score':doc_score,'top_feat':num_top_terms,\
                   #'doc_rel':doc_label\
                                   })

        consistency_df = pd.DataFrame(consistency_scores)
        consistency_df.to_csv(name+'_kendal_score'+str(num_top_terms)+'.csv', sep=',', index=False)

        explanation_df = pd.DataFrame(explanation_vectors)
        explanation_df.to_csv(name+'_explanation'+str(num_top_terms)+'.csv', sep=',', index=False)

        return (explanation_vectors, ranked_lists)

    def explain(self, query, doc_id,  doc_score,  reranker, params, doc_text=None):

        #kernel_range = [10,15], num_top_terms=7, sampling_method='random_words'
        #Initialize parameters:
        sampling_method = params["sampling_method"]
        num_top_terms = params["top_terms"]
        kernel_range = params["kernel_range"]

        #print("index_path: ", self.index_path)
        #index_reader = IndexReader(self.index_path)

        document_vectors = self.get_document_vector( doc_id)

        ranker_explanation = lime_ranker.LimeRankerExplainer(#kernel=uniform_kernel,\
										  kernel_width = np.sqrt(100000) * .80,\
										  random_state=123456, verbose=True, \
										  relevance_labels=[0,1,2,3,4])

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


    def find_relevant_docs(self, query_id, num_rel_docs=10, dataset_name = "msmarco-passage/trec-dl-hard"):
        '''
        '''
        dataset = ir_datasets.load(dataset_name)
        
        rel_docs_list = []
        for qrel in dataset.qrels_iter():
            if(qrel.query_id  == query_id and qrel.relevance == 1):
                print(qrel)
                rel_docs_list.append(qrel.doc_id)

            if(len(rel_docs_list) >= num_rel_docs  ):
              break
        
        return rel_docs_list
    
    def generate_ground_truth_terms(self, relevant_doc_list=None):
        '''
        relevant_doc_list: list of rel doc-ids for the given query

        compute relevant doc vectors for each rel doc: terms with their normalized tfidf values across all rel docs
        '''
        if(relevant_doc_list is None):
            #how to get query_id (and other params for this function) ?
            relevant_doc_list = find_relevant_docs(query_id)
        
        rel_doc_vectors = {}

        for doc_id in relevant_doc_list:
            print("doc_id: ", doc_id)
            doc_vector = self.get_document_vector(doc_id)
            for term , value in doc_vector.items():
                if term not in rel_doc_vectors:
                    rel_doc_vectors[term] = value
                else:
                    rel_doc_vectors[term] += value

        # normalize the vectors.
        for term in rel_doc_vectors.keys():
            rel_doc_vectors[term] /= len(relevant_doc_list)

        return rel_doc_vectors

    """
    inheriting from Pointwise
    def visualize(self, term_vectors, show_top: int=10, saveto: str='visualization.pdf'):

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
          #neg_idx = sorted_idx[:show_top]

          #words = np.append(vocabs[pos_idx], vocabs[neg_idx])
          words = vocabs[pos_idx]
          #y = np.append(pos_y, neg_y)
          y = pos_y

          fig, ax = plt.subplots(figsize=(8, 10))
          colors = ['green' if val >0 else 'red' for val in y]
          pos = np.arange(len(y)) + .5
          ax.barh(pos, y, align='center', color=colors)
          ax.set_yticks(np.arange(len(y)))
          ax.set_yticklabels(words, fontsize=10)
          ax.spines['top'].set_visible(False)
          ax.spines['right'].set_visible(False)
          ax.spines['bottom'].set_visible(False)
          ax.spines['left'].set_visible(False)
          fig.savefig(saveto)
    """

