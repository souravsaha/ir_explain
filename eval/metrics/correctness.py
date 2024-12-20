import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import ir_datasets


class PointWiseCorrectness():
    def __init__(self, pointwise_class):
        self.pointwise_class = pointwise_class

    def find_relevant_docs(self, query_id, dataset_name = "msmarco-passage/trec-dl-2019"):
        """ Read QRel and find relevant documents of that particular qid"""    
        dataset = ir_datasets.load(dataset_name)
        
        rel_docs_list = []
        for qrel in dataset.qrels_iter():
            if(qrel.query_id  == query_id and qrel.relevance == 1):
                print(qrel)
                rel_docs_list.append(qrel.doc_id)

            #if(len(rel_docs_list) >= num_rel_docs  ):
            #   break

        return rel_docs_list

    def generate_ground_truth_terms(self, relevant_doc_list):
        """ 
        relevant_doc_list: list of rel doc-ids for the given query
        compute relevant doc vectors for each rel doc: terms with their normalized tfidf values across all rel docs
        """
        if(relevant_doc_list is None):
            #how to get query_id (and other params for this function) ?
            # relevant_doc_list = self.find_relevant_docs(query_id)
            SystemError("Relevant document list is empty!!")

        rel_doc_vectors = {}

        for doc_id in relevant_doc_list:
            print("doc_id: ", doc_id)
            # doc_vector = self.get_document_vector(doc_id)
            # TODO: later change maybe, it's a bit hacky way
            doc_vector = self.pointwise_class.document_vector
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

    def correctness(self, doc_id, exp_term_vec, rel_doc_vector):
        '''
        exp_term_vec: list of (term, exp_score, tfidf_score)
        '''
        ##Correctness : divergence measures
        exp_term_vec_tfidf = {}
        for vec in exp_term_vec:
            exp_term_vec_tfidf[vec[0]] = vec[2]

        #document_dict = self.get_document_vector(doc_id)

        pe = self.divergence_from_truth(rel_doc_vector, exp_term_vec_tfidf)
        cpe = self.divergence_from_truth_cosine(rel_doc_vector, exp_term_vec_tfidf)

        print("pe, cpe : ", pe, cpe)
        return (pe,cpe)

    def evaluate(self, query_id, doc_id, explanation_vector):
        """ Compute the pointwise correctness metric as porposed in LIRME paper"""
        rel_docs_list = self.find_relevant_docs(query_id)
        relevant_doc_vector = self.generate_ground_truth_terms(rel_docs_list)
        return self.correctness(doc_id, explanation_vector, relevant_doc_vector)