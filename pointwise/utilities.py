import numpy as np
from scipy.stats import kendalltau

def get_results_from_index(query, index_searcher, num_docs=10):
    #Perform sparse retrieval to get 'num_docs' number of documents from the index specified by 'index_searcher'
    
    hits = index_searcher.search(query, num_docs)
    docs = []
    doc_ids = []
    retrieved_scores = []
    for hit in hits:
        #doc_text = (hit.docid + " " + hit.raw.replace("\n"," ")).strip()
        #print(hit.contents)
        #doc_text = hit.contents() # .replace("\n"," ").strip()
        doc_text = hit.lucene_document.get('raw').replace("\n"," ").strip()
        docs.append(doc_text)
        doc_ids.append(hit.docid)
        retrieved_scores.append(hit.score)
    print(docs)
    print(doc_ids)

    retrieved_dict = {'doc_ids' : np.array(doc_ids), 'docs' : np.array(docs), 'retrieved_scores' : np.array(retrieved_scores)}

    return retrieved_dict

def kendall_tau_two_word_lists(list1, list2):
    #assuming lists have same words in arbitrary orders, compute kendall tau between ranks of words in the two lists.
    
    rank_map1 = {word: rank for rank, word in enumerate(list1)}
    ranks1 = list(range(len(list1)))
    ranks2 = [rank_map1[word] for word in list2]
    
    tau, p_value = kendalltau(ranks1, ranks2)
    return tau, p_value


def normalize_scores_by_total(score_dict):
    #Given a term:score dict, normalize the scores by dividing the value by sum of scores and return the normalized term:score dict
    
    # Calculate the total score
    total_score = sum(score_dict.values())
    
    # Normalize each score by dividing by the total score
    normalized_scores_dict = {term: score / total_score for term, score in score_dict.items()}
    
    return normalized_scores_dict


def normalize_scores_by_min_max(score_dict):
    #Given a term:score dict, normalize the scores using min and max values and return the normalized term:score dict
    
    scores = list(score_dict.values())
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle the case where all scores are the same
    if min_score == max_score:
        return {term: 0.5 for term in score_dict}
    
    normalized_scores_dict = {
        term: (score - min_score) / (max_score - min_score)
        for term, score in score_dict.items()
    }
    
    return normalized_scores_dict


def compute_explanation_similarity(term_vec1,term_vec2):
    #THIS IS THE PROPOSED METRIC
    #higher value means more dis-similar
    
    #Normalize scores 
    term_vec1_norm = normalize_scores_by_min_max(term_vec1)
    term_vec2_norm = normalize_scores_by_min_max(term_vec2)
    
    score = 0
    num_common_terms = 0
    for term in term_vec1.keys():
        if term in term_vec2.keys():
            print(term, term_vec1[term] , term_vec2[term])
            score += abs(term_vec1[term] - term_vec2[term])
            num_common_terms+=1
    
    num_unique_terms = len(list(set(list(occ_term_vec_norm.keys()) + list(exs_term_vec_norm.keys()))))
    normalizing_ratio = num_common_terms/num_unique_terms
    
    score = score/normalizing_ratio
    #print("similarity score: ", score)
    return score

