import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from beir.reranking.models import CrossEncoder
import re
import pandas as pd

from EXS import ExplainableSearch
from utilities import *


model = 'cross-encoder/ms-marco-electra-base'
reranker = CrossEncoder(model)
explainer = ExplainableSearch(reranker, 'svm', 100)

index_path = "/b/administrator/collections/indexed/msmarco-v1-passage-full"
searcher = LuceneSearcher(index_path)

dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")


query_list = []
query_id_list = []
for TrecQuery in dataset.queries_iter():
    query_list.append(TrecQuery.text)
    query_id_list.append(TrecQuery.query_id)


num_queries = 5
num_top_docs = 3
expln_occlusion_consistency_list = []

for idx, query_id in enumerate(query_id_list[:num_queries]):
    query = query_list[idx]
    print("query: ", query)
    retrieved_dict = get_results_from_index(query, searcher, num_docs=10)
    
    doc_ids = retrieved_dict["doc_ids"]
    docs = retrieved_dict["docs"]
    
    #Rerank using the Neural model
    rerank_scores = reranker.predict(list(zip([query]*len(doc_ids),docs)), batch_size=10)
    retrieved_dict["rerank_scores"] = rerank_scores
    
    docids_reranked = retrieved_dict["doc_ids"][np.argsort(retrieved_dict["rerank_scores"])[::-1]]  # descending order.
    
    for rank in range(1, num_top_docs + 1):
        doc_id_at_r = docids_reranked[rank-1]
        doc_at_r = retrieved_dict["docs"][np.argsort(retrieved_dict["rerank_scores"])][-rank]
        doc_score_at_r = np.sort(retrieved_dict["rerank_scores"])[-rank]
        
        #print("doc_at_r:")
        #print(doc_at_r)
        
        try:
            results = explainer.explain(query, retrieved_dict["doc_ids"], retrieved_dict["rerank_scores"], rank, doc_at_r, 'topk-bin')
            results_no_stopword = explainer.remove_stopwords_from_explanation(results, stopword_file_path = "stop.txt")
        except:
            pass
        topk = 10
        vocabs = results_no_stopword[query][0]
        coef = results_no_stopword[query][1]
        
        sorted_idx = np.flip(np.argsort(coef))
        
        topk_words_idx = sorted_idx[:topk]
        topk_words = vocabs[topk_words_idx]
        topk_words_scores = coef[topk_words_idx]
        
        print("topk_words, coef: ", topk_words, topk_words_scores)
        
        word_importance_vector = get_occlusion_word_importance_vector(query,doc_at_r, reranker, doc_score_at_r, topk_words)
        
        tau = kendall_tau_two_word_lists(topk_words, [x[0] for x in word_importance_vector])
        
        #print([x[0] for x in word_importance_vector])
        print("tau: " , tau)
        
        #print(word_change_in_score_dict)
        #print(dict(word_importance_vector[:]))
        
        #print(dict(zip(topk_words, topk_words_scores)))    
        
        #print("NORMALIZED")
        exs_term_vec_norm = normalize_scores_by_min_max(dict(zip(topk_words, topk_words_scores)))
        occ_term_vec_norm = normalize_scores_by_min_max(dict(word_importance_vector[:]))
        #print(exs_term_vec_norm)
        #print(occ_term_vec_norm)
        
        dissim_score = compute_explanation_similarity(exs_term_vec_norm,occ_term_vec_norm)
        print("dissim score: ", dissim_score)
        
        print("\n\n")
        expln_occlusion_consistency_list.append([query_id, doc_id_at_r, tau[0], dissim_score])


exs_occl_consistency_df = pd.DataFrame(expln_occlusion_consistency_list, columns=["query_id", "doc_id", "kendall tau coefficient", "dissim"])

print(exs_occl_consistency_df)