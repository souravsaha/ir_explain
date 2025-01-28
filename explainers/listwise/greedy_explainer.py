from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
from typing import List, Dict, Any
from utils.rbo import RankingSimilarity
import random, sys
from queue import PriorityQueue
from utils.priority_queue import DualPriorityQueue 
import scipy.stats as stats
from utils import similarity_measures
from explainers.listwise.base_listwise import BaseListwiseExplainer

class GreedyListwiseExplainer(BaseListwiseExplainer):

    def __init__(self, index_path : str, exp_model: str, hparams: Dict[str, Any] ) -> None:
    
        self.indexer = IndexReader(index_path)

        self.greedy_vocab_terms =  hparams['GREEDY_VOCAB_TERMS']
        self.greedy_top_docs_num =  hparams['GREEDY_TOP_DOCS_NUM']
        self.greedy_max_depth =  hparams['GREEDY_MAX_DEPTH']        
        self.bfs_top_docs =  hparams['BFS_TOP_DOCS']
        self.correlation_measure = hparams['CORRELATION_MEASURE']

        if exp_model.lower() == 'bm25':
            pass
            
    
    def compute_similarity(self, bm25_hits, dr_hits):
        """
        Compute rbo between two ranked list 
        """
        p = 0.9
        depth = 10
        
        bm25_list = []
        dr_list = dr_hits[:self.bfs_top_docs]

        for i in range(0, min(self.bfs_top_docs, len(bm25_hits))):
            bm25_list.append(bm25_hits[i].docid)

        #print(bm25_list)
        #print(dr_list)

        if self.correlation_measure == "RBO":
            score = RankingSimilarity(bm25_list, dr_list).rbo(p = 0.9)

        # TODO : Problematic
        elif self.correlation_measure == "KENDALL":
            score, p_value = stats.kendalltau(bm25_list, dr_list)
        # TODO : Problematic
        elif self.correlation_measure == "SPEARMAN":
            score, p_value = stats.spearmanr(bm25_list, dr_list)
        
        elif self.correlation_measure == "JACCARD":
            score = similarity_measures.compute_jaccard(bm25_list, dr_list)


        #rbo_score = rbo.RankingSimilarity(bm25_list, dr_list).rbo(p = 0.9)
        #print(f'rbo score {rbo_score}')
        return score
    

    def _greedy(self, qid:str, query_str:str, term_weight_list:dict, searcher, dense_ranking, debug: bool):
        """
        greedy algorithm to generate the expanded query
        Paper link: https://arxiv.org/pdf/2304.12631.pdf
        """
        searcher.unset_rm3()
        # for debug purpose:
        if debug:
            print('Using Rm3? ', searcher.is_using_rm3())  # this should be false at this moment
            print(f'Entire query string {query_str}')    

        #term_weight_list_till_k = heapq.nlargest(min(len(term_weight_list), GREEDY_VOCAB_TERMS), term_weight_list)
        topk_terms = list()
        
        top_k = min(len(term_weight_list), self.greedy_vocab_terms)

        term_weight_list_till_k = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True)[:top_k])

        for term in query_str.split():
            if debug:
                print(f'running for query term {term}')

            bm25_hits = searcher.search(term)
            if debug:
                print(len(bm25_hits))
            if len(bm25_hits) == 0:
                continue
            similarity_m = self.compute_similarity(bm25_hits, dense_ranking[qid])
            #similarity_rbo = self.compute_rbo(bm25_hits, dense_ranking[qid])
            
            for key in term_weight_list_till_k:
                new_query = term + " " + key
                new_hits = searcher.search(new_query, self.greedy_top_docs_num)
                
                new_similarity = self.compute_similarity(new_hits, dense_ranking[qid])

                topk_terms.append(tuple((key, new_similarity - similarity_m)))

        topk_terms.sort(key = lambda tup: tup[1], reverse = True)
        final_query = ""

        i = 0
        while len(final_query.split()) < self.greedy_max_depth:  
            if debug:
                print(f'length of topk_terms vector {len(topk_terms)}')
                print(topk_terms)
            term, rbo_contribution = topk_terms[i]

            if term not in final_query:
                final_query = final_query + " " + term
            i = i + 1 
        
        final_hits = searcher.search(final_query)
        final_similarity = self.compute_similarity(final_hits, dense_ranking[qid])
        #final_similarity = self.compute_rbo(final_hits, dense_ranking[qid])

        return tuple((final_similarity, final_query))
    
    def explain(self, query_id:str, query_str:str, term_weight_list:dict, searcher, dense_ranking, debug) -> List[str]:
        
        best_state = self._greedy(query_id, query_str, term_weight_list, searcher, dense_ranking, debug)
        
        return best_state