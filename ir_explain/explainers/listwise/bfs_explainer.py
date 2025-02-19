from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
from typing import List, Dict, Any, Callable
from ir_explain.utils.rbo import RankingSimilarity
import random, sys
from queue import PriorityQueue
from ir_explain.utils.priority_queue import DualPriorityQueue 
import scipy.stats as stats
from ir_explain.utils import similarity_measures
from ir_explain.explainers.listwise.base_listwise import BaseListwiseExplainer

class BFSListwiseExplainer(BaseListwiseExplainer):
    """
    Implementation of the BFS algorithm
    Paper link : https://dl.acm.org/doi/abs/10.1145/3539618.3591982
    """
    def __init__(self, index_path : str, indexer_type: str, exp_model: str, hparams: Dict[str, Any] ) -> None:
        self.indexer = IndexReader(index_path)

        self.queue_max_depth =  hparams['QUEUE_MAX_DEPTH']
        self.bfs_max_exploration =  hparams['BFS_MAX_EXPLORATION']
        self.bfs_vocab_terms =  hparams['BFS_VOCAB_TERMS']        
        self.bfs_max_depth =  hparams['BFS_MAX_DEPTH']
        self.bfs_top_docs =  hparams['BFS_TOP_DOCS']
        self.correlation_measure = hparams['CORRELATION_MEASURE']
        # as of now this is dummy, we do not process anything
        self.indexer_type = indexer_type
        if exp_model.lower() == 'bm25':
            pass
    
    
    def compute_similarity(self, bm25_hits, dr_hits):
        """
        Compute similarity measure (e.g., rbo, jaccard) between two ranked list 
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

        #score = rbo.RankingSimilarity(bm25_list, dr_list).rbo(p = 0.9)
        #if debug:
        #    print(f'rbo score {rbo_score}')
        return score

    def _sample_terms(self, term_weight_list, num_terms):
        """
        Sample terms from the feedback terms
        term_weight_list : sorted by weights
        """
        total_weight = 0                                # ideally total_weight should be 1
        for value in term_weight_list.values():
            total_weight += value
        terms_list = []

        #print(f'Sampling terms...')
        while len(terms_list) < num_terms and len(terms_list) < len(term_weight_list):
            index = random.uniform(0, 1)*total_weight

            for term, weight in term_weight_list.items():
                index = index - weight

                if index <= 0 :
                    if term not in terms_list:
                        terms_list.append(term)
                        break
        
        #print('Sampling completed...')
        return terms_list


    def _bfs(self, qid, query_str, term_weight_list, searcher, dense_ranking, debug: bool):
        """
        bfs algorithm to generate the expanded query
        Paper link: https://arxiv.org/pdf/2304.12631.pdf
        """
        best_state = None
        searcher.unset_rm3()
        if debug:
            print('Using Rm3? ', searcher.is_using_rm3())  # this should be false at this moment
            print(f'Entire query string {query_str}')    
        
        for term in query_str.split():
            if debug:
                print(f'running for query term {term}')
                print(f'original qid {qid}\nquery {query_str}')
            
            maxQ = DualPriorityQueue(self.queue_max_depth, maxPQ=True)
            bm25_hits = searcher.search(term)
            if debug:
                print(len(bm25_hits))
            if len(bm25_hits) == 0:
                continue
            
            similarity_m = self.compute_similarity(bm25_hits, dense_ranking[qid]) 

            if debug:
                print(f'similarity score {similarity_m}:{self.correlation_measure}')
            initial_state = tuple((similarity_m, term))                   

            if best_state is None:
                best_state = initial_state
            
            maxQ.put(initial_state)
            states_explored = 0
            if debug:
                print(f'queue at start : {dict(maxQ.queue)}')
            while (not maxQ.empty()) and (states_explored < self.bfs_max_exploration):
                current_best = maxQ.get()
                states_explored += 1
                
                if current_best[0] > best_state[0]:
                    best_state = current_best
                
                # sample 30 terms from the feedbackDocs
                sampled_terms = self._sample_terms(term_weight_list, self.bfs_vocab_terms)
                if debug:
                    print(f'size of sampled terms {len(sampled_terms)}')
                #print(sampled_terms)
                
                for vocab_term in sampled_terms:
                    new_query = ""

                    #print(f"term want to add {vocab_term}")
                    if vocab_term not in current_best[1]:
                        new_query = current_best[1] + " " + vocab_term
                    else:
                        new_query = current_best[1]
                    
                    if debug:
                        print(f"new query : {new_query}")
                        print(f'queue : {dict(maxQ.queue)}')
                    if  new_query not in dict(maxQ.queue) and len(new_query.split()) < self.bfs_max_depth:
                        # retrieve with the new expanded query
                        new_top_docs = searcher.search(new_query, self.bfs_top_docs)

                        new_similarity_m = self.compute_similarity(new_top_docs, dense_ranking[qid])

                        if new_similarity_m >= current_best[0] and new_similarity_m > 0:
                            element = tuple((new_similarity_m, new_query))
                            #print(f"adding {element} to the queue")
                            
                            #print(f"Size of queue {len(dict(q.queue))} earlier")
                            #print(f"States explored {states_explored}")
                            maxQ.put(element)
                            #print(f"Size of queue {len(dict(q.queue))} later")
                            #print(f'queue : {dict(q.queue)}')

                        if new_similarity_m > best_state[0]:
                            best_state = tuple((new_similarity_m, new_query))
                            if debug:
                                print(f'best state as of now {best_state}')
            if debug:
                print(f'max exploration done {states_explored}')
        
        return best_state


    def explain(self, query_id:str, query_str:str, term_weight_list:dict, searcher, dense_ranking, debug) -> List[str]:
        
        best_state = self._bfs(query_id, query_str, term_weight_list, searcher, dense_ranking, debug)
        return best_state