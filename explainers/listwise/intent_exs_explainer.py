import numpy as np
from typing import Any, Callable, Dict, Tuple, List, Union
from pyserini.index.lucene import IndexReader
import math, random, nltk 
from functools import partial
from itertools import combinations
from pyserini.analysis import Analyzer, get_lucene_analyzer
import gensim.downloader as api
from collections import Counter
from datasets.dataIterDrmm import matching_histogram_mapping
from explainers.listwise.base_listwise import BaseListwiseExplainer

pretrained_embeddings = 'glove-wiki-gigaword-100'
Word_Vectors = api.load(pretrained_embeddings)


# generate candidates
# generate pairs
# generate matrix
# optimize : 1. Greedy 2. Geno

class IntentListwiseExplainer(BaseListwiseExplainer):
    """
    Implementation of "Model Agnostic Interpretability of Rankers via Intent Modelling".

    Paper reference : https://dl.acm.org/doi/pdf/10.1145/3351095.3375234
    """
    def __init__(self, ranker: Callable, index_path: str, indexer_type: str, exp_model: str, seed: int=10) -> None:
        """ Init the ranking model to be explained, a pre-computed index, a simple explaination model, only support BM25."""
        self.ranker = ranker
        # TODO: what if the indexing is done with pyterrier
        self.indexer = IndexReader(index_path)  # init with a pre-computed index.
        # as of now this is dummy, we do not process anything
        self.indexer_type = indexer_type
        # TODO : support for other statistical models like LMJM, LMDIR, ...
        if exp_model.lower() == 'bm25':
            self.exp_model = self._bm25_model
        elif exp_model.lower() == 'lmjm':
            pass
        elif exp_model.lower() == 'lmdir':
            pass
        elif exp_model.lower() == 'saliency':
            self.exp_model = self._saliency
        elif exp_model.lower() == 'semantic':
            self.exp_model = self._semantic
        else:
            raise NotImplementedError(f'Only support bm25.')

        self.seed = seed
        self. _gen_candidates = partial(gen_candidates, self.indexer, self.ranker)
        self._gen_pairs = partial(gen_pairs, seed=self.seed)
        self._gen_matrix = partial(gen_matrix, self.exp_model)
        self._optimize = greedy

    def _bm25_model(self, doc_id: str, term: str, doc, analyzer, max_token: int = 510):
        """ Use BM25 model as the explainer."""
        return self.indexer.compute_bm25_term_weight(doc_id, term, analyzer=None)
    
    
    def _saliency(self, doc_id, token: Union[str, List[str]], doc: str, analyzer, max_tokens: int = 510) -> Union[float, List[float]]:
        doc = doc.lower()
        def analyze_sent(doc: str):
            sents = nltk.sent_tokenize(doc)
            if analyzer:
                tokens = analyzer.analyze(doc)
                sents_analyzed = [Counter(analyzer.analyze(s)) for s in sents]
            else:
                #tokens = nltk.word_tokenize(doc)   # nltk tokenization
                sents_analyzed = [Counter(nltk.word_tokenize(doc)) for s in sents]    
                #sents_analyzed = [Counter(analyzer.analyze(s)) for s in sents]
            return sents_analyzed

        analyzed = analyze_sent(doc)

        def single(token, analyzed):
            tfs = [s[token] for s in analyzed]
            sent_freq = np.sign(tfs).sum() / float(len(tfs))
            if sent_freq > 0:
                tfs = [(s_i, tf) for s_i, tf in enumerate(tfs) if tf>0]
                #if norm:
                    #term_freq = math.log(sum([tf ** (1 / (1 + s)) for s, tf in tfs]) + 1)
                #else:
                term_freq = math.log(sum([2 ** (1 / (1 + s)) for s, tf in tfs]) + 1)  # ignore concrete term frequency value.
                return sent_freq * term_freq   # > 0 
            else:
                return 0   # token doesn't exist.

        if isinstance(token, str):
            token = [token]
        scores = [single(tok, analyzed) for tok in token]
        if len(scores) == 1:
            return scores[0]
        else:
            return scores

    def _semantic(self, doc_id, token: Union[str, List[str]], doc: str, analyzer, max_tokens: int=510) -> Union[float, List[float]]:
        """ too  slow, change to doc centered."""
        doc = doc.lower()
        def word2vec(vectors, token):
            try:
                vec = vectors[token]
            except:
                vec = np.array([None])
            return vec
        
        def similarity(t_vec, d_vec, num_bins=20):
            if t_vec.any():
                hist = matching_histogram_mapping([t_vec], d_vec, num_bins=num_bins)
                hist = hist[0]
                similarity = sum([hist[i]*i for i in range(len(hist))])/len(emb_doc)/(len(hist)-1) 
            else:
                similarity = 0
            return similarity

        if analyzer:
            emb_doc = [word2vec(Word_Vectors, t) for t in analyzer.analyze(doc) if word2vec(Word_Vectors, t).any()]
        else:
            emb_doc = [word2vec(Word_Vectors, t) for t in nltk.word_tokenize(doc) if word2vec(Word_Vectors, t).any()]
        
        if isinstance(token, str):
            token = [token]
        emb_token = [word2vec(Word_Vectors, t) for t in token]
        similarities = [similarity(emb_t, emb_doc) for emb_t in emb_token]
        if len(similarities) == 1:   # only a single word/value
            return similarities[0]
        else:
            return similarities


    def explain(self, corpus: Dict[str, Any], params: Dict[str, Union[int, str]]) -> List[str]:
        """ 
        A pipeline including candidates, matrix generation, and extract intent using greedy algorithm.
        
            Args:
                corpus: dict-type input data, must have query, scores, docs domains. 
                        e.g. {'query': xxx, 'scores': {'doc_id': score}, 'docs': {'doc': doc_text}}.
                params: necessary parameters needed for candidates gen, matrix gen... 
                        e.g. {'top_idf': 10, 'topk': 5, 'max_pair': 100, 'max_intent': 10}
            
            Returns:
                A list of terms/words/tokens, used as expansion/intent/explanation.
        """
        # descending order e document id gulo sorted in doc_ids  
        doc_ids = [score[0] for score in sorted(corpus['scores'].items(), key=lambda item: item[1], reverse=True) ] # sorted in descending order.
        candidates = self._gen_candidates(corpus, params['top_idf'])
        self.candidates = candidates
        doc_pairs = self._gen_pairs(params['style'], len(doc_ids), params['topk'], params['max_pair'])
        #print(f'sampled pairs: {len(doc_pairs)}')
        matrix = self._gen_matrix(doc_ids, candidates, doc_pairs, corpus)
        self.matrix = matrix
        expansion = self._optimize(candidates, matrix, params['max_intent'])
        return expansion


def gen_candidates(indexer: IndexReader, ranker: Callable, corpus: Dict[str, Dict[str, Any]], top_idf: int=10) -> List[str]:
    """ Generates candidate tokens for documents of a query."""
    query = corpus['query']
    doc_ids = list(corpus['scores'].keys())
    candidates = []
    for doc_id in doc_ids:
        #print(doc_id)
        doc = corpus['docs'][doc_id]
        score = corpus['scores'][doc_id]
        #print(score)
        score = float(score)
        # get the terms sorted by tf-idf scores for a particular doc 
        terms_sorted = terms_tfidf(indexer, doc_id)[:2*top_idf]   # keep 2*top_idf terms for perturbation for now.
        new_input_pairs = [(query, doc_perturb(doc, term[0])) for term in terms_sorted]
        # individual term gulo diye predict korchi; store it in the scores_new 
        scores_new = ranker.predict(new_input_pairs)
        # sort korchi based on the score_new - score 
        term_idx = np.argsort(-np.abs(scores_new - score))[:top_idf]   # descending order.
        terms_select = np.array([term[0] for term in terms_sorted])[term_idx].tolist()
        candidates.extend(terms_select)
    candidates = list(set(candidates))
    return candidates


def gen_pairs(style: str, length: int, topk: int, max_pair: int, seed: int) -> List[Tuple[int, int]]:
    """ Sample document pairs by rank. e.g., [(0, 5), (1, 9), (3, 6),...]
        Style: 
        'random' randomly choose mC2 pairs
        'topk_random' one from top-order and another from the bottom order
        'topk_rank_random' 
        length: length of the list
        topk: 5
        max_pair: how many pairs you want to generate 

    """
    # params['style'], len(doc_ids), params['topk'], params['max_pair']
    if style == 'random':
        pairs = list(combinations(range(length), 2))
    elif style == 'topk_random':
        assert(topk <= length)
        ranked_list = list(range(topk))
        tail_list = list(range(topk, length))
        pairs = [(a, b) for a in ranked_list for b in tail_list]
    elif style == 'topk_rank_random':
        pairs = list(combinations(range(topk), 2))
    else:
        raise ValueError(f'Not supported style {style}')
    
    if len(pairs) < max_pair:
        max_pair = len(pairs)
    random.seed(seed)    
    pairs = random.sample(pairs, max_pair)
    return pairs


def gen_matrix(exp_model: Callable, doc_ids: List[str], candidates: List[str], pairs: List[Tuple], corpus: Dict[str, Dict[str, Any]]) -> np.array:
    """ Generate the matrix, given candidates and sampled document pairs.
        exp_model : simple explanation model like BM25, LMDIR, etc...
        doc_ids: list of documents ids
        candidates: expanded terms/ candidate terms (column of the matrix) 
        pairs : {d_i, d_j} pairs (row of the matrix)
    """
    matrix = []
    idx_set = set([p for pair in pairs for p in pair])
    # compute all bm25 scores.
    # TODO : plug different scoring fuctions
    BM25_scores = {}
    for idx in idx_set:
        doc_id = doc_ids[idx]
        doc = corpus['docs'][doc_id]
        BM25_scores[doc_id] = np.array([exp_model(doc_id, term, doc, analyzer = None) for term in candidates])

    for rank_h, rank_l in pairs:
        docid_h, docid_l = doc_ids[rank_h], doc_ids[rank_l]
        bm25_scores_h, bm25_scores_l = BM25_scores[docid_h], BM25_scores[docid_l]
        column = (bm25_scores_h - bm25_scores_l) * (1 + math.log(rank_l - rank_h))
        matrix.append(column)
    
    matrix = np.array(matrix).transpose(1, 0)   # terms at dimension first. 
    return matrix


def greedy(candidates: List[str], matrix_arg: np.array, select_max: int) -> List[str]:
    """
    Solving the optimization with greedy
    """
    matrix = matrix_arg.tolist()  # copy the argument, otherwise it'll be modified.     
    #print(matrix.shape)
    expansion = []
    pcov = np.zeros(len(matrix[0]))   # init expansion and features
    for _ in range(select_max):
        covered = (pcov > 0.0).sum()
        pcov_update = pcov.copy()
        picked = None   # in case no candidate can be found.
        for candidate, array in zip(candidates, matrix):
            pcov_expand = pcov + array   
            u = (pcov_expand > 0.0).sum()
            if covered < u:   # always pick the one with largest utility.
                covered = u
                picked = candidate
                pcov_update = pcov_expand.copy()

        if picked:
            # print('Picked!')
            expansion.append(picked)
            pcov = pcov_update.copy()
            picked_id = candidates.index(picked)
            del candidates[picked_id]  # remove the picked item from candidates.
            del matrix[picked_id]  # remove the picked features from matrix
        else:  # greedy select algorithm stops here, because the utility does not improve anymore.
            break
    return expansion





def show_pairs(style: str, length: int, topk: int, max_pair: int, seed: int):
    """
    Display the pairs that we want to explain
    """


    pass

def terms_tfidf(indexer: IndexReader, doc_id: str) -> List[Tuple[str, float]]:
    """ Given a document id, return the terms sorted by tf-idf scores.
    Args:
        doc_id: str, the id of doc in corpus.

    Return:
        terms with tf-idf score, sorted, a list of tuple. 
    """
    num_docs = indexer.stats()['documents']  # the number of documents
    print(f'docid: {doc_id}')
    tf = indexer.get_document_vector(doc_id)

    df = {term: (indexer.get_term_counts(term, analyzer=None))[0] for term in tf.keys()}
    tf_idf = {term: tf[term] * math.log(num_docs/(df[term] + 1)) for term in tf.keys() } 
    terms_sorted = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    return terms_sorted


# perturb a document
def doc_perturb(doc, term):
    doc_new = doc.replace(f' {term} ', '')
    return doc_new

