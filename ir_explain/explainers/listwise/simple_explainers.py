import nltk
import math
import numpy as np
from collections import Counter
from typing import Any, Tuple, List, Dict, Callable, Union
from ir_explain.utils import utility
#from Datasets.dataIterDrmm import matching_histogram_mapping
from ir_explain.datasets.dataIterDrmm import matching_histogram_mapping
import gensim.downloader as api

import sys
sys.path.append('Datasets')
EXP_model = ['language_model', 'saliency', 'semantic']  # saliency is position-weighted term frequency.
pretrained_embeddings = 'glove-wiki-gigaword-100'
Word_Vectors = api.load(pretrained_embeddings)

def get_explainer(name: str) -> Callable:
    if name == EXP_model[0]:
        return _lm
    elif name == EXP_model[1]:
        return _saliency
    elif name == EXP_model[2]:
        return _semantic
    else:
        raise ValueError(f'Explainer types must be one of {EXP_model}')


def _lm(token: Union[str, List[str]], doc: str, analyzer, max_tokens: int = 510) -> Union[float, List[float]]:
    """ Language model explainer, unnecessary arguments to keep consistent with other explainers """
    doc = doc.lower()
    if analyzer:
        tokens = analyzer.analyze(doc)
    else:
        tokens = nltk.word_tokenize(doc)   # nltk tokenization
    if max_tokens:  # only counts the max_tokens? bert tokenization truncates.
        tokens = tokens[:max_tokens]
    lm = Counter(tokens)
    def single(token, lm):
        if token in lm:
            # prob = math.log(1 + lm.get(token)/len(tokens))   # > 0
            prob = lm.get(token)/len(tokens)  # >0
        else:
            prob = 0   # without smoothing, just return None. changed to 0
        return prob
    if isinstance(token, str):
        token = [token]
    scores = [single(tok, lm) for tok in token]
    if len(scores) == 1:
        return scores[0]
    else:
        return scores


def _saliency(token: Union[str, List[str]], doc: str, analyzer, max_tokens: int = 510) -> Union[float, List[float]]:
    doc = doc.lower()
    def analyze_sent(doc: str):
        sents = nltk.sent_tokenize(doc)
        sents_analyzed = [Counter(analyzer.analyze(s)) for s in sents]
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



def _semantic(token: Union[str, List[str]], doc: str, analyzer, max_tokens: int=510) -> Union[float, List[float]]:
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

    emb_doc = [word2vec(Word_Vectors, t) for t in analyzer.analyze(doc) if word2vec(Word_Vectors, t).any()]
    if isinstance(token, str):
       token = [token]
    emb_token = [word2vec(Word_Vectors, t) for t in token]
    similarities = [similarity(emb_t, emb_doc) for emb_t in emb_token]
    if len(similarities) == 1:   # only a single word/value
        return similarities[0]
    else:
        return similarities


def rerank(explainer: Callable, query_expand: List[str], docs: List[str], analyzer, max_token: int=510) -> List[float]:
    """ Apply the chosen explainer to rank documents based on expanded query terms.
        The ranking score is simply summed up by all query terms.
    """
    scores = []
    for doc in docs:
        score = 0
        for query in query_expand:
            s = explainer(query, doc, analyzer, max_token)
            if s:
                score += s
        scores.append(score)
    return scores

def multi_rank(Exp_model:List[str], query_expand: List[str], docs: List[str], analyzer, max_token: int=510) -> Union[np.array, List[np.array]]:
    """ Apply the chosen explainer to rank documents based on expanded query terms.
        The ranking score is simply summed up by all query terms.
    """
    SCORE = []
    for exp in Exp_model:
        explainer = get_explainer(exp)
        score = rerank(explainer, query_expand, docs, analyzer, max_token)
        SCORE.append(np.array(score))
    if len(SCORE) <= 1:
        return SCORE[0]
    else:
        return SCORE

