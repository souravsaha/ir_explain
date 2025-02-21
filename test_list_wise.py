import json

import numpy as np
import torch
from ir_explain.explainers import (BFSListwiseExplainer,
                                   GreedyListwiseExplainer,
                                   IntentListwiseExplainer,
                                   MultiplexListwiseExplainer)
from ir_explain.utils.utility import load_from_res
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder

# one can fetch query and query id from ir_datasets (https://ir-datasets.com/) 
query_str = "what is the daily life of thai people"
query_id = '1112341'


# provide res file path of the blackbox ranker
res_file_path = "examples/runs/NRMs/ANCE.2019.res"

# MSMARCO index path: 
# one can download a pre-built index from pyserini
index_path = "/disk_a/junk/msmarco-v1-passage-full"

# load dense ranking result and scores
dense_ranking, dense_scores = load_from_res(res_file_path)

dense_ranking_list = dense_ranking['1112341']
dense_score_list = dense_scores['1112341']

# dense_ranking_list = dense_ranking['88495']
# dense_score_list = dense_scores['88495']

# # initialize the parameters of BFS
params = {
    "QUEUE_MAX_DEPTH" : 1000,
    "BFS_MAX_EXPLORATION" : 30,
    "BFS_VOCAB_TERMS" : 30,
    "BFS_MAX_DEPTH" : 10,
    "BFS_TOP_DOCS" : 10,
    "CORRELATION_MEASURE" : "RBO",
    }
exp_model = "bm25"
indexer_type = "pyserini"
# # initialize the BFS class
bfs = BFSListwiseExplainer(index_path, indexer_type, exp_model, params)

# # initialize LuceneSearcher, we use LuceneSearcher from pyserini
searcher = LuceneSearcher(index_path)
searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))

# retrieve with BM25 
bm25_hits = searcher.search(query_str)

# set parameters for RM3
searcher.set_rm3(1000, 10, 0.9)

# generate the feedback terms 
term_weight_list = searcher.get_feedback_terms(query_str)

# sort the feedback terms
term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True))

# # call BFS explainer module
print(bfs.explain(query_id, query_str, term_weight_list, searcher, dense_ranking, debug = False))


# # initialize the parameters of Greedy
params = {
    "GREEDY_VOCAB_TERMS" : 100,
    "GREEDY_TOP_DOCS_NUM" : 10,
    "GREEDY_MAX_DEPTH" : 10,
    "BFS_TOP_DOCS" : 10,
    "CORRELATION_MEASURE" : "RBO"
    }

exp_model = "bm25"

# # initialize the Greedy class
greedy = GreedyListwiseExplainer(index_path, indexer_type, exp_model, params)

# we use LuceneSearcher from pyserini
searcher = LuceneSearcher(index_path)
searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))

# retrieve with BM25 
bm25_hits = searcher.search(query_str)

# set parameters for RM3
searcher.set_rm3(1000, 10, 0.9)   # set parameter for rm3

# generate the feedback terms
term_weight_list = searcher.get_feedback_terms(query_str)

# sort the feedback terms
term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True))

# call Greedy explainer module
print(greedy.explain(query_id, query_str, term_weight_list, searcher, dense_ranking, debug = False))

searcher = LuceneSearcher(index_path)
# for the top k documents fetch their contents
docs = dict([(hit, json.loads(searcher.doc(hit).raw())['contents']) for hit in dense_ranking_list[:20]])

# Load a reranking model
model_name = "cross-encoder/ms-marco-electra-base"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = load_your_model()
reranker = CrossEncoder(model_name, max_length = 512, device = device)

corpus = {'query': query_str,
        'scores': dict([(doc_id, score) for doc_id, score in zip(dense_ranking_list[:20], dense_score_list[:20])]),
        'docs': docs
    }

# set parameters fro IntentEXS
params = {'top_idf': 200, 'topk': 20, 'max_pair': 100, 'max_intent': 20, 'style': 'random'}

# Init the IntentEXS object.
Intent = IntentListwiseExplainer(reranker, index_path, indexer_type, 'bm25')

# # call explain method of IntentEXS
expansion = Intent.explain(corpus, params)

print(expansion)

# set parameters for Multiplex
params = {
    "dataset" : "clueweb09",
    "top_d" : 10,
    "top_tfidf" : 100,
    "top_r" : 50,
    "candi_method" : "bm25",
    "ranked" : 5,
    "pair_num" : 20,
    "max_k" : 10,
    "min_k" : 4,
    "candidates_num" : 20,
    "style" : "random",
    "vote" : 1,
    "tolerance" : .005,
    "EXP_model" : "language_model",
    "optimize_method" : "geno",
    "mode" : "candidates"
    }
# pass the topk file
params ["top_file"] = "/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/src/clueweb09/top.tsv"
# pass the query file
params ["queries_file"] = "/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/src/clueweb09/queries.tsv"

# one can fetch query and query id from ir_datasets (https://ir-datasets.com/)
query_str = 'lps laws definition'
qid = '443396'

# provide res file path of the blackbox ranker
res_file_path = "examples/runs/NRMs/ANCE.2019.res"

# MSMARCO index path:
# one can download it from pyserini
index_path = "/disk_a/junk/msmarco-v1-passage-full"

# load dense ranking result and scores
dense_ranking, dense_scores = load_from_res(res_file_path)

dense_ranking_list = dense_ranking['443396']
dense_score_list = dense_scores['443396']

params["dense_ranking"] = dense_ranking_list
params["dense_ranking_score"] = dense_score_list

# initialize the Multiplex class
multi = MultiplexListwiseExplainer(index_path, indexer_type)
params["EXP_model"] = "multi"
params["optimize_method"] = "geno_multi"

# call Multiplex explainer module
multi.explain(qid, query_str, params)

# DEBUG/EXPLAIN Multiplex with additional details
# e.g., 1) generate_candidates, 2) show_matrix
multi.generate_candidates(qid, query_str, params)

#multi.generate_doc_pairs(qid, query_str, params)
# show matrix for any simple explainer. For that, you need to change the parameters to one of the following:
# 1. language_model 
# 2. saliency
# 3. semantic
params["EXP_model"] = "language_model"
multi.show_matrix(qid, query_str, params)
