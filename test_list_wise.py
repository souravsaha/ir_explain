from explainers import BFSListwiseExplainer
from explainers import GreedyListwiseExplainer
from explainers import IntentListwiseExplainer
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import Analyzer, get_lucene_analyzer
from utils.utility import load_from_res
import numpy as np
import json, torch
from sentence_transformers import CrossEncoder      

# one can fetch query and query id from ir_datasets (https://ir-datasets.com/) 
query_str = "what is the daily life of thai people"
query_id = '1112341'


# provide res file path of the blackbox ranker
res_file_path = "/disk_a/junk/explain/ir_explain/runs/NRMs/ANCE.2019.res"

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
# params = {
#     "QUEUE_MAX_DEPTH" : 1000,
#     "BFS_MAX_EXPLORATION" : 30,
#     "BFS_VOCAB_TERMS" : 30,
#     "BFS_MAX_DEPTH" : 10,
#     "BFS_TOP_DOCS" : 10,
#     "CORRELATION_MEASURE" : "RBO",
#     }
# exp_model = "bm25"

# # initialize the BFS class
# bfs = BFSListwiseExplainer(index_path, exp_model, params)

# # initialize LuceneSearcher, we use LuceneSearcher from pyserini
# searcher = LuceneSearcher(index_path)
# searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
# searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))

# # retrieve with BM25 
# bm25_hits = searcher.search(query_str)

# # set parameters for RM3
# searcher.set_rm3(1000, 10, 0.9)

# # generate the feedback terms 
# term_weight_list = searcher.get_feedback_terms(query_str)

# # sort the feedback terms
# term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True))

# # call BFS explainer module
# print(bfs.explain(query_id, query_str, term_weight_list, searcher, dense_ranking, debug = False))

# # initialize the parameters of Greedy
# params = {
#     "GREEDY_VOCAB_TERMS" : 100,
#     "GREEDY_TOP_DOCS_NUM" : 10,
#     "GREEDY_MAX_DEPTH" : 10,
#     "BFS_TOP_DOCS" : 10,
#     "CORRELATION_MEASURE" : "RBO"
#     }

# exp_model = "bm25"

# # initialize the Greedy class
# greedy = GreedyListwiseExplainer(index_path, exp_model, params)

# # we use LuceneSearcher from pyserini
# searcher = LuceneSearcher(index_path)
# searcher.set_bm25(1.2, 0.75)     # set BM25 parameter
# searcher.set_analyzer(get_lucene_analyzer(stemmer='porter'))

# # retrieve with BM25 
# bm25_hits = searcher.search(query_str)

# # set parameters for RM3
# searcher.set_rm3(1000, 10, 0.9)   # set parameter for rm3

# # generate the feedback terms
# term_weight_list = searcher.get_feedback_terms(query_str)

# # sort the feedback terms
# term_weight_list = dict(sorted(term_weight_list.items(), key=lambda item: item[1], reverse = True))

# # call Greedy explainer module
# print(greedy.explain(query_id, query_str, term_weight_list, searcher, dense_ranking, debug = False))

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
Intent = IntentListwiseExplainer(reranker, index_path, 'bm25')

# call explain method of IntentEXS
expansion = Intent.explain(corpus, params)

print(expansion)