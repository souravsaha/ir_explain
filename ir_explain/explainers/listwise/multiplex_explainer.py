import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ir_explain.datasets.dataIterBert
import ir_explain.datasets.dataIterDpr
import ir_explain.datasets.dataIterDrmm
import ir_explain.datasets.trec
import ir_explain.datasets.trecdl
import ir_explain.explainers.listwise.simple_explainers
import numpy as np
from ir_explain.explainers.listwise.base_listwise import BaseListwiseExplainer
from ir_explain.explainers.listwise.multiplex_base import (Explain, analyzer,
                                                           device)
from ir_explain.explainers.listwise.simple_explainers import EXP_model
from ir_explain.models.bert_model import BertRanker
from ir_explain.models.dpr_model import DprRanker
from ir_explain.models.drmm_model import DRMMRanker
from ir_explain.utils.optimization import (dense_optimize, geno_solver,
                                           geno_solver_multi, gradients_expand,
                                           kendalltau_concord,
                                           preference_coverage)
from tqdm import tqdm
# from pytorch_lightning import seed_everything
#from Datasets import dataIterBert, dataIterDrmm, dataIterDpr, trec, trecdl
from transformers import (AutoModel, BertConfig, BertModel, DistilBertModel,
                          DistilBertTokenizer)

#bert_type = 'bert-base-uncased' 
bert_type = 'bert-base-uncased' 
dpr_question = 'facebook/dpr-question_encoder-multiset-base'
dpr_context = 'facebook/dpr-ctx_encoder-multiset-base'
#project_dir = Path('/home/lyu/ExpRank')
project_dir = Path('/a/administrator/codebase/neural-ir/RankingExplanation_bkp')
#bert_cache = Path('/home/lyu/pretrained/bert/')
#dpr_cache = Path('/home/lyu/pretrained/dpr/')
#glove_cache = Path('/home/lyu/ExpRank/Datasets/pkl/glove_vocab.pkl')
bert_cache = Path('/a/administrator/codebase/neural-ir/RankingExplanation_bkp/pretrained/bert/')
dpr_cache = Path('/a/administrator/codebase/neural-ir/RankingExplanation_bkp/pretrained/dpr/')
glove_cache = Path('/a/administrator/codebase/neural-ir/RankingExplanation_bkp/Datasets/Datasets2/Datasets/pkl/glove_vocab.pkl')

datasets = ['clueweb09', 'robust04', 'msmarco_p']

#transformers.logging.set_verbosity_error()
import logging

logging.disable(logging.WARNING)

#def init_explainer(queries_file: Path, top_file: Path, dataset: str='clueweb09', reranker_type: str='bert', model_fold: str='fold_1'):

def init_explainer(query: str, dense_ranking: list, index_path):
    #print('dataset : ', dataset)
    kwargs = {'RankModel': dense_ranking, 'query': query, 'index_dir': index_path}
    Explainer = Explain(kwargs)
    
    return Explainer

    """
    if dataset in  datasets:
        #data_dir = project_dir / f"Datasets/src/{dataset}"
        data_dir = project_dir / f"Datasets/Datasets2/Datasets/src/{dataset}"
        documents_file = data_dir / 'documents.tsv'
        index_dir = str(data_dir / 'indexes' /f"{dataset}_indexes")
        print(index_dir)

        if dataset == 'msmarco_p':
            queries = trecdl.get_queries(queries_file)
            print('queries', len(queries))
        else:
            queries = trec.get_queries(queries_file)
        
        kwargs = {'data_file': None, 'train_file': None, 'val_file': None, 'test_file': None,
                'training_mode': None, 'rr_k': None, 'num_workers': None, 'bert_type': bert_type, 'bert_cache': bert_cache, 'freeze_bert': False}
        print('reranker_type ', reranker_type) 
        if reranker_type == 'bert': 
            #model_dir = project_dir / f'trained/{dataset}/pairwise/{model_fold}/{reranker_type}/lightning_logs/version_0/checkpoints/'
            model_dir = project_dir / f'pretrained/{dataset}/{reranker_type}/{model_fold}/lightning_logs/version_0/checkpoints/'
            print(model_dir)
            checkpoint = str(list(model_dir.glob('*.ckpt'))[0])
            
            model = BertRanker.load_from_checkpoint(checkpoint, **kwargs).to(device).eval()
            #model = DistilBertModel.from_pretrained(bert_type).to(device).eval()
            #model = AutoModel.from_pretrained(bert_type).to(device).eval()
            InferenceDataset = dataIterBert.InferenceDataset(documents_file, top_file, bert_type, bert_cache, DATA=dataset, device=device)
            dataIterate = dataIterBert
        elif reranker_type == 'drmm':
            model_dir = project_dir / f'trained/{dataset}/pairwise/{model_fold}/{reranker_type}/lightning_logs/version_0/checkpoints/'
            checkpoint = str(list(model_dir.glob('*.ckpt'))[0])
            kwargs['vocab_file'] = glove_cache
            model = DRMMRanker.load_from_checkpoint(checkpoint, **kwargs).to(device).eval() 
            InferenceDataset = dataIterDrmm.InferenceDataset(documents_file, top_file, model.vocab, DATA=dataset, device=device)
            dataIterate = dataIterDrmm
        elif reranker_type == 'dpr':
            dpr_args = {'question_model': dpr_question, 'context_model': dpr_context,'dpr_cache': dpr_cache,'loss_margin':0.2, 'batch_size':32 }
            kwargs.update(dpr_args)
            #model = DprRanker(kwargs).to(device).eval()
            model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device).eval()
            InferenceDataset = dataIterDpr.InferenceDataset(documents_file, top_file, dpr_question, dpr_context, dpr_cache, DATA=dataset, device=device)
            dataIterate = dataIterDpr
        
        kwargs = {'dataset': dataset, 'RankModel': model, 'InferenceDataset': InferenceDataset, 'dataIterate': dataIterate, 'queries': queries, 'index_dir': index_dir}
        Explainer = Explain(kwargs)
        print(Explainer)
        return Explainer
    else:
        raise ValueError('dataset: clueweb09')
    """

def load_pickle(file: Path):
    if os.path.isfile(file):
        with open(file, 'rb')as f:
            out = pickle.load(f)
        return out
    else:
        return None

def load_json(file: Path):
    if os.path.isfile(file):
        with open(file, 'r')as f:
            out = json.load(f)
        return out
    else:
        return None

def filter_query(hparams: Dict[str, Any]):
    """ Filter out queries with fewer than 90 instances"""
    qids_useful = []
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    for q_id in tqdm(hparams['q_ids'], desc='Generating candidates for each query'):
        Explainer._init_query(q_id, False)
        if Explainer.InferenceDataset.length >= 90:
            qids_useful.append(q_id)
    
    print(f'Filtered {len(qids_useful)} queries from 1000 candidates.')
    with open(hparams['queries_file'].parent / f"useful_query_{hparams['query_type']}.txt", 'w')as f:
        for q in qids_useful:
            f.write(q+'\n')


def gen_candidates(hparams: Dict[str, Any]):
    """ Pre-generate candidates for more folds. """
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    for q_id in tqdm(hparams['q_ids'], desc='Generating candidates for each query'):
        candidates = Explainer.get_candidates_reranker(q_id, hparams['top_d'], hparams['top_tfidf'], hparams['top_r'], hparams['candi_method'])
        save_dir = hparams['exp_fold'] / f"matrix_{hparams['Rerank_model']}" / f"{q_id}_candidates.json"
        print('candidates', len(candidates))
        with open(save_dir, 'w')as f:
            json.dump(candidates, f)


def gen_matrix(hparams: Dict[str, Any]):
    """ Pre-generate candidates for more folds. """
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    for q_id in tqdm(hparams['q_ids'], desc='Generating matrix for each query'):
        candidates_file = hparams['exp_fold'] / f"matrix_{hparams['Rerank_model']}" / f'{q_id}_candidates.json'
        candidates_tokens = load_json(candidates_file)
        print(candidates_tokens)
        print('candidates_file', candidates_file)
        if isinstance(candidates_tokens, dict):
            candidates_tokens = list(candidates_tokens.keys())
        Explainer._init_query(q_id, True)
        doc_pairs = Explainer.sample_doc_pair(hparams['ranked'], hparams['pair_num'], hparams['style'], hparams['tolerance'])
        if doc_pairs:
            matrix = Explainer.build_matrix(candidates_tokens, doc_pairs, hparams['EXP_model'])
            print(matrix)
            save_dir = hparams['exp_fold'] / f"matrix_{hparams['Rerank_model']}" / f"{q_id}_matrix_{hparams['EXP_model']}.pkl"
            with open(save_dir, 'wb')as f:
                pickle.dump(matrix, f)
        else:
            print('Empty doc pairs. Maybe reduce the tolerance value.')


def remove_all_zeros(candidates: List[str], matrix: List[List[float]]):
    """ Remove all zeros entries/candidates, for future experiments."""
    matrix = np.array(matrix)
    signs = np.where(np.array([np.all(matrix[i, :]) for i in range(matrix.shape[0])]) != 0)[0]
    new_candidates = np.array(candidates)[signs]
    new_matrix = matrix[signs]
    return new_candidates.tolist(), new_matrix.tolist()


def explain_single(q_id: str, explainer, hparams: Dict[str, Any], exp_model: List[str], candidates_tokens, doc_pairs):
    """ generate expansion terms with a single matrix/explainer/method. """
    #candidates_file = hparams['exp_fold'] / f"matrix_{hparams['Rerank_model']}" / f'{q_id}_candidates.json'
    #print(candidates_file)
    #exit(1)
    #candidates_tokens = load_json(candidates_file)
    if hparams['EXP_model'] == 'multi':
        matrix_1 = explainer.build_matrix(hparams['dense_ranking'], candidates_tokens, doc_pairs, "language_model")
        matrix_2 = explainer.build_matrix(hparams['dense_ranking'], candidates_tokens, doc_pairs, "saliency")
        matrix_3 = explainer.build_matrix(hparams['dense_ranking'], candidates_tokens, doc_pairs, "semantic")

        matrixs = [matrix_1, matrix_2, matrix_3]
        print(f'stacked three matrices...')
    else:
        matrix = explainer.build_matrix(hparams['dense_ranking'], candidates_tokens, doc_pairs, hparams['EXP_model'])
        print(f'after matrix building: {len(matrix)}')
    if hparams['EXP_model'] == 'saliency' and hparams.get('norm_saliency', False):
        matrix = np.array(matrix)
        for i in range(matrix.shape[1]):
            if (matrix[:, i] != 0).any():
                matrix[:, i] = matrix[:, i] / np.abs(matrix[:, i]).sum()
        matrix = matrix.tolist()
    if isinstance(candidates_tokens, dict):
        print('isinstance?')
        candidates_tokens = list(candidates_tokens.keys())
        print('here are the candidates_tokens', candidates_tokens)

    # extract expansion terms with different methods.
    if hparams['optimize_method'] == 'greedy':
        expansion, utility = preference_coverage.greedy(candidates_tokens, matrix, hparams['max_k'], hparams['min_k'])
    elif hparams['optimize_method'] == 'linear_program':
        expansion, utility = preference_coverage.linear_program(candidates_tokens, matrix, hparams['max_k'], hparams['min_k'])
    elif hparams['optimize_method'] == 'geno':
        #print("matrix ", matrix)
        print("max k ", hparams['max_k'])
        print("min k ", hparams['min_k'])
        expansion, utility = geno_solver.run(candidates_tokens, matrix, hparams['max_k'], hparams['min_k'])
        print(expansion)
        print(utility)
    elif hparams['optimize_method'] == 'geno_multi':
        print(f'before calling geno solver, canidates_tokens : {candidates_tokens}')
        expansion, utility = geno_solver_multi.run(candidates_tokens, matrixs, hparams['max_k'], hparams['min_k'])
        print(expansion)
        print(utility)
        #exit(1)
    elif hparams['optimize_method'] == 'greedy_multi':
        expansion, utility = preference_coverage.greedy_multi(candidates_tokens, matrixs, hparams['max_k'], hparams['min_k'])
    elif hparams['optimize_method'] == 'feature_select':
        expansion, utility = preference_coverage.feature_select(candidates_tokens, matrix, hparams['max_k'], hparams['min_k'])
    elif hparams['optimize_method'] == 'dense_learn':
        pass
        #expansion = dense_optimize.Optimize(candidates_tokens, exp_model, candidates_file.parent, q_id)
        #utility = 0   # can't compute it directly from this method
    elif hparams['optimize_method'] == 'dense_greedy':
        pass
        #weights_rescale = dense_optimize.rescale_matrix(exp_model, candidates_file.parent, q_id)
        #expansion, utility = preference_coverage.greedy(candidates_tokens, weights_rescale, hparams['max_k'], hparams['min_k'])
    else:
        raise ValueError(f"Optimize method: {hparams['optimize_method']}")
    print('Successfully executed explain single function')
    print(f'exapnsion terms : {expansion}')
    print(f'utility score : {utility}')
    
    return expansion, utility


def explain_by_query(q_tokens: List[str]):
    """ No expansion. Query terms are used as explanations."""
    return q_tokens.copy(), 0


def explain_by_gradients(rerank_model:str, model, tokenizer, query: str, docs: List[str], prediction: List[float], max_k):
    """ Explain by gradients. extract top k words with highest average gradients. """
    word_importance_all = {}
    rank = np.log(np.argsort(np.argsort(prediction)) + 2)  # rank starts from 2, to avoid log0. 
    if rerank_model == 'bert':
        gradient_method = gradients_expand.attributes_bert
    elif rerank_model == 'dpr':
        gradient_method = gradients_expand.attributes_dpr

    for doc, r  in tqdm(zip(docs, rank), desc='gradients for each doc...'):
        word_importance = gradient_method(model, tokenizer, query, doc, device=device)
        for w, s in word_importance[:max_k]:
            #print(w, s, '\n', '='*20, word_importance_all)
            if w not in word_importance_all:
                word_importance_all[w] = [s*r]
            else:
                old_scores = word_importance_all[w]
                old_scores.append(s*r)
                word_importance_all[w] = old_scores
    # average across all docs
    for w, v in word_importance_all.items():
        word_importance_all[w] = sum(v) / len(v)
    sorted_importance = sorted(word_importance_all.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in sorted_importance[:max_k]], 0


def evaluate(hparams: Dict[str, Any]):
    """ Evaluate kendalltau scores of expansion terms. """
    if hparams['EXP_model'] == 'multi':
        exp_model = EXP_model.copy()
    else:
        exp_model = [hparams['EXP_model']]
    #Explainer = init_explainer(hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    CORREL, Expansion = [], []
    
    for q_id in tqdm(hparams['q_ids'], desc='Processing query'):
        try:
            Explainer._init_query(q_id, True)
            print("after initing query....")
            if hparams['optimize_method'] == 'gradient':
                expansion, utility = explain_by_gradients(hparams['Rerank_model'], Explainer.model, Explainer.InferenceDataset.tokenizer,
                            Explainer.InferenceDataset.query, Explainer.InferenceDataset.top_docs, Explainer.InferenceDataset.prediction, hparams['max_k'])
            elif hparams['optimize_method'] == 'query_tokens':
                expansion, utility = explain_by_query(Explainer.InferenceDataset.query_tokens)
            else:
                print('coming to else part')
                expansion, utility = explain_single(q_id, hparams, exp_model)
                expansion = list(set(expansion + Explainer.InferenceDataset.query_tokens.copy() ))   # query + expansion
                print(expansion)

            corr_0, corr_1, corr_2, corr_3 = Explainer.evaluate_fidelity(expansion, exp_model, vote =hparams['vote'], tolerance=hparams['tolerance'])
            CORREL.append((corr_0, corr_1, corr_2, corr_3, utility))
            #print(f'kendall tau: {corr_0}, {corr_1}, {corr_2}, {corr_3}, {utility}')
            Expansion.append(expansion)
        except Exception as e:
            print(f'Error happend in query {q_id}, ignore it for now.... :(')
            print(f'Error: {e}')
            continue

    save_dir = hparams['exp_fold'] / 'kendalls' /f"{hparams['Rerank_model']}" / f"{hparams['optimize_method']}_{hparams['EXP_model']}_kendall.json"
    expan_dir = hparams['exp_fold'] / 'kendalls' /f"{hparams['Rerank_model']}" / f"{hparams['optimize_method']}_{hparams['EXP_model']}_terms.json"
    print('save_dir:  ', save_dir)
    print('expan_dir: ', expan_dir)
    # compute average
    AVG_strict, AVG_relax = [], []
    for i in range(len(CORREL[0])):
        avg_s = sum([C[i] for C in CORREL])/len(CORREL)
        relax = [C for C in CORREL if C[i]]
        if relax:
            avg_r = sum([C[i] for C in relax])/len(relax)
        else:
            avg_r = 0
        AVG_strict.append(avg_s)
        AVG_relax.append(avg_r)
    CORREL.append(AVG_strict)
    CORREL.append(AVG_relax)

    with open(save_dir, 'w')as f:
        print('dumping kendall...')
        json.dump(CORREL, f)
    
    with open(expan_dir, 'w')as f:
        print('dumping terms...')
        json.dump(Expansion, f)


def ablation(hparams: Dict[str, Any]):
    """Ablation experiments with different numbers of sampling pairs"""  
    exp_model = EXP_model.copy() 
    CORREL, Expansion = [], []
    doc_num = hparams['pair_num']
    #Explainer = init_explainer(hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
   
    """ Pre-generate candidates for more folds. """  
    for q_id in tqdm(hparams['q_ids'], desc='Generating matrix for each query'):
        candidates_file = hparams['exp_fold'] / f"matrix_{hparams['Rerank_model']}" / f'{q_id}_candidates.json'
        candidates_tokens = load_json(candidates_file)
        if isinstance(candidates_tokens, dict):
            candidates_tokens = list(candidates_tokens.keys())
        Explainer._init_query(q_id, True)
        doc_pairs = Explainer.sample_doc_pair(hparams['ranked'], doc_num, hparams['style'], hparams['tolerance'])    
        for M in exp_model:
            matrix = Explainer.build_matrix(candidates_tokens, doc_pairs, M)
            save_dir = hparams['exp_fold'] / f"matrix_{hparams['Rerank_model']}" / 'ablation' / f"{q_id}_matrix_{M}_{doc_num}.pkl"
            with open(save_dir, 'wb')as f:
                pickle.dump(matrix, f)
    

    """Explain"""
    for q_id in tqdm(hparams['q_ids'], desc='Processing query'):
        candidates_file = hparams['exp_fold'] / f"matrix_{hparams['Rerank_model']}" / f'{q_id}_candidates.json'
        candidates_tokens = load_json(candidates_file)
        if isinstance(candidates_tokens, dict):
            candidates_tokens = list(candidates_tokens.keys())
        Explainer._init_query(q_id, True)
        matrix_file_1 = candidates_file.parent / 'ablation' / f"{q_id}_matrix_{exp_model[0]}_{doc_num}.pkl"
        matrix_file_2 = candidates_file.parent / 'ablation' / f"{q_id}_matrix_{exp_model[1]}_{doc_num}.pkl"
        matrix_file_3 = candidates_file.parent / 'ablation' / f"{q_id}_matrix_{exp_model[2]}_{doc_num}.pkl"
        matrixs = [load_pickle(matrix_file_1), load_pickle(matrix_file_2), load_pickle(matrix_file_3)]
        expansion, utility = geno_solver_multi.run(candidates_tokens, matrixs, hparams['max_k'], hparams['min_k'])
        corr_0, corr_1, corr_2, corr_3 = Explainer.evaluate_fidelity(expansion, exp_model, vote=hparams['vote'], tolerance=hparams['tolerance'])
        #print(corr_0, corr_1, corr_2, corr_3)
        CORREL.append((corr_0, corr_1, corr_2, corr_3, utility))
        Expansion.append(expansion)
        
    save_dir = hparams['exp_fold'] / 'kendalls' / f"{hparams['Rerank_model']}" / 'ablation' / f"{doc_num}_kendall.json"
    expan_dir = hparams['exp_fold'] / 'kendalls' / f"{hparams['Rerank_model']}" / 'ablation' / f"{doc_num}_terms.json"
    # compute average
    AVG_strict, AVG_relax = [], []
    for i in range(len(CORREL[0])):
        avg_s = sum([C[i] for C in CORREL])/len(CORREL)
        relax = [C for C in CORREL if C[i]]
        if relax:
            avg_r = sum([C[i] for C in relax])/len(relax)
        else:
            avg_r = 0
        AVG_strict.append(avg_s)
        AVG_relax.append(avg_r)
    CORREL.append(AVG_strict)
    CORREL.append(AVG_relax)

    with open(save_dir, 'w')as f:
        json.dump(CORREL, f)
    
    with open(expan_dir, 'w')as f:
        json.dump(Expansion, f)


def single_doc_score(exp_model: List[str], expand_query: List[str], doc: str, analyzer, max_token: int=510):
    # scores of all explainers for a doc with expansion query.
    scores = []
    for e in exp_model:
        # Exp = explainers.get_explainer(e)
        Exp = simple_explainers.get_explainer(e)
        score = 0
        for q in expand_query:
            s = Exp(q, doc, analyzer, max_token)
            if s:
                score += s
        scores.append(score)
    return scores


def compute_coverage(hparams: Dict[str, Any]):
    """ compute coverage of expansion terms. """
    if hparams['EXP_model'] == 'multi':
        exp_model = EXP_model.copy()
    else:
        exp_model = [hparams['EXP_model']]
    #Explainer = init_explainer(hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Coverage = []
    expansion_dir = hparams['exp_fold'] / 'kendalls' /f"{hparams['Rerank_model']}" / f"{hparams['optimize_method']}_{hparams['EXP_model']}_terms.json"
    print('expansion_dir :', expansion_dir)
    terms = load_json(expansion_dir)
    #if hparams['optimized_method'] == 'query_tokens':

    for q_id, exp_term in tqdm(zip(hparams['q_ids'], terms), desc='Processing query for coverage calculation'):  
        Explainer._init_query(q_id, True)
        doc_pairs = Explainer.sample_doc_pair(hparams['ranked'], hparams['pair_num'], hparams['style'], hparams['tolerance'])
        coverage = np.zeros(3)
        for rank_h_id, rank_l_id in doc_pairs:
            doc_h_id = Explainer.InferenceDataset.rank[rank_h_id]
            doc_l_id = Explainer.InferenceDataset.rank[rank_l_id]
            doc_h = Explainer.InferenceDataset.top_docs[doc_h_id]
            doc_l = Explainer.InferenceDataset.top_docs[doc_l_id]
            doc_h_title = Explainer.InferenceDataset.top_docs_id[doc_h_id]
            doc_l_title = Explainer.InferenceDataset.top_docs_id[doc_l_id]
            scores_h = single_doc_score(exp_model, exp_term, doc_h_title, doc_h, analyzer, Explainer.index_reader)
            scores_l = single_doc_score(exp_model, exp_term, doc_l_title, doc_l, analyzer, Explainer.index_reader)
            cover = ((np.array(scores_h) - np.array(scores_l)) > 0).astype('uint8')
            coverage += cover
        coverage = max(coverage) / len(doc_pairs)
        print(f'Coverage: {coverage}')
        Coverage.append(coverage)
    coverage_dir = hparams['exp_fold'] / 'kendalls' /f"{hparams['Rerank_model']}" / f"{hparams['optimize_method']}_{hparams['EXP_model']}_coverage.json"
    with open(coverage_dir, 'w')as f:
        json.dump(Coverage, f)
    #return Coverage
    

def correct_local_tau(hparams: Dict[str, Any]):
    """ correct gapped local tau with expansion terms. """
    if hparams['EXP_model'] == 'multi':
        exp_model = EXP_model.copy()
    else:
        exp_model = [hparams['EXP_model']]
    #Explainer = init_explainer(hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    expansion_dir = hparams['exp_fold'] / 'kendalls' /f"{hparams['Rerank_model']}" / f"{hparams['optimize_method']}_{hparams['EXP_model']}_terms.json"
    terms = load_json(expansion_dir)
    Tau_gl = []
    for q_id, expansion in tqdm(zip(hparams['q_ids'], terms), desc='Processing query for tau correction...'):
        Explainer._init_query(q_id, True)
        corr_3 = Explainer.evaluate_fidelity(expansion, exp_model, vote =hparams['vote'], tolerance=hparams['tolerance'])
        Tau_gl.append(corr_3)
    save_dir = hparams['exp_fold'] / 'kendalls' /f"{hparams['Rerank_model']}" / f"{hparams['optimize_method']}_{hparams['EXP_model']}_fixedlocaltau.json"
    # compute average
    avg_s = sum(Tau_gl)/len(Tau_gl)
    relax = [C for C in Tau_gl if C]
    
    if relax:
        avg_r = sum(relax)/len(relax)
    else:
        avg_r = 0
    
    Tau_gl.append(avg_s)
    Tau_gl.append(avg_r)

    with open(save_dir, 'w')as f:
        json.dump(Tau_gl, f)


def model_diff(hparams: Dict[str, Any]):  
    ranker = ['bert', 'dpr', 'drmm']
    Tau1, Tau2, Tau3 = [],[],[]
    for q_id in tqdm(hparams['q_ids'], desc='Processing query'):
        #Explainer_1 = init_explainer(hparams['dataset'], ranker[0], hparams['FOLD_NAME'])
        Explainer_1 = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], ranker[0], hparams['FOLD_NAME'])
        #Explainer_2 = init_explainer(hparams['dataset'], ranker[1], hparams['FOLD_NAME'])
        Explainer_2 = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], ranker[1], hparams['FOLD_NAME'])
        #Explainer_3 = init_explainer(hparams['dataset'], ranker[2], hparams['FOLD_NAME'])
        Explainer_3 = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], ranker[2], hparams['FOLD_NAME'])
        Explainer_1._init_query(q_id, True)
        Explainer_2._init_query(q_id, True)
        Explainer_3._init_query(q_id, True)
        tau1 = kendalltau_concord.kendalltau_gap(Explainer_1.InferenceDataset.prediction.copy(), Explainer_2.InferenceDataset.prediction.copy(), 0)
        tau2 = kendalltau_concord.kendalltau_gap(Explainer_1.InferenceDataset.prediction.copy(), Explainer_3.InferenceDataset.prediction.copy(), 0)
        tau3 = kendalltau_concord.kendalltau_gap(Explainer_2.InferenceDataset.prediction.copy(), Explainer_3.InferenceDataset.prediction.copy(), 0)
        Tau1.append(tau1)
        Tau2.append(tau2)
        Tau3.append(tau3)
    
    avg1 = sum(Tau1)/len(Tau1)
    avg2 = sum(Tau2)/len(Tau2)
    avg3 = sum(Tau3)/len(Tau3)
    print(f'Average tau: {avg1}, {avg2}, {avg3}')
    Tau1.append(avg1)
    Tau2.append(avg2)
    Tau3.append(avg3)
    with open(hparams['exp_fold'], 'w')as f:
        json.dump([Tau1, Tau2, Tau3], f)


def perturb_doc(doc: str, tokens: List[str]):
    return ' '.join(tokens) + ' ' + doc


def evaluate_exp_by_perturb(hparams: Dict[str, Any]):
    def rank_diff(EXPlainer, query, doc_id, tokens):
        doc = EXPlainer.InferenceDataset.top_docs[doc_id]
        adv_doc = perturb_doc(doc, tokens)
        adv_inp = EXPlainer.InferenceDataset.get_single_input(query, adv_doc)
        score = EXPlainer.model(adv_inp).data.item()
        new_pred = EXPlainer.InferenceDataset.prediction.copy()
        new_pred[doc_id] = score
        new_rank = np.argsort(-new_pred)
        diff = np.where(EXPlainer.InferenceDataset.rank == doc_id)[0].item() - np.where(new_rank==doc_id)[0].item()
        return diff

    #Explainer = init_explainer(hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    if hparams['optimize_method'] == 'query_tokens_repeat':
        Terms = []
    elif hparams['optimize_method'] == 'wikipedia':
        Terms = []
    else:
        terms_file = hparams['exp_fold'] / 'kendalls' /f"{hparams['Rerank_model']}" / f"{hparams['optimize_method']}_{hparams['EXP_model']}_terms.json"
        Terms = load_json(terms_file)
    Change = []
    for i, q_id in tqdm(enumerate(hparams['q_ids']), desc='Processing query'):  
        Explainer._init_query(q_id, True) 
        query = Explainer.InferenceDataset.query
        doc_id = Explainer.InferenceDataset.rank[-1]
        if Terms:
            terms = Terms[i]     # without query
            change = rank_diff(Explainer, query, doc_id, terms)      
        else:
            if hparams['optimize_method'] == 'query_tokens_repeat':
                terms = Explainer.InferenceDataset.query_tokens.copy()   # Avishek's hypothesis
                terms_repeat = [[t]*10 for t in terms]
                change = sum([rank_diff(Explainer, query, doc_id, terms_rep) for terms_rep in terms_repeat])/len(terms_repeat)

            else:
                terms = ['wikipedia']
                change = rank_diff(Explainer, query, doc_id, terms)      

        print(f'Rank changed by: {change}')
        Change.append({query: change})
        AVG = sum([[*v.values()][0] for v in Change])/len(Change)
        Change.append({'avg': AVG})
    print(f'Average rank diff: {AVG}')
    save_dir = hparams['exp_fold'] / 'kendalls' / f"{hparams['Rerank_model']}"/ f"{hparams['optimize_method']}_rank_diff_by_perturb_2.json"
    with open(save_dir, 'w')as f:
        json.dump(Change, f)


def evaluate_remove_wikipedia(hparams: Dict[str, Any]):
    #Explainer = init_explainer(hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
    Change = []
    for i, q_id in tqdm(enumerate(hparams['q_ids']), desc='Processing query'):  
        Explainer._init_query(q_id, True) 
        query = Explainer.InferenceDataset.query
        rank = Explainer.InferenceDataset.rank.copy()
        prediction = Explainer.InferenceDataset.prediction.copy()
        length = len(rank)
        for t in range(length):
            doc_id = rank[t]
            doc = Explainer.InferenceDataset.top_docs[doc_id]
            if 'wikipedia' in doc:
                adv_doc = doc.replace('wikipedia', '[MASK]').replace('encyclopedia', '[MASK]')
                adv_inp = Explainer.InferenceDataset.get_single_input(query, adv_doc)
                score = Explainer.model(adv_inp).data.item()
                prediction[doc_id] = score
                new_rank = np.argsort(-prediction)
                new_t = np.where(new_rank==doc_id)[0].item()
                rank_diff = new_t - t
                print(f'Rank changed by: {rank_diff}')
                Change.append({query:rank_diff})
                break
            else:
                continue
            
    AVG = sum([[*v.values()][0] for v in Change])/len(Change)
    Change.append({'avg': AVG})
    print(f'Average rank diff: {AVG}')
    save_dir = hparams['exp_fold'] / 'kendalls' / f"{hparams['Rerank_model']}"/ f"{hparams['optimize_method']}_rank_diff_remove_wiki.json"
    with open(save_dir, 'w')as f:
        json.dump(Change, f)


def computation_cost(hparams: Dict):
     """ Evaluate kendalltau scores of expansion terms. """
     if hparams['EXP_model'] == 'multi':
        exp_model = EXP_model.copy()
     else:
        exp_model = [hparams['EXP_model']]
     #Explainer = init_explainer(hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])
     Explainer = init_explainer(hparams['queries_file'], hparams['top_file'], hparams['dataset'], hparams['Rerank_model'], hparams['FOLD_NAME'])

     for q_id in tqdm(hparams['q_ids'], desc='Processing query'):
        start = time.time()
        
        Explainer._init_query(q_id, True)
        if hparams['optimize_method'] == 'gradient':
            expansion, utility = explain_by_gradients(hparams['Rerank_model'], Explainer.model, Explainer.InferenceDataset.tokenizer,
                            Explainer.InferenceDataset.query, Explainer.InferenceDataset.top_docs, Explainer.InferenceDataset.prediction, hparams['max_k'])
        elif hparams['optimize_method'] == 'query_tokens':
            expansion, utility = explain_by_query(Explainer.InferenceDataset.query_tokens)
        else:
            expansion, utility = explain_single(q_id, hparams, exp_model)
            expansion = list(set(expansion + Explainer.InferenceDataset.query_tokens.copy() ))   # query + expansion
        end = time.time()
        duration = (end-start)/60
        print(duration)

            
class MultiplexListwiseExplainer(BaseListwiseExplainer):
    #def __init__(self, hparams: Dict[str, Any]):
    #    gen_candidates(hparams)
    #    gen_matrix(vars(args))
    
    def __init__(self, index_path, indexer_type):
        self.index_path = index_path
        # as of now this is dummy, we do not process anything
        self.indexer_type = indexer_type

    def _evaluate(self, qid, query_tokens, candidates_tokens, doc_pairs, explainer, hparams: Dict[str, Any]):
        """ Evaluate kendalltau scores of expansion terms. """
        if hparams['EXP_model'] == 'multi':
            exp_model = EXP_model.copy()
        else:
            exp_model = [hparams['EXP_model']]

        CORREL, Expansion = [], []
        try:
            if hparams['optimize_method'] == 'gradient':
                expansion, utility = explain_by_gradients(hparams['Rerank_model'], explainer.model, explainer.InferenceDataset.tokenizer,
                                explainer.InferenceDataset.query, explainer.InferenceDataset.top_docs, explainer.InferenceDataset.prediction, hparams['max_k'])

            elif hparams['optimize_method'] == 'query_tokens':
                #expansion, utility = explain_by_query(explainer.InferenceDataset.query_tokens)
                expansion, utility = explain_by_query(query_tokens)
            
            else:
                #print('coming to else part')
                expansion, utility = explain_single(qid, explainer, hparams, exp_model, candidates_tokens, doc_pairs)
                #expansion = list(set(expansion + explainer.InferenceDataset.query_tokens.copy() ))   # query + expansion

                expansion = list(set(expansion + query_tokens.split()))   # query + expansion
                print(f'expansion terms : {expansion}')
                print(f'utility : {utility}')
                corr_0, corr_1, corr_2, corr_3 = explainer.evaluate_fidelity(hparams['dense_ranking'], hparams['dense_ranking_score'], expansion, exp_model, hparams['top_d'], vote =hparams['vote'], tolerance=hparams['tolerance'])
                #corr_0, corr_1, corr_2, corr_3 = 0, 0, 0, 0
                CORREL.append((corr_0, corr_1, corr_2, corr_3, utility))
                print(f'kendall tau: {corr_0}, {corr_1}, {corr_2}, {corr_3}, {utility}')
                Expansion.append(expansion)
        except Exception as e:
            print(f"Error happend while processing {e}")
        
        #return CORREL, Expansion
    
        # compute average
        
        AVG_strict, AVG_relax = [], []
        for i in range(len(CORREL[0])):
            avg_s = sum([C[i] for C in CORREL])/len(CORREL)
            relax = [C for C in CORREL if C[i]]
            if relax:
                avg_r = sum([C[i] for C in relax])/len(relax)
            else:
                avg_r = 0
            AVG_strict.append(avg_s)
            AVG_relax.append(avg_r)
        CORREL.append(AVG_strict)
        CORREL.append(AVG_relax)
        
        return CORREL, Expansion

    def generate_candidates(self, qid, query_str, hparams):
        """
        Generate initial candidate terms for top-$k$ documents
        """
        # read queries for explanation
        Explainer = init_explainer(query_str, hparams['dense_ranking'], self.index_path)
        candidates = Explainer.get_candidates_reranker(qid, hparams['top_d'], hparams['top_tfidf'], hparams['top_r'], hparams['dense_ranking'], hparams['candi_method'])
        #print(candidates)

        return candidates

    def show_matrix(self, qid, query_str, hparams):
        """
        Showing the term-document pair preference matrix
        Use only one simple explainer
        """
        # read queries for explanation
        Explainer = init_explainer(query_str, hparams['dense_ranking'], self.index_path)
        candidates = Explainer.get_candidates_reranker(qid, hparams['top_d'], hparams['top_tfidf'], hparams['top_r'], hparams['dense_ranking'], hparams['candi_method'])
        candidates_tokens = []
        
        if isinstance(candidates, dict):
            candidates_tokens = list(candidates.keys())
        
        #Explainer._init_query(qid, True)
        doc_pairs = Explainer.sample_doc_pair(hparams['dense_ranking'], hparams['dense_ranking_score'], hparams['ranked'], hparams['pair_num'], hparams['style'], hparams['tolerance'])        
        print(doc_pairs)
        #return
        if doc_pairs:
            matrix = Explainer.build_matrix(hparams['dense_ranking'], candidates_tokens, doc_pairs, hparams['EXP_model'])
            print(matrix)
            return matrix

    def explain(self, qid, query_str, hparams):
        
        # read queries for explanation
        Explainer = init_explainer(query_str, hparams['dense_ranking'], self.index_path)

        candidates = Explainer.get_candidates_reranker(qid, hparams['top_d'], hparams['top_tfidf'], hparams['top_r'], hparams['dense_ranking'], hparams['candi_method'])
        #print(candidates)
        #return
        candidates_tokens = []
        if isinstance(candidates, dict):
            candidates_tokens = list(candidates.keys())
        
        #Explainer._init_query(qid, True)
        doc_pairs = Explainer.sample_doc_pair(hparams['dense_ranking'], hparams['dense_ranking_score'], hparams['ranked'], hparams['pair_num'], hparams['style'], hparams['tolerance'])        
        print(doc_pairs)
        #return
        if doc_pairs:
            
            #matrix = Explainer.build_matrix(hparams['dense_ranking'], candidates_tokens, doc_pairs, hparams['EXP_model'])
            # seems like build matrix is redundant
            #return
            CORREL, Expansion = self._evaluate(qid, query_str, candidates_tokens, doc_pairs, Explainer, hparams)
            
            print(f'Correlation : {CORREL[0]}')
            print(f'Expansion terms : {Expansion[0]}')
            return CORREL, Expansion  

"""     
if __name__ == '__main__':

    ap = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--FOLD_NAME', type=str, help='Name of the fold. e.g. fold_1')
    ap.add_argument('--Rerank_model', type=str, help='Name of the rerank-model. e.g. bert/drmm')
    ap.add_argument('--dataset', type=str, default='clueweb09', help='Dataset: cluweb09')
    ap.add_argument('--query_type', type=str, default='test', help='train, dev, test' )
    ap.add_argument('--top_d', type=int, help='Extract candidates from top-d ranked documents.')
    ap.add_argument('--top_tfidf', type=int, default=100, help='Keep top_tfidf candidates by tfidf scores')
    ap.add_argument('--top_r', type=int, help='Eventually keep top_r candidates after permutation filtering.')
    ap.add_argument('--candi_method', type=str, default='perturb', help='method to generate candidates')
    ap.add_argument('--ranked', type=int, default=5,  help='Consider top ranked docs during sampling documents.')
    ap.add_argument('--pair_num', type=int, help='The number of sampled doc_pairs')
    ap.add_argument('--max_k', type=int, default=10, help='Explained query token number, max.')
    ap.add_argument('--min_k', type=int, default=4, help='Explained query token number, min.')
    ap.add_argument('--candidates_num', type=int, help='The number of candiate query tokens to build matrix.')
    ap.add_argument('--style', type=str, help='How to sample doc pairs?, random.')
    ap.add_argument('--vote', type=int, default=2, help='how many explainers should agree with a rank?')
    ap.add_argument('--tolerance', type=float, help='prediction score difference, bert: 2, drmm: 0.1')
    ap.add_argument('--EXP_model', type=str, help='language_model, saliency, semantic, multi')
    ap.add_argument('--optimize_method', type=str, default='greedy', help='greedy, linear_program, dense_learn, feature_select, gradient, geno, geno_multi. our method multiplex == geno_multi. ')
    ap.add_argument('--mode', type=str, help='candidates, matrix, or explain?')
    args = ap.parse_args()

    exp_fold = project_dir / 'Exp_results' / args.dataset / args.FOLD_NAME
    args.exp_fold = exp_fold
    
    # read queries for explanation
    if args.query_type == 'test':
        args.queries_file = project_dir / 'Datasets/src' / args.dataset / 'queries.tsv'   # trec-dl test.
        args.top_file = args.queries_file.parent / 'top.tsv'
    else:
        args.queries_file = project_dir / 'Datasets/src' / args.dataset /  f'queries_{args.query_type}.tsv'
        args.top_file = args.queries_file.parent / f'top_{args.query_type}.tsv'

    q_ids_dir = project_dir / 'Datasets/src' / args.dataset / 'folds' / args.FOLD_NAME / f'{args.query_type}_ids.txt'
    with open(q_ids_dir, 'r')as f:
        q_ids = [l.strip() for l in f]
    
    args.q_ids = q_ids   # mamximum 1000
    random_seed = 123
    seed_everything(random_seed, workers=True)

    print('arguments passed: \n', args)
    if args.mode == 'candidates':
        gen_candidates(vars(args))
    elif args.mode == 'matrix':
        gen_matrix(vars(args))
    elif args.mode == 'explain':
        evaluate(vars(args))
    elif args.mode == 'ablation':
        ablation(vars(args))
    elif args.mode == 'coverage':
        compute_coverage(vars(args))
    elif args.mode == 'fixtau':
        correct_local_tau(vars(args))
    elif args.mode == 'modeldiff':
        model_diff(vars(args))
    
    elif args.mode == 'rank_diff_by_perturb':
        evaluate_exp_by_perturb(vars(args))

    elif args.mode == 'rank_diff_by_remove':
        evaluate_remove_wikipedia(vars(args))
    elif args.mode == 'computation':
        computation_cost(vars(args))

    elif args.mode == 'query_filter':
        filter_query(vars(args))

    else:
        raise ValueError(f"mode: {args.mode}")
"""
