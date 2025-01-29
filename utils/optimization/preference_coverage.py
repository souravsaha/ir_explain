from typing import Tuple, List
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
import numpy as np


def feature_select(candidates: List[str], weights: List[List[float]], select_max: int, select_min: int, select_weights=np.array([])):
    size = len(candidates)
    matrix = np.array(weights)  # term_num, doc_pair_num
    assert(size == matrix.shape[0])
    positive_pairs = matrix.transpose(1, 0)   # doc_pair_num, term_num
    negative_pairs = -positive_pairs
    Data = np.concatenate((positive_pairs, negative_pairs), axis=0)
    Label = np.array([1]*positive_pairs.shape[0]+[0]*positive_pairs.shape[0])
    model = LogisticRegression(solver='lbfgs')
    rfe = RFE(model, select_max)
    fit = rfe.fit(Data, Label)
    picked = np.where(fit.support_ == True)[0]
    expansion = list(np.array(candidates)[picked])
    return expansion, 0

def linear_program(candidates: List[str], weights: List[List[float]], select_max: int, select_min: int, select_weights=np.array([])) -> Tuple[List[str], float]:
    """ define a problem with $size$ selection dimension, which minimizes the negative utilities."""
    size = len(candidates)
    weights = np.array(weights)
    assert(size == weights.shape[0])
    selection = cp.Variable(size, boolean=True)
    weights_cp = cp.Constant(weights)
    constrains = []
    if select_max:
        constrain_max = cp.sum(selection) <= select_max
        constrains.append(constrain_max)

    else:
        constrain_max = cp.sum(selection) <= select_min
        constrains.append(constrain_max)

    if select_min:
        constrain_min = cp.sum(selection) >= select_min
        constrains.append(constrain_min)
    else:
        raise ValueError('Missing minimum select number.')
    
    if select_weights.any():
        select_weights = cp.Constant(select_weights.reshape(1, size))
        utility_sum = cp.sum( cp.sum(cp.reshape(selection, (size, 1)) @ select_weights, axis=0) @ weights_cp)
        # utility_sum = cp.sum(select_weights @ (selection @ weights))
    else:
        # utility_reduce_neg = cp.sum(cp.neg(selection @ weights))   # all negative values, minimize abs of it.
        utility_sum = cp.sum(selection @ weights_cp)   # all negative values, minimize abs of it.
    problem = cp.Problem(cp.Maximize(utility_sum), constrains)
    problem.solve(solver=cp.GLPK_MI)
    
    coverage = ((selection.value @ weights) >= 0).sum()
    covered_ratio = float(coverage) / weights.shape[1]
    picked = np.where(selection.value > 0)[0]
    expansion = list(np.array(candidates)[picked])
    return expansion, covered_ratio


def greedy(candidates_arg: List[str], matrix_arg: List[List[float]], select_max: int, select_min: int = 1, select_weights=np.array([])) -> Tuple[List[str], float]:
    #print(f'Optimizing preference coverage in greedy...')
    candidates = candidates_arg.copy()
    matrix = matrix_arg.copy()  # copy the argument, otherwise it'll be modified.     
    assert(len(candidates) == len(matrix))
    expansion = []
    pcov = np.zeros(len(matrix[0]))   # init expansion and features
    for i in range(select_max):
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
            expansion.append(picked)
            pcov = pcov_update.copy()
            picked_id = candidates.index(picked)
            del candidates[picked_id]  # remove the picked item from candidates.
            del matrix[picked_id]  # remove the picked features from matrix
        else:
            # print(f'Cannot find any token which improves the utility, pick the maximum value instead.')
            '''
            # pick the candidate with maximum pcv.
            picked_id = np.argmax([(pcov+array).sum() for array in matrix])
            print(f'Picked: {candidates[picked_id]}')
            expansion.append(candidates[picked_id])
            pcov_update = pcov + matrix[picked_id]
            pcov = pcov_update.copy()
            covered = (pcov >= 0.0).sum()
            del candidates[picked_id]
            del matrix[picked_id]
            '''
            pass
    return expansion, covered/float(len(pcov))


def greedy_multi(candidates_arg: List[str], matrix_arg: List[List[List[float]]], select_max: int, select_min: int = 1, select_weights=np.empty([])) -> Tuple[List[str], float]:
    #print(f'Optimizing preference coverage in greedy...')
    candidates = candidates_arg.copy()
    term_num = len(candidates)
    assert len(matrix_arg) == 3
    matrix_a, matrix_b, matrix_c = matrix_arg.copy()  # copy the argument, otherwise it'll be modified.     
    assert term_num == len(matrix_a)
    doc_num = len(matrix_a[0])
    
    expansion = []
    pcov = np.zeros(doc_num)   # init expansion and features
    for i in range(select_max):
        covered = (pcov > 0.0).sum()
        pcov_update = pcov.copy()
        picked = None 
        for candidate, array_a, array_b, array_c in zip(candidates, matrix_a, matrix_b, matrix_c):
            pcov_expand_a = pcov + array_a 
            pcov_expand_b = pcov + array_b
            pcov_expand_c = pcov + array_c 
            pcov_expand_combine = (np.sign(pcov_expand_a) + np.sign(pcov_expand_b) + np.sign(pcov_expand_c) + 3)/float(2) 
            u = (pcov_expand_combine > 0.0).sum()

            if covered < u:   # always pick the one with largest utility.
                covered = u
                picked = candidate
                pcov_update = ((pcov_expand_a + pcov_expand_b + pcov_expand_c)/3).copy()
            
        if picked:
            expansion.append(picked)
            pcov = pcov_update.copy()
            picked_id = candidates.index(picked)
            del candidates[picked_id]  # remove the picked item from candidates.
            del matrix_a[picked_id]  # remove the picked features from matrix
            del matrix_b[picked_id]
            del matrix_c[picked_id]
        else:
            pass

    return expansion, covered/float(doc_num)
