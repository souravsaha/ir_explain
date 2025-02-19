from collections import namedtuple
from typing import List
#from scipy.stats.stats import _contains_nan
#from scipy.stats.stats import mstats_basic
from scipy.stats._stats import _kendall_dis
import scipy.special as special
import numpy as np
from itertools import combinations

KendalltauResult = namedtuple('KendalltauResult', ('correlation', 'pvalue'))

def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy


def _kendall_p_exact(n, c):
    # Exact p-value, see Maurice G. Kendall, "Rank Correlation Methods" (4th Edition), Charles Griffin & Co., 1970.
    if n <= 0:
        raise ValueError('n ({n}) must be positive')
    elif c < 0 or 4*c > n*(n-1):
        raise ValueError(f'c ({c}) must satisfy 0 <= 4c <= n(n-1) = {n*(n-1)}.')
    elif n == 1:
        prob = 1.0
    elif n == 2:
        prob = 1.0
    elif c == 0:
        prob = 2.0/np.math.factorial(n) if n < 171 else 0.0
    elif c == 1:
        prob = 2.0/np.math.factorial(n-1) if n < 172 else 0.0
    elif 4*c == n*(n-1):
        prob = 1.0
    elif n < 171:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3,n+1):
            new = np.cumsum(new)
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = 2.0*np.sum(new)/np.math.factorial(n)
    else:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3, n+1):
            new = np.cumsum(new)/j
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = np.sum(new)

    return np.clip(prob, 0, 1)


def kendalltau(x, y, initial_lexsort=None, nan_policy='propagate',
               method='auto', variant='b'):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same "
                         f"size, found x-size {x.size} and y-size {y.size}")
    elif not x.size or not y.size:
        # Return NaN if arrays are empty
        return KendalltauResult(np.nan, np.nan)

    # check both x and y
    cnx, npx = _contains_nan(x, nan_policy)
    cny, npy = _contains_nan(y, nan_policy)
    contains_nan = cnx or cny
    if npx == 'omit' or npy == 'omit':
        nan_policy = 'omit'

    if contains_nan and nan_policy == 'propagate':
        return KendalltauResult(np.nan, np.nan)

    elif contains_nan and nan_policy == 'omit':
        x = ma.masked_invalid(x)
        y = ma.masked_invalid(y)
        if variant == 'b':
            return mstats_basic.kendalltau(x, y, method=method, use_ties=True)
        else:
            raise ValueError("Only variant 'b' is supported for masked arrays")

    if initial_lexsort is not None:  # deprecate to drop!
        warnings.warn('"initial_lexsort" is gone!')

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
                (cnt * (cnt - 1.) * (cnt - 2)).sum(),
                (cnt * (cnt - 1.) * (2 * cnt + 5)).sum())

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)  # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)  # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        return KendalltauResult(np.nan, np.nan)

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - dis  # here is the modification
    if variant == 'b':
        tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)
    elif variant == 'c':
        minclasses = min(len(set(x)), len(set(y)))
        tau = 2 * con_minus_dis / (size ** 2 * (minclasses - 1) / minclasses)
    else:
        raise ValueError(f"Unknown variant of the method chosen: {variant}. "
                         "variant must be 'b' or 'c'.")

    # Limit range to fix computational errors
    tau = min(1., max(-1., tau))

    # The p-value calculation is the same for all variants since the p-value
    # depends only on con_minus_dis.
    if method == 'exact' and (xtie != 0 or ytie != 0):
        raise ValueError("Ties found, exact method cannot be used.")

    if method == 'auto':
        if (xtie == 0 and ytie == 0) and (size <= 33 or
                                          min(dis, tot - dis) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'

    if xtie == 0 and ytie == 0 and method == 'exact':
        #pvalue = mstats_basic._kendall_p_exact(size, min(dis, tot - dis))
        pvalue = _kendall_p_exact(size, min(dis, tot - dis))

    elif method == 'asymptotic':
        # con_minus_dis is approx normally distributed with this variance [3]_
        m = size * (size - 1.)
        var = ((m * (2 * size + 5) - x1 - y1) / 18 +
               (2 * xtie * ytie) / m + x0 * y0 / (9 * m * (size - 2)))
        pvalue = (special.erfc(np.abs(con_minus_dis) /
                               np.sqrt(var) / np.sqrt(2)))
    else:
        raise ValueError(f"Unknown method {method} specified.  Use 'auto', "
                         "'exact' or 'asymptotic'.")

    return KendalltauResult(tau, pvalue)


def kendalltau_gap(x, y, tolerance:float = 2.0):
    """ Only consider pairs with the value of gap being >= 2"""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same "
                         f"size, found x-size {x.size} and y-size {y.size}")
    elif not x.size or not y.size:
        # Return NaN if arrays are empty
        raise ValueError("Empty arrays")

    # check both x and y
    cnx, _ = _contains_nan(x)
    cny, _ = _contains_nan(y)
    contains_nan = cnx or cny
    if contains_nan:
        raise ValueError('Contains NAN.')
    
    # generate all pairs with tolerance gap.
    """ with or without perm should be the same, no bug."""
    # perm = np.argsort(-x)
    all_pairs = list(combinations(range(len(x)), 2))
    tot_pairs = [(a, b) for a, b in all_pairs if abs(x[a]-x[b]) >= tolerance ]  # no tie
    #concord = sum([np.sign(x[a]-x[b]) * np.sign(y[a] - y[b])+1 for a, b in tot_pairs])/2    # bug, sign=0
    concord = [np.sign(x[a]-x[b])*np.sign(y[a]-y[b]) for a, b in tot_pairs]
    concord_num = concord.count(1)
    if tot_pairs:
        tau = float(concord_num) / len(tot_pairs) + 0.01  #smooth
        return tau
    else:
        return 0


def coverage_multi(x: List[float], y: List[List[float]], vote: int = 2, tolerance: float = 2.0):
    """ Compute the concordant coverage by multiple vote"""
    all_pairs = list(combinations(range(len(x)), 2))
    tot_pairs = [(a, b) for a, b in all_pairs if abs(x[a]-x[b]) >= tolerance ]  # no tie
    concord_all = []
    for y_single in y:
        concord = [np.sign(x[a]-x[b])*np.sign(y_single[a]-y_single[b]) for a, b in tot_pairs]
        concord_all.append(concord)
    concord_num = 0
    for p in range(len(tot_pairs)):
        c_p = [concord[p] for concord in concord_all]
        if c_p.count(1) >= vote:   # multiple agree
            concord_num += 1

    if tot_pairs:
        tau = float(concord_num) / len(tot_pairs) + 0.01  # smooth 
        return tau
    else:
        return 0


    


