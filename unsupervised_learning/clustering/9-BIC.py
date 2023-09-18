#!/usr/bin/env python3
"""
A function that finds the best number of clusters for a GMM using
the Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    function that finds the best number of clusters for a GMM using
    the Bayesian Information Criterion
    X: is a numpy.ndarray of shape (n, d) containing the data set
    kmin: is a positive integer containing the minimum number of clusters to check for (inclusive)
    kmax: is a positive integer containing the maximum number of clusters to check for (inclusive)
        If kmax is None, kmax should be set to the maximum number of clusters possible
   iterations: is a positive integer containing the maximum number of iterations for the EM algorithm
   tol: is a non-negative float containing the tolerance for the EM algorithm
   verbose: is a boolean that determines if the EM algorithm should print information to the standard output
   Returns: best_k, best_result, l, b, or None, None, None, None on failure
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    # X: array of shape (n, d) containing the data set
    n, d = X.shape

    # Define pi_t, m_t, S_t: arrays containing the relevant
    # parameters for all the clusters
    all_pis = []
    all_ms = []
    all_Ss = []
    all_lkhds = []
    all_bs = []

    # Iterate over the ((kmax + 1) - kmin) clusters
    for k in range(kmin, kmax + 1):
        pi, m, S, g, lkhd = expectation_maximization(X, k, iterations,
                                                     tol, verbose)
        all_pis.append(pi)
        all_ms.append(m)
        all_Ss.append(S)
        all_lkhds.append(lkhd)
        # p: the number of parameters required for the model
        p = (k * d * (d + 1) / 2) + (d * k) + (k - 1)
        # b: array containing the BIC value for each cluster size tested
        b = p * np.log(n) - 2 * lkhd
        all_bs.append(b)

    all_lkhds = np.array(all_lkhds)
    all_bs = np.array(all_bs)
    best_k = np.argmin(all_bs)
    best_result = (all_pis[best_k], all_ms[best_k], all_Ss[best_k])

    return best_k+1, best_result, all_lkhds, all_bs
