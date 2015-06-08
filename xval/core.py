#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date: Mar. 2015

from __future__ import print_function, division


import numpy as np
from numpy.linalg import solve

import scipy as sp
from scipy import log, exp
from scipy.special import gamma, beta
from scipy.optimize import minimize

import numdifftools as nd



#------------------------------------------------------------------------------


def Y_of_T(t, LAMBDA=1):
    t = np.asarray(t)
    sel = t > LAMBDA
    res = np.empty(shape=t.shape)
    res.fill(np.nan)
    res[sel] = -log(-log(1 - LAMBDA / t[sel]))

    return res

#------------------------------------------------------------------------------# -----------------------------------------------------------------------------


def F_of_T(t, LAMBDA=1):
    return 1 - 1. / (LAMBDA * t)

# -----------------------------------------------------------------------------


def F_of_Y(y):
    return exp(-exp(-y))

# -----------------------------------------------------------------------------


def T_of_F(f, LAMBDA=1):
    return 1 / (1 - f) / LAMBDA

# -----------------------------------------------------------------------------


def T_of_Y(y, LAMBDA=1):
    return LAMBDA / (1 - exp(-exp(-y)))

# -----------------------------------------------------------------------------


def Y_of_F(f):
    return -log(-log(f))

# -----------------------------------------------------------------------------

def minim(func, init, method='Nelder-Mead', probs=(0.025, 0.975), **kwargs):

    n = len(init)

    uncert = kwargs.pop('uncert', True)

    if method == 'Nelder-Mead':
        options = {'xtol': 1e-10, 'disp': False, 'maxiter' : 10000, 
                'maxfev' : 10000}

        
        kwargs = dict(options=options)
    


    # minimize nagative log-likelihood
    res = minimize(func, init, method=method, options=options)

    # calculate some other properties
    if res.success & uncert:
        # hessian
        hess = nd.Hessian(func)(res.x)
        
        res.hess = hess

        # covariance matrix
        res.covmat = solve(hess, np.identity(n))
        # standard error
        res.std_err = np.sqrt(np.diag(res.covmat))
        
        # confidence intervall
        ppf = sp.stats.norm().ppf(probs)
        
        # confidence intervall of parametes (delta method)
        res.conf_int = np.atleast_2d(res.x).T + np.atleast_2d(res.std_err).T*ppf
        # half the 95% confidence intervall
        res.std95 = res.std_err*sp.stats.norm().ppf(0.975)
        
    else:
        res.covmat = np.ones([n, n]) * np.nan

    return res

# -----------------------------------------------------------------------------
def propagate_uncertainty(covmat, coeff, idx=None):

    coeff = np.atleast_2d(coeff)

    if idx is not None:
        x = np.ix_(idx, idx)
        covmat = covmat[x]

    uncert = np.sqrt(np.dot(np.dot(coeff, covmat), coeff.T))

    return uncert


# -----------------------------------------------------------------------------
# generic function for all l-moment fits


def __fit_gen_lmom__(dist, data, LAMBDA=1., ret=(5, 10, 50, 100, 500), chi=False):
    from xval import XGEV, XGPD
    
    data = check_data(data)
    is_GEV = dist == 'GEV'
    is_GUMBEL = dist == 'GUMBEL'
    is_EXPON = dist == 'EXPON'
    is_GPD = dist == 'GPD'

    if is_EXPON or is_GPD:
        chi = chi if chi else np.min(data)
        if np.any(data < chi):
            raise Exception('data must be larger than threshold chi')

    kthresh = 0.000001  # only needed for GUMBEL

    ret = np.asarray(ret) / LAMBDA
    length = len(data)
    data_sort = data.copy()
    data_sort.sort()
    jm1 = np.arange(length)
    jm2 = np.arange(length) - 1

    jm1.reshape(data.shape)
    jm2.reshape(data.shape)

    # CALCULATE PSTAR AND B VALUES
    # (see Hosking 1990, Eq. 2.3 and p. 114.)
    p00, p10, p11, p20, p21, p22 = 1., -1, 2., 1., -6., 6.
    b0 = data_sort.mean()
    b1 = np.mean(jm1 / (length - 1) * data_sort)
    b2 = np.mean(jm1 * jm2 / (length - 1) / (length - 2) * data_sort)

    # ESTIMATE L-MOMENTS
    l1 = p00 * b0
    l2 = p10 * b0 + p11 * b1
    l3 = p20 * b0 + p21 * b1 + p22 * b2

    # ESTIMATE PARAMETERS FROM L-MOMENTS (see Hosking 1990, Table 2)

    if is_GEV or is_GUMBEL:
        zz = 2 / (3 + l3 / l2) - sp.log(2) / sp.log(3)
        k = 7.8590 * zz + 2.9554 * zz ** 2 if is_GEV else kthresh
        gk = gamma(1 + k)
        alpha = l2 * k / ((1 - 2 ** (-k)) * gk)
        chi = l1 + alpha * (gk - 1) / k
        if is_GUMBEL:
            k = 0
    elif is_EXPON or is_GPD:  # notice that l1 actually is l1-chi
        k = 0. if is_EXPON else (l1 - chi) / l2 - 2.0
        alpha = (1 + k) * (l1 - chi)

    # DETERMINE EXTREME VALUES FOR GIVEN RETURN PERIODS
    f = F_of_T(LAMBDA, ret)

    if is_GEV or is_GUMBEL:
        fitted = XGEV(np.asarray(f), chi, alpha, k)
    elif is_EXPON or is_GPD:
        fitted = XGPD(np.asarray(f), alpha, chi, k)

    res = {'data': data,
           'LAMBDA': LAMBDA,
           'alpha': alpha,
           'chi': chi,
           'k': k,
           'fit': {'ret': ret, 'fitted': fitted},
           'dist': dist,
           'estim': 'lmom'
           }

    return res


# =============================================================================


def must_be_any_of(arg, argname, anyof):
    if arg not in anyof:
        formstr = "%s must be any of " % argname + len(anyof) * "'%s' " % anyof
        raise Exception(formstr)

# -----------------------------------------------------------------------------


def check_data(data):
    data = np.squeeze(data)
    if data.ndim > 1:
        raise Exception("input data must be 1D")
    return data

# -----------------------------------------------------------------------------