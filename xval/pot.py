#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mathias Hauser
# Date: 2014


from __future__ import print_function, division

import numpy as np
from numpy.linalg import solve

import scipy as sp
from scipy import log, exp
from scipy.special import gamma, beta
from scipy.optimize import minimize
from scipy.stats import poisson, rankdata, pearsonr

from collections import OrderedDict

import numdifftools as nd

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext


from core import *

# -----------------------------------------------------------------------------


def XGPD(ff, alpha, chi, k):
    """
    extreme value for a given GPD parameters and
    # the cumulative frequency.
    # Invert cumulative pdf for Generalised Pareto Distribution.
    # Is used for random number generator.
    # see e.g. Palutikov et al. 1999.
    #
    # ff   : cumulative frequency
    # chi  : location parameter
    # alpha: scale parameter
    # k    : shape parameter

    # function is listable in argument ff
    # kthresh defines a threshold for k. if abs(k)<kthresh then the
    # Exponential Distribution is taken.
    """

    kthresh = 0.000001

    def expo(f):
        return -np.log(1 - f)

    def pare(f, k):
        return (1 - (1 - f) ** k) / k

    if abs(k) < kthresh:
        return alpha * expo(ff) + chi
    else:
        return alpha * pare(ff, k) + chi

#------------------------------------------------------------------------------


def gpd_lik(data, chi=0, tim=False):
    tim = tim if tim else len(data)
    kthresh = 0.000001  # threshold to distinguish EXPON
    length = len(data)     # sample size

    def gpd_lik_internal(params):
        LAMBDA, alpha, k = params

        # part from gpd only
        y = (data - chi) / alpha

        if abs(k) > kthresh:  # gpd
            y = (1 - k * y)
            if np.any(y <= 0) or alpha <= 0:
                ll_gpd = np.inf
            else:
                ll_gpd = length * \
                    sp.log(alpha) + (1 - 1 / k) * np.sum(sp.log(y))
        else:  # expon
            ll_gpd = length * sp.log(alpha) + y.sum()

        # add part from peak frequency
         # expand likelihood function to account for sample size distribution
         # see theoretical details in Coles p.82.
         # Here we deviate from the approach in Coles in the sense that lamda
         # (numb of exceedances per time unit) is considered as the third
         # parameter. Moreover, the the variance of lamda is modelled assuming
         # a poisson model for the exceedances rather than a binomial model.
         # This avoids knowledge of the full sample size (number of total
         # observations) which is not really known anyway if the exceedances
         # were determined after declustering. The assumption of the poisson
         # model is just that the exceedances are very rare, which is probably
         # a good assumption for for high thresholds anyway.
         # The formuli are in my notes.

        if LAMBDA <= 0:
            ll_poisson = np.inf
        else:
            ll_poisson = -poisson.logpmf(length, LAMBDA * tim)

        return ll_gpd + ll_poisson

    return gpd_lik_internal


#------------------------------------------------------------------------------


def expon_lik(data, chi=0, tim=False):
    tim = tim if tim else len(data)

    def expon_lik_internal(params):
        params = np.asarray([params[0], params[1], 0])
        return gpd_lik(data, chi, tim)(params)
    return expon_lik_internal

#------------------------------------------------------------------------------


def fitEXPON_mlik(data, LAMBDA=1, ret=(5, 10, 50, 100, 500), chi=False):
    data = check_data(data)
    chi = chi if chi else np.min(data)
    if np.any(data < chi):
        raise Exception('data must be larger than threshold chi')

    ret = np.asarray(ret) / LAMBDA
    length = len(data)

    # LIKELIHOOD ESTIMATOR FOR EXPONENTIAL IS ANALYTICAL
    alpha = np.mean(data - chi)
    var_alpha = alpha ** 2 / length
    k = 0.0
    var_lamda = LAMBDA ** 2 / length

    # DETERMINE EXTREME VALUES FOR GIVEN RETURN PERIODS
    fitted = XGPD(F_of_T(LAMBDA, ret), alpha, chi, k)

    # covariance matrix
    covmat = np.array([[var_alpha, 0], [0, var_lamda]])

    res = {'data': data,
           'LAMBDA': LAMBDA,
           'alpha': alpha,
           'chi': chi,
           'k': k,
           'fit': {'ret': ret, 'fitted': fitted},
           'cov': covmat,
           'dist': 'EXPON',
           'estim': 'mlik'
           }
    return res


#------------------------------------------------------------------------------


def fitGPD_mlik(data, LAMBDA=1, ret=(5, 10, 50, 100, 500), chi=False, **kwargs):
    data = check_data(data)
    chi = chi if chi else np.min(data)
    if np.any(data < chi):
        raise Exception('data must be larger than threshold chi')

    method = kwargs.get('method', 'Nelder-Mead')
    maxit = kwargs.get('maxit', 10000)

    ret = np.asarray(ret)

    length = len(data)
    tim = length / LAMBDA

    # INITIAL VALUES OF OPTIMIZATION
    # (exact alpha for EXPON, small k such that distribution has no upper bound
    #  this is different from Coles)
    init_alpha = np.mean(data - chi)
    init = np.asarray([LAMBDA, init_alpha, 0.])

    func = gpd_lik(data, chi, tim)

    # print(func(init))

    res = minimize(func, init, method=method, options={'xtol': 1e-10, 'disp': False})

    if res.success:
        LAMBDA, alpha, k = res.x
    else:
        alpha, k = np.nan, np.nan

    # DETERMINE EXTREME VALUES FOR GIVEN RETURN PERIODS
    if res.success:
        f = F_of_T(LAMBDA, ret)
        fitted = XGPD(np.asarray(f), alpha, chi, k)
    else:
        fitted = np.ones(shape=ret.shape) * np.nan

    # CONVARIENCE MATRIX OF PARAMETER ESTIMATES
    if res.success:
        hess = nd.Hessian(func)(res.x)
        covmat = solve(hess, np.identity(3))
    else:
        covmat = np.ones([3, 3]) * np.nan

    res = {'data': data,
           'LAMBDA': LAMBDA,
           'alpha': alpha,
           'chi': chi,
           'k': k,
           'fit': {'ret': ret, 'fitted': fitted},
           'dist': 'GPD',
           'cov': covmat,
           'estim': 'mlik',
           'method': method,
           'maxit': maxit
           }
    return res



# -----------------------------------------------------------------------------


def fitEXPON_lmom(data, LAMBDA=1., ret=(5, 10, 50, 100, 500), chi=False):
    return __fit_gen_lmom__('EXPON', data, LAMBDA, ret, chi)

# -----------------------------------------------------------------------------


def fitGPD_lmom(data, LAMBDA=1., ret=(5, 10, 50, 100, 500), chi=False):
    return __fit_gen_lmom__('GPD', data, LAMBDA, ret, chi)
