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
from scipy.stats import poisson, rankdata

from collections import OrderedDict

import numdifftools as nd

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter

#------------------------------------------------------------------------------


def Y_of_T(t, LAMBDA=1):
    t = np.asarray(t)
    sel = t > LAMBDA
    res = np.empty(shape=t.shape)
    res.fill(np.nan)
    res[sel] = -log(-log(1 - LAMBDA / t[sel]))

    return res

# -----------------------------------------------------------------------------


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


def XGEV(ff, alpha, chi, k):
    kthresh = 0.000001

    def gumb(f):
        return -sp.log(-sp.log(f))

    def weib(f, k):
        return (1. - sp.exp(k * sp.log(-sp.log(f)))) / k

    def gev(f, k):
        if f > 1 or f < 0:
            return np.nan

        if abs(k) < kthresh:
            return alpha * gumb(f) + chi
        else:
            return alpha * weib(f, k) + chi

    f = np.vectorize(gev)

    return f(ff, k)


# -----------------------------------------------------------------------------

def RGEV(alpha, chi, k, n):
    """
    random distributed number with Generalised Extreme Value 
    Distribution see e.g. Zwiers and Kharin 1998 
    chi  : location parameter
    alpha: scale parameter
    k    : shape parameter
    n    : number of samples
    """

    kthresh = 0.000001
    rr = np.random.uniform(size=n)
    return XGEV(rr, alpha, chi, k)


# -----------------------------------------------------------------------------

def XGPD(ff, alpha, chi, k):
    # ==================================================================
    # Returns extreme value for a given setting of GPD parameters and
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

# use currying to use the function for the optimization and hessian matrix calc


def gev_lik(data):
    def gev_lik_internal(params):

        chi = params[0]
        alpha = params[1]
        k = params[2]
        # computes neg log lik of gev model including gumbel limit
        kthresh = 0.000001     # threshold to distinguish GUMBEL
        length = len(data)     # sample size
        y = (data - chi) / alpha
        if abs(k) > kthresh:
            y = (1 - k * y)
            if np.any(y <= 0) or alpha <= 0:
                return 10 ** 9
            return length * sp.log(alpha) + np.sum(y ** (1 / k)) + np.sum(sp.log(y) * (1 - 1 / k))
        else:
            return length * sp.log(alpha) + np.sum(y) + np.sum(sp.exp(-y))
    return gev_lik_internal

#------------------------------------------------------------------------------

# Geophysical prior function PPI(kk) from Martins and Stedinger p. 740


def ppi(b):

    # parameters for geophys prior from Martins and Stedinger p. 740
    pp, qq = 6, 9
    if abs(b) < 0.5:
        return ((0.5 + b) ** (pp - 1) * (0.5 - b) ** (qq - 1)) / beta(pp, qq)
    else:
        return 0
    #sel = abs(b) < 0.5
    #b[sel] = ((0.5 + b[sel])**(pp-1)*(0.5-b[sel])**(qq-1))/beta(pp,qq)
    #b[np.invert(sel)] = 0


#------------------------------------------------------------------------------

def gev_plik(data):
    def gev_lik_internal(params):
        # computes neg log lik of gev model including gumbel limit
        chi = params[0]
        alpha = params[1]
        k = params[2]
        kthresh = 0.000001     # threshold to distinguish GUMBEL
        length = len(data)     # sample size

        y = (data - chi) / alpha
        prior = ppi(k)

        if abs(k) > kthresh:
            y = (1 - k * y)
            if np.any(y <= 0) or alpha <= 0 or prior == 0:
                return 10 ** 9
            return length * sp.log(alpha) + np.sum(y ** (1 / k)) + np.sum(sp.log(y) * (1 - 1 / k)) - np.log(prior)
        else:
            if alpha <= 0 or prior == 0:
                return 10 ** 9
            return length * sp.log(alpha) + np.sum(y) + np.sum(sp.exp(-y))
    return gev_lik_internal

#------------------------------------------------------------------------------


def gumbel_lik(data):
    def gumbel_lik_internal(params):
        params = np.asarray([params[0], params[1], 0])
        return gev_lik(data)(params)
    return gumbel_lik_internal

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
                ll_gpd = 10 ** 9
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
            ll_poisson = 10 ** 9
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


def fitGEV(data, dist='GEV', estim='mlik', **kwargs):

    must_be_any_of(dist, 'dist', ("GEV", "GUMBEL"))
    must_be_any_of(estim, 'estim', ("mlik", "pmlik", "lmom"))

    if dist == 'GEV':
        if estim == 'mlik':
            res = fitGEV_mlik(data, **kwargs)
        elif estim == 'lmom':
            res = fitGEV_lmom(data, **kwargs)
        elif estim == 'pmlik':
            res = fitGEV_pmlik(data, **kwargs)
        else:
            raise Exception('unknown method: %s for distribution GEV' % dist)
    else:
        if estim == 'mlik':
            res = fitGUMBEL_mlik(data, **kwargs)
        elif estim == 'lmom':
            res = fitGUMBEL_lmom(data, **kwargs)

    return res

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


def fitGUMBEL_mlik(data, LAMBDA=1,  ret=(5, 10, 50, 100, 500), **kwargs):
    data = check_data(data)
    ret = np.asarray(ret) / LAMBDA

    method = kwargs.get('method', 'Nelder-Mead')
    maxit = kwargs.get('maxit', 10000)

    length = len(data)

    # INITIAL VALUES OF OPTIMIZATION (taken from L-moments method)
    init_fit = fitGUMBEL_lmom(data, LAMBDA, ret=2 / LAMBDA)

    # print(init_fit)

    init_alpha = init_fit['alpha']
    init_chi = init_fit['chi']

    init = np.asarray([init_chi, init_alpha])

    # initialize the function with the data (curried fcn)
    func = gumbel_lik(data)

    res = minimize(func, init, method=method,
                   options={'xtol': 1e-10, 'disp': False})

    if res.success:
        chi, alpha = res.x
        k = 0
    else:
        chi, alpha, k = np.nan, np.nan, np.nan

    # DETERMINE EXTREME VALUES FOR GIVEN RETURN PERIODS
    if res.success:
        f = F_of_T(LAMBDA, ret)
        fitted = XGEV(np.asarray(f), alpha, chi, k)
    else:
        fitted = np.ones(shape=ret.shape) * np.nan

    # CONVARIENCE MATRIX OF PARAMETER ESTIMATES
    if res.success:
        hess = nd.Hessian(func)(res.x)
        covmat = solve(hess, np.identity(2))
    else:
        covmat = np.ones([3, 3]) * np.nan

    res = {'data': data,
           'LAMBDA': LAMBDA,
           'alpha': alpha,
           'chi': chi,
           'k': k,
           'fit': {'ret': ret, 'fitted': fitted},
           'cov': covmat,
           'dist': 'GEV',
           'estim': 'mlik',
           'method': method,
           'maxit': maxit
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

    res = minimize(func, init, method=method,
                   options={'xtol': 1e-10, 'disp': False})

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

#------------------------------------------------------------------------------


# calculates fit to GEV for mlik and pmlik
def __fitGEV_mlik_gen__(estim_method, data, LAMBDA=1,  ret=(5, 10, 50, 100, 500), **kwargs):
    data = check_data(data)
    #must_be_any_of(fit_method, 'fit_method', ('mlik', 'pmlik'))
    is_mlik = estim_method == 'mlik'

    ret = np.asarray(ret) / LAMBDA

    method = kwargs.get('method', 'Nelder-Mead')
    maxit = kwargs.get('maxit', 10000)

    length = len(data)

    # INITIAL VALUES OF OPTIMIZATION (taken from L-moments method)
    init_fit = fitGEV_lmom(data, LAMBDA, ret=2 / LAMBDA)

    init_alpha = init_fit['alpha']
    init_chi = init_fit['chi']
    init_k = init_fit['k']
    # prevent initial values outside +-0.5
    if not is_mlik:
        init_k = min(max(-0.3, init_k), 0.1)

    init = np.asarray([init_chi, init_alpha, init_k])

    # initialize the function with the data (curried fcn)
    if is_mlik:
        func = gev_lik(data)
    else:
        func = gev_plik(data)

    res = minimize(func, init, method=method,
                   options={'xtol': 1e-10, 'disp': False})

    if res.success:
        chi, alpha, k = res.x
    else:
        chi, alpha, k = np.nan, np.nan, np.nan

    # DETERMINE EXTREME VALUES FOR GIVEN RETURN PERIODS
    if res.success:
        f = F_of_T(LAMBDA, ret)
        fitted = XGEV(np.asarray(f), alpha, chi, k)
    else:
        fitted = np.ones(shape=ret.shape) * np.nan

    # CONVARIENCE MATRIX OF PARAMETER ESTIMATES
    if res.success:
        hess = nd.Hessian(func)(res.x)
        covmat = solve(hess, np.identity(3))
        # print(covmat)
    else:
        covmat = np.ones([3, 3]) * np.nan

    res = {'data': data,
           'LAMBDA': LAMBDA,
           'alpha': alpha,
           'chi': chi,
           'k': k,
           'fit': {'ret': ret, 'fitted': fitted},
           'dist': 'GEV',
           'cov': covmat,
           'estim': estim_method,
           'method': method,
           'maxit': maxit
           }
    return res


#------------------------------------------------------------------------------

def fitGEV_pmlik(data, LAMBDA=1,  ret=(5, 10, 50, 100, 500), **kwargs):
    """
    Estimates parameters of GEV-fit to sample data.
    The estimation is based on maximum likelihood.
    Also estimates values with a given return period.
    """

    # call generic mlik function
    return __fitGEV_mlik_gen__('pmlik', data, LAMBDA=1,  ret=ret, **kwargs)

#------------------------------------------------------------------------------


def fitGEV_mlik(data, LAMBDA=1, ret=(5, 10, 50, 100, 500), **kwargs):
    """
    Estimates parameters of GEV-fit to sample data.
    The estimation is based on maximum likelihood.
    Also estimates values with a given return period.
    """
    # call generic mlik function
    return __fitGEV_mlik_gen__('mlik', data, LAMBDA=1,  ret=ret, **kwargs)

#------------------------------------------------------------------------------


def fitGEV_lmom(data, LAMBDA=1., ret=(5, 10, 50, 100, 500)):
    return __fit_gen_lmom__('GEV', data, LAMBDA, ret)

# -----------------------------------------------------------------------------


def fitGUMBEL_lmom(data, LAMBDA=1., ret=(5, 10, 50, 100, 500)):
    return __fit_gen_lmom__('GUMBEL', data, LAMBDA, ret)

# -----------------------------------------------------------------------------


def fitEXPON_lmom(data, LAMBDA=1., ret=(5, 10, 50, 100, 500), chi=False):
    return __fit_gen_lmom__('EXPON', data, LAMBDA, ret, chi)

# -----------------------------------------------------------------------------


def fitGPD_lmom(data, LAMBDA=1., ret=(5, 10, 50, 100, 500), chi=False):
    return __fit_gen_lmom__('GPD', data, LAMBDA, ret, chi)

# -----------------------------------------------------------------------------

# generic function for all l-moment fits


def __fit_gen_lmom__(dist, data, LAMBDA=1., ret=(5, 10, 50, 100, 500), chi=False):
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
        fitted = XGEV(np.asarray(f), alpha, chi, k)
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


def plot_xval(fit, ax=None, LAMBDA=1, xlim=(0.2, 400),
              kwargs_fit=dict(), kwargs_data=dict()):

    plot_xval_fit(fit, ax=ax, xlim=xlim, **kwargs_fit)
    plot_xval_data(fit, LAMBDA, ax=ax, **kwargs_data)

# -----------------------------------------------------------------------------


def plot_xval_fit(fit, ax=None, xlim=(0.2, 400), **kwargs):

    LAMBDA = fit['LAMBDA']

    ax, plot_LAMBDA, kwargs = __prepare_plot_xval__(ax, LAMBDA, **kwargs)

    params = (fit['alpha'], fit['chi'], fit['k'])

    # T_of_Y expects the log of the return period
    xlim = np.log(np.asarray(xlim))

    t_fit = np.linspace(xlim[0], xlim[1])

    y_fit = XGEV(F_of_T(T_of_Y(t_fit, plot_LAMBDA), LAMBDA), *params)

    ax.semilogx(np.e ** (t_fit), y_fit, basex=10, **kwargs)

    # ax.xaxis.set_major_formatter(ScalarFormatter())
# -----------------------------------------------------------------------------


def plot_xval_data(data, LAMBDA=1, ax=None, **kwargs):

    ax, plot_LAMBDA, kwargs = __prepare_plot_xval__(ax, LAMBDA, **kwargs)

    if type(data) is dict:
        data = data['data']

    sample_size = len(data)

    # calc plotting points
    dt_rank = rankdata(data)
    dt_freq = dt_rank / (sample_size + 1)
    dt_GUMB = Y_of_T(T_of_F(dt_freq, LAMBDA), plot_LAMBDA)
    dt_GUMB = np.e ** dt_GUMB

    ax.semilogx(dt_GUMB, data, '.', basex=10, **kwargs)

    # ax.xaxis.set_major_formatter(ScalarFormatter())

# -----------------------------------------------------------------------------


def plot_conf_xval(xval_conf, ax=None, xlim=(1.01, 1000),
                   probs=(2.5, 97.5), line=False, **kwargs):

    if line and len(probs) != 2:
        raise ValueError(
            "Need exactly two 'probs' when plotting 'fill_between'")

    LAMBDA = xval_conf['LAMBDA']
    ax, plot_LAMBDA, color = __prepare_plot_xval__(ax, LAMBDA, **kwargs)

    conf_retv = xval_conf['conf_retv']

    ret = xval_conf['ret']

    xmin = np.max([1.00001 / LAMBDA, xlim[0], np.min(ret)])
    xmax = np.min([xlim[1], np.max(ret)])

    xlim = np.asarray([xmin, xmax])

    plot_x = Y_of_T(xlim, LAMBDA=plot_LAMBDA)
    plot_x = np.linspace(*plot_x)

    xx = Y_of_T(ret, LAMBDA=plot_LAMBDA)

    av_probs = xval_conf['probs']

    IDX = []

    for i in probs:
        x = np.nonzero(i == av_probs)[0]
        if len(x):
            IDX.append(x[0])

    if not len(IDX):
        raise ValueError("desired prob not available")

    if line:
        kwargs.setdefault('lw', 0.5)
        for i in IDX:
            # y = sp.interpolate.interp1d(xx, conf_retv[i, ], kind='cubic', bounds_error=False)(plot_x)
            y = sp.interpolate.UnivariateSpline(
                xx, conf_retv[i, ], k=3)(plot_x)
            ax.semilogx(np.e ** plot_x, y, color=color, **kwargs)

    else:

        y1 = sp.interpolate.UnivariateSpline(xx, conf_retv[0, ], k=3)(plot_x)
        y2 = sp.interpolate.UnivariateSpline(xx, conf_retv[1, ], k=3)(plot_x)

        # add some options for the plot if not set already
        kwargs.setdefault('alpha', 0.15)
        kwargs.setdefault('zorder', -1000)
        kwargs.setdefault('lw', 0)
        ax.fill_between(np.e ** plot_x, y1, y2, **kwargs)
        ax.xaxis.set_major_formatter(LogFormatter(base=10))

    # ax.xaxis.set_major_formatter(ScalarFormatter())


# -----------------------------------------------------------------------------

def __prepare_plot_xval__(ax, LAMBDA, **kwargs):

    # determine axis
    if ax is None:
        ax = plt.gca()

    # set format to 1, 10, 100 instead of 10**0, 10**1, 10**2
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # add LAMBDA as attrubute to the axis
    if hasattr(ax, 'plot_LAMBDA'):
        plot_LAMBDA = ax.plot_LAMBDA
    else:
        plot_LAMBDA = LAMBDA
        ax.plot_LAMBDA = LAMBDA

    # search kwargs for 'color' or 'c'
    color = kwargs.pop('color', None)
    if color is None:
        color = 'b'

    kwargs.setdefault('c', color)

    return (ax, plot_LAMBDA, kwargs)

# -----------------------------------------------------------------------------


def set_xticks(ax=None, xlim=(0.2, 400)):
    if ax is None:
        ax = plt.gca()

    # xmin, xmax = ax.get_xlim()

    p_min = np.ceil(np.log10(xmin))
    p_max = np.floor(np.log10(xmax))


# =============================================================================

# CONFIDENCE BOUNDS CALCULATION

def conf_bounds_xval(xval, ret=(1.1, 5, 10, 50, 100, 500), probs=(2.5, 97.5),
                     size=500, cal='MLE', profs_out=False):

    if np.isnan(xval['k']):
        raise ValueError("Confidence Bounds Can not be Calculated")
    else:
        if cal == 'SIM':
            conf = __conf_bounds_xval_sim__(xval, ret, probs, size=size)
        elif cal == 'MLE':
            conf = __conf_bounds_xval_mle__(xval, ret, probs)
        elif cal == 'LPROF':
            conf = __conf_bounds_xval_lprof__(xval, ret, probs,
                                              profs_out=profs_out)
        else:
            raise KeyError('Unknown Method {cal}'.format(cal=cal))

    return conf

# ----------------------------------------------------------------------


def __conf_bounds_xval_sim__(xval, ret, probs, size):

    probs = np.asarray(probs)

    length = len(xval['data'])
    estim = xval['estim']

    alpha = xval['alpha']
    chi = xval['chi']
    k = xval['k']
    LAMBDA = xval['LAMBDA']

    ret = np.asarray(ret) / LAMBDA
    kthresh = 0.000001

    est = np.empty(shape=(size, len(ret))) * np.nan
    # for the simulation results (chi, alpha k)
    ppp = np.empty(shape=(size, 3)) * np.nan

    if estim == 'mlik':
        opt = dict(
            method=xval['method'],
            maxit=xval['maxit']
        )
    else:
        opt = dict()

    for i in xrange(size):
        sim = RGEV(alpha, chi, k, length)

        res = fitGEV(sim, LAMBDA=LAMBDA, ret=ret, dist=xval['dist'],
                     estim=xval['estim'], **opt)

        est[i, ] = res['fit']['fitted']
        ppp[i, ] = res['chi'], res['alpha'], res['k']

    conf = np.percentile(est, probs, axis=0)
    conf_paras = np.percentile(ppp, probs, axis=0)

    out = OrderedDict(
        conf_retv=conf,
        conf_paras=conf_paras,
        conf_chi=conf_paras[:, 0],
        conf_alpha=conf_paras[:, 1],
        conf_res=conf_paras[:, 2],
        paras_sim=ppp,
        ret=ret,
        probs=probs,
        LAMBDA=LAMBDA,
        est=est,
        cal='SIM'
    )
    return out


# ----------------------------------------------------------------------

def __conf_bounds_xval_mle__(xval, ret, probs):
    raise NotImplementedError()

# ----------------------------------------------------------------------


def __conf_bounds_xval_lprof__(xval, ret, probs, profs_out=False):
    raise NotImplementedError()

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

# if __name__ == '__main__':

#     zuri_hmax = np.asarray([35.3, 27.1, 12.2, 12.6, 15.8, 26.2, 20.9, 71.2, 23.2,
#      45.0, 10.7, 10.7, 32.7, 18.8, 21.2, 13.9, 29.1, 9.4, 15.9, 32.2, 22.3,
#      17.6, 23.7, 12.7, 14.7, 13.4, 21.6, 18.8, 17.8, 11.2])

#     res = fitGEV_lmom(zuri_hmax)
#     print(res)

#     res = fitGEV_mlik(zuri_hmax)
#     print(res)
#     res = fitGEV_pmlik(zuri_hmax)
#     print(res)

#     res = fitGUMBEL_mlik(zuri_hmax)
#     print(res)

#     res = fitGPD_lmom(zuri_hmax)
#     print(res)


#     print('\n\n\n')
#     res = fitEXPON_mlik(zuri_hmax)
#     print(res)

#     print('\n\n\n')
#     res = fitGPD_mlik(zuri_hmax)
#     print(res)

#     res = fitGEV_mlik(zuri_hmax)
#     plot_xval(res)
