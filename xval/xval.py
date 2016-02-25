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

import core
from core import *


def XGEV(ff, chi, alpha, k):
    """
    Quantile Function for Generalised Extreme Value Distribution

    ==========   ==========  ================================
    Argument     Type        Description
    ==========   ==========  ================================
    ff           array_like  cumulative frequencies for which 
                             to calculate the quantiles
    chi          float       location parameter
    alpha        float       scale parameter               
    k            float       shape parameter
    ==========   ==========  ================================
    """
    kthresh = 0.000001

    ff = np.asarray(ff)
    chi = np.asarray(chi)
    alpha = np.asarray(alpha)
    k = np.asarray(k)

    def gumb(f):
        return -sp.log(-sp.log(f))

    def weib(f, k):
        return (1. - sp.exp(k * sp.log(-sp.log(f)))) / k

    def gev(f, k):
        if abs(k) < kthresh:
            return alpha * gumb(f) + chi
        else:
            return alpha * weib(f, k) + chi

    return gev(ff, k)

# -----------------------------------------------------------------------------


def FGEV(x, chi, alpha, k):
    """
    cumulative pdf for Generalised Extreme Value Distribution


    Parameters
    ----------
    x : float
        quantile
    chi : float
        location parameter
    alpha : float
        scale parameter
    k : float
        shape parameter


    Notes
    ----- 
    In the notation of Coles (2001) the present (chi, alpha, k) 
    correspond to (mu,sigma,-chi)

    kthresh defines a threshold for k. if abs(k)<kthresh then the
    Gumbel Distribution is taken.

    References
    ----------
    .. [1] e.g. Zwiers and Kharin 1998
    """

    kthresh = 0.000001
    x = np.asarray(x)
    ar = (x - chi)/alpha

    # Gumbel Distribution
    if abs(k) <= kthresh:
        res = exp(-exp(-ar))
    
    # Weibull Distribution
    elif k > kthresh:
        res = np.ones_like(ar)
        sel = ar < 1./k
        res[sel] = exp(-(1-k*ar[sel])**(1./k))
    
    # Frechet Distribution
    elif k < kthresh:
        res = np.zeros_like(ar)
        sel = ar > 1./k
        res[sel] = exp(-(1-k*ar[sel])**(1./k))

    return res



# -----------------------------------------------------------------------------

def RGEV(alpha, chi, k, n):
    """
    Random distributed number with Generalised Extreme Value Distribution


    Parameters
    ----------
    alpha : float
        scale parameter
    chi : float
        location parameter
    k : float
        shape parameter
    n : integer
        number of samples

    Returns
    -------
    r : ndarray
        array containg random samples of the GEV    

    References
    ----------
    .. [1] e.g. Zwiers and Kharin 1998

    """

    kthresh = 0.000001
    rr = np.random.uniform(size=n)
    return XGEV(rr, chi, alpha, k)


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

# use currying to use the function for the optimization and hessian matrix calc


def gev_lik(data):
    """
    returns negative log likelihood of GEV distribution

    """
    # computes neg log lik of gev model including gumbel limit
    kthresh = 0.000001     # threshold to distinguish GUMBEL
    length = len(data)     # sample size
    
    def gev_lik_internal(params):
        chi, alpha, k = params
        y = (data - chi) / alpha
        if abs(k) > kthresh:
            y = (1 - k * y)
            if np.any(y <= 0) or alpha <= 0:
                return np.inf
            return length * sp.log(alpha) + np.sum(y ** (1 / k)) + np.sum(sp.log(y) * (1 - 1 / k))
        else:
            return length * sp.log(alpha) + np.sum(y) + np.sum(sp.exp(-y))
    return gev_lik_internal

# -----------------------------------------------------------------------------
# COVARIATES

def get_cov(length, cov):
    ones = np.ones(shape=(length, 1))
    if cov is None:
        return ones
    else:
        if cov.ndim == 1:
            cov = np.atleast_2d(cov).T
        cov = np.hstack((ones, cov))
        return cov

# -----------------------------------------------------------------------------


def def_indices(n_chi, n_alpha, n_k):
    i_chi = [0, ] + range(3, 2 + n_chi)
    start = max(i_chi[-1], 2)
    i_alpha = [1, ] + range(start + 1, start + n_alpha)

    start = max(i_chi[-1], i_alpha[-1], 2)
    i_k = [2, ] + range(start + 1, start + n_k)

    return i_chi, i_alpha, i_k


# -----------------------------------------------------------------------------

def gev_cov_lik(data, cov_chi=None, cov_alpha=None, cov_k=None):
    length = len(data)     # sample size

    cov_chi = get_cov(length, cov_chi)
    cov_alpha = get_cov(length, cov_alpha)
    cov_k = get_cov(length, cov_k)

    i_chi, i_alpha, i_k = def_indices(cov_chi.shape[1], cov_alpha.shape[1], cov_k.shape[1])

    kthresh = 0.000001     # threshold to distinguish GUMBEL

   
    def gev_lik_internal(params):

        params = np.asarray(params)
        # calculate specific params for every data point
        chi = np.dot(cov_chi, params[i_chi])
        alpha = np.dot(cov_alpha, params[i_alpha])
        k = np.dot(cov_k, params[i_k])

        # computes neg log lik of gev model including gumbel limit
        y = (data - chi) / alpha

        sel = abs(k) > kthresh

        y[sel] = (1 - k[sel] * y[sel])
        if np.any(y[sel] <= 0) or np.any(alpha[sel] <= 0):
            return np.inf
        
        # FRECHET OR WEIBULL
        mlik = np.sum(sp.log(alpha[sel]) + y[sel] ** (1 / k[sel])) + np.sum(sp.log(y[sel]) * (1 - 1 / k[sel]))
        # GUMBEL
        mlik += np.sum(sp.log(alpha[~sel])) + np.sum(y[~sel]) + np.sum(sp.exp(-y[~sel]))
    
        return np.asarray(mlik)
    return gev_lik_internal

#------------------------------------------------------------------------------

# Geophysical prior function PPI(kk) from Martins and Stedinger p. 740


def ppi(b):

    # parameters for geophys prior from Martins and Stedinger p. 740
    if abs(b) < 0.5:
        pp, qq = 6, 9
        return ((0.5 + b) ** (pp - 1) * (0.5 - b) ** (qq - 1)) / beta(pp, qq)
    else:
        return 0


#------------------------------------------------------------------------------

def gev_plik(data):
    def gev_lik_internal(params):
        # computes neg log lik of gev model including gumbel limit
        chi, alpha, k = params
        kthresh = 0.000001     # threshold to distinguish GUMBEL
        length = len(data)     # sample size

        y = (data - chi) / alpha
        prior = ppi(k)

        if abs(k) > kthresh:
            y = (1 - k * y)
            if np.any(y <= 0) or alpha <= 0 or prior == 0:
                return np.inf
            return length * sp.log(alpha) + np.sum(y ** (1 / k)) + np.sum(sp.log(y) * (1 - 1 / k)) - np.log(prior)
        else:
            if alpha <= 0 or prior == 0:
                return np.inf
            return length * sp.log(alpha) + np.sum(y) + np.sum(sp.exp(-y))
    return gev_lik_internal

#------------------------------------------------------------------------------

def gumbel_lik(data):
    def gumbel_lik_internal(params):
        params = np.asarray([params[0], params[1], 0])
        return gev_lik(data)(params)
    return gumbel_lik_internal

#------------------------------------------------------------------------------


def estim(func, init=(32, 0.1, 0), label=None):
    opt_minim = dict(options={'xtol': 1e-10, 
                     'disp': False, 'maxiter' : 10000, 'maxfev' : 10000})
    
    res = minim(func, init, **opt_minim)
    
    if label is not None:
        print(label)

    if res.success:
        print('  x: {0}'.format(res.x))
        print('fun: {0}'.format(res.fun))
        print('covmat:\n{0}'.format(np.diag(res.covmat)))

        # print('conf_int:\n{0}'.format(res.conf_int))

        print('std95:\n{0}'.format(res.std95))


    else:
        print(res)

    print('AIC: {0}'.format(2*res.fun+2*len(init)))
    print('~~~~~~~~~~~~~~~~~~~~')

    return tuple(res.x), res

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
        fitted = XGEV(np.asarray(f), chi, alpha, k)
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
        fitted = XGEV(np.asarray(f), chi, alpha, k)
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
    return core.__fit_gen_lmom__('GEV', data, LAMBDA, ret)

# -----------------------------------------------------------------------------


def fitGUMBEL_lmom(data, LAMBDA=1., ret=(5, 10, 50, 100, 500)):
    return core.__fit_gen_lmom__('GUMBEL', data, LAMBDA, ret)

# =============================================================================


def plot_xval(fit, ax=None, LAMBDA=1, xlim=(0.2, 400),
              kwargs_fit=dict(), kwargs_data=dict()):

    plot_xval_fit(fit, ax=ax, xlim=xlim, **kwargs_fit)
    plot_xval_data(fit, LAMBDA, ax=ax, **kwargs_data)

# -----------------------------------------------------------------------------


def plot_xval_fit(fit, ax=None, xlim=(0.2, 400), **kwargs):
    """
    plot GEV given the parameters

    Parameters
    ----------
    fit : dict
        fitted object with fitGEV
    ax : a matplotlib axes object or None
       If no ax is given, uses plt.gca() to create a new one.
    LAMBDA : float
           transformation
    xlim : two-element list
        defines the x limit of the plotted line
    kwargs : named arguments
        passed to the semilogx plotting function

    Returns
    -------
    line : matplotlib.lines.Line2D object


    """

    LAMBDA = fit['LAMBDA']
    params = (fit['chi'], fit['alpha'], fit['k'])

    return plot_xval_params(params, ax=ax, LAMBDA=LAMBDA, xlim=xlim, **kwargs)
    
# -----------------------------------------------------------------------------

def plot_xval_params(params, ax=None, LAMBDA=1, xlim=(0.2, 400), **kwargs):
    """
    plot GEV given the parameters

    Parameters
    ----------
    params : array_like
           The three parameters of a GEV: alpha, chi, k
    ax : a matplotlib axes object or None
       If no ax is given, uses plt.gca() to create a new one.
    LAMBDA : float
           transformation
    xlim : two-element list
        defines the x limit of the plotted line
    kwargs : named arguments
        passed to the semilogx plotting function

    Returns
    -------
    line : matplotlib.lines.Line2D object


    """

    ax, plot_LAMBDA, kwargs = __prepare_plot_xval__(ax, LAMBDA, **kwargs)

    # T_of_Y expects the log of the return period
    xlim = np.log(np.asarray(xlim))

    t_fit = np.linspace(xlim[0], xlim[1])

    y_fit = XGEV(F_of_T(T_of_Y(t_fit, plot_LAMBDA), LAMBDA), *params)

    return ax.semilogx(np.e ** (t_fit), y_fit, basex=10, **kwargs)

    # ax.xaxis.set_major_formatter(ScalarFormatter())
# -----------------------------------------------------------------------------

def plot_xval_data(data, LAMBDA=1, ax=None, **kwargs):
    """
    Plot Gumbel Diagramm with empirical return times

    """
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

def plot_conf_sampled_xval(params, ax=None, LAMBDA=1, xlim=(1.01, 1000),
                   probs=(2.5, 97.5), line=False, **kwargs):
    """
    plot uncertainty of GEV given the sampled parameters

    Parameters
    ----------
    params : array_like
           The three parameters of a GEV: alpha, chi, k
    ax : a matplotlib axes object or None
       If no ax is given, uses plt.gca() to create a new one.
    xlim : two-element list
        defines the x limit of the plotted line
    probs : tuple
        probability for which to plot the uncertainty bounds
    line : bool
        if true uses line else fill_between
    kwargs : named arguments
        passed to the plot/ fill_between plotting function

    Returns
    -------
    handle : handle of the plot/ fill_between object


    """

    if not line and len(probs) != 2:
        msg = "Need exactly two 'probs' when plotting 'fill_between'"
        raise ValueError(msg)

    ax, plot_LAMBDA, kwargs = __prepare_plot_xval__(ax, LAMBDA, **kwargs) 


    xmin = np.amax([1.00001 / LAMBDA, xlim[0]])
    xmax = xlim[1]

    xlim = np.log(np.asarray([xmin, xmax]))

    t_fit = np.linspace(xlim[0], xlim[1])

    n_samples = params.shape[0]
    n_plot_points = t_fit.shape[0]

    sampled = np.empty(shape=(n_samples, n_plot_points))
    
    ft = F_of_T(T_of_Y(t_fit, plot_LAMBDA), LAMBDA)
    for i, param in enumerate(params):
        y_fit = XGEV(ft, *param)

        sampled[i, :] = y_fit

    y = np.percentile(sampled, probs, axis=0)

    if line:
        kwargs.setdefault('lw', 0.5)
        y = y.transpose()
        return ax.semilogx(np.e ** t_fit, y, basex=10, **kwargs)
    else:
        y1 = y[0, :]
        y2 = y[1, :]

        # add some options for the plot if not set already
        kwargs.setdefault('alpha', 0.15)
        kwargs.setdefault('zorder', -1000)
        kwargs.setdefault('lw', 0)
        ax.set_xscale('log', basex=10)
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        return ax.fill_between(np.e ** t_fit, y1, y2, **kwargs)

# -----------------------------------------------------------------------------


def plot_conf_xval(xval_conf, ax=None, xlim=(1.01, 1000),
                   probs=(2.5, 97.5), line=False, **kwargs):

    if not line and len(probs) != 2:
        raise ValueError(
            "Need exactly two 'probs' when plotting 'fill_between'")

    LAMBDA = xval_conf['LAMBDA']
    ax, plot_LAMBDA, kwargs = __prepare_plot_xval__(ax, LAMBDA, **kwargs)

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
            ax.semilogx(np.e ** plot_x, y, **kwargs)

    else:

        y1 = sp.interpolate.UnivariateSpline(xx, conf_retv[0, ], k=3)(plot_x)
        y2 = sp.interpolate.UnivariateSpline(xx, conf_retv[1, ], k=3)(plot_x)

        # add some options for the plot if not set already
        kwargs.setdefault('alpha', 0.15)
        kwargs.setdefault('zorder', -1000)
        kwargs.setdefault('lw', 0)
        ax.fill_between(np.e ** plot_x, y1, y2, **kwargs)
        ax.xaxis.set_major_formatter(LogFormatterMathtext())

    # ax.xaxis.set_major_formatter(ScalarFormatter())

# -----------------------------------------------------------------------------

def qq_xval(data, fit, ax=None, **kwargs):
    """
    Q-Q plot of a sample for the Generalized Extreme Value Distribution.
 
    This function uses a fitted GEV as input

    Parameters
    ----------
    data : array_like
         the data to plot
    fit : dict
        Fitted GEV object.
    ax : Matplotlib AxesSubplot instance, optional
       If given, this subplot is used to plot in instead of using 
       plt.gca()
    kwargs : dict
           is passed to the plot function

    Returns 
    -------
    line : matplotlib.lines.Line2D object

    See Also
    --------
    qq_xval

    """

    chi, alpha, k = fit['chi'], fit['alpha'], fit['k']
    return qq_xval_params(data, chi, alpha, k, ax=None, **kwargs)

# -----------------------------------------------------------------------------

def qq_xval_params(data, chi, alpha, k, ax=None, **kwargs):
    """
    Q-Q plot of a sample for the Generalized Extreme Value Distribution.
 
    This function uses the parameters as input

    Parameters
    ----------
    data : array_like
         the data to plot
    chi : float
        location parameter
    alpha : float
        scale parameter
    k : float
        shape parameter
    ax : Matplotlib AxesSubplot instance, optional
       If given, this subplot is used to plot in instead of using 
       plt.gca()
    **kwargs : dict
             is passed to the plot function

    Returns 
    -------
    line : matplotlib.lines.Line2D object

    See Also
    --------
    qq_xval

    """    


    if ax is None:  
        ax = plt.gca()

    data = np.asarray(data)

    q_empir = GEV_standardize(data, chi, alpha, k)
    q_empir = np.sort(q_empir)

    n = len(data)
    q_theor = -log(-log(np.arange(1, n+1)/(n+1)))

    # pp = __ppoints__(data)
    # IDX = np.argsort(data)
    # q_empir = data[IDX]

    # pp = pp
    
    # par = tuple()
    # for i, p in enumerate(params):
    #     if not np.isscalar(p):
    #         par += (p[IDX], )
    #     else:
    #         par += (p, )

    # q_theor = XGEV(pp, *par)


    ax.set_title("GEV Q-Q Plot")
    ax.set_xlabel("Theoretical Quantile")
    ax.set_ylabel("Empirical Quantile")
    
    # make sure points are above 1:1 line
    kwargs['zorder'] = kwargs.pop('zorder', 3)

    h = ax.plot(q_theor, q_empir, '.', **kwargs)

    ax.set_aspect('equal', adjustable='box')

    # 1 on 1 line
    xlim = ax.get_xlim()

    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'k')

    # display r^2
    r, __ = pearsonr(q_theor, q_empir)
    posx = 0.95 # xlim[0] + 0.95 * (xlim[1] - xlim[0])
    posy = 0.02 # xlim[0] + 0.01 * (xlim[1] - xlim[0])
    # ax.text(posx, posy, "$R^2=%1.4f$" % r**2, ha='right', transform=ax.transAxes)


    return h 

# -----------------------------------------------------------------------------

def return_time(value, chi, alpha, k):
    """
    return time of a value for a given parameter set

    Parameters
    ----------
    value : ndarray
        value to calculate the return time for
    chi : float
        location parameter
    alpha : float
        scale parameter
    k : float
        shape parameter

    Returns
    -------
    return_time : ndarray
        return time of value

    """
    cdf = FGEV(value, chi, alpha, k)
    return 1./(1. - cdf)

# -----------------------------------------------------------------------------


def GEV_standardize(data, chi, alpha, k):
    """
    normalize maxima for qq and probability plot

    Parameters
    ----------
    data : array_like
         data to normalize (data where GEV was fitted)
    chi : float
        location parameter
    alpha : float
        scale parameter
    k : float
        shape parameter

    Returns
    -------
    zt : ndarray
       normalized data

    Notes
    -----
    To get normalized data with covariates: pass chi, alpha, k as vector.

    References
    ----------
    .. [1] Coles p. 110
    """

    return 1./(-k)*log(1 - k*((data - chi)/alpha))


# -----------------------------------------------------------------------------

def __ppoints__(n):
    """
    sequence of probability points

    Generates the sequence of probability points

    ..math:: (1:m - a)/(m + (1-a)-a),

    where m is either n, if len(n)==1, or len(n).
    

    Parameters
    ----------
    n : integer or array_like
        defines the number of points that are returned.


    Returns
    -------
    pp : 


    Notes
    ----- 
    In the notation of Coles (2001) the present (chi, alpha, k) 
    correspond to (mu,sigma,-chi)

    kthresh defines a threshold for k. if abs(k)<kthresh then the
    Gumbel Distribution is taken.

    References
    ----------
    .. [1] Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988) 
           The New S Language. Wadsworth & Brooks/Cole.
    .. [2] Blom, G. (1958) Statistical Estimates and Transformed Beta
           Variables. Wiley

    """
    if len(n) > 1:
        n = len(n)

    if n <= 10:
        a = 3./8.
    else:
        a = 1./2.


    pp = (np.arange(1, n + 1) - a)/(n + 1 - 2*a)

    return pp


# -----------------------------------------------------------------------------

def __prepare_plot_xval__(ax, LAMBDA, **kwargs):
    """
    parse plot arguments and ready the axis
    """
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
    color = kwargs.pop('c', 'b')
    kwargs.setdefault('color', color)

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

    dist = xval['dist']

    alpha = xval['alpha']
    chi = xval['chi']
    k = xval['k']
    LAMBDA = xval['LAMBDA']


    F_of_T(ret, LAMBDA)


    raise NotImplementedError()

# ----------------------------------------------------------------------


def __conf_bounds_xval_lprof__(xval, ret, probs, profs_out=False):
    raise NotImplementedError()



# if __name__ == '__main__':

     # zuri_hmax = np.asarray([35.3, 27.1, 12.2, 12.6, 15.8, 26.2, 20.9, 71.2, 23.2,
     #  45.0, 10.7, 10.7, 32.7, 18.8, 21.2, 13.9, 29.1, 9.4, 15.9, 32.2, 22.3,
     #  17.6, 23.7, 12.7, 14.7, 13.4, 21.6, 18.8, 17.8, 11.2])

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
