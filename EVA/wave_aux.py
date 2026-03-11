from math import dist

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import approx_fprime
from scipy.optimize import minimize
from joblib import Parallel, delayed

#=====================================================================

def readWave(iu,fp,sr):
    """
    readWave(iu,fp,sr) - read wave data from input file 
    iu - data source: 1) DICCA hd full
                      2) DICCA hd peaks
    fp - input file
    sr - skip rows (0/...) 
    """
    if iu==1:
        df = pd.read_table(fp, header=None,sep ='\s+', skiprows=sr)
        df.index = pd.to_datetime(df[0] + " " + df[1])
        df = df.drop([0, 1], axis=1)
        df.index.rename('YYYY-MM-DD hh:mm:ss',inplace=True)
        cnames=['Hs','T02','T01','Tm10','Tp','Dirm','Dirp','Spr','Hs0','Tp0','Dirp0','gam0','spr0','Hs1','Tp1','Dirp1','gam1','spr1','Hs2','Tp2','Dirp2','gam2','spr2','Hs3','Tp3','Dirp3','gam3','spr3','uw','vw']
        df.columns=cnames
        for var in ('T02','T01','Tm10','Tp'):
            df.loc[df[var]<=0,var]=np.nan
            df[var].interpolate(method='time',limit=12)

    if iu==2:
        df = pd.read_table(fp, header=None,sep ='\s+', parse_dates=[[0,1,2,3]],index_col=0)
        df.index.rename('YYYY-MM-DD hh:mm:ss',inplace=True)
        df.columns=['Hs','Tm','Tp','Dirm','Dirp','Spr','Lm','Lp','uw','vw','Hsws','Tmws','Dirws','s1Hs','s1Tm','s1Dir','s2Hs','s2Tm','s2Dir']
        for var in ('Tm','Tp'):
            df.loc[df[var]<=0,var]=np.nan
            df[var].interpolate(method='time',limit=12)
    return df

#=====================================================================

def wvlngth(Lt,dd,T):
    """
    wvlngth(Lt,dd,T) - Calculate wavelength with dispersion relationship
    Lt - trial wavelength
    dd - local depth
    T  - wave period
    """
    # TODO - add current
    
    eps=0.000000000001
    kn=2.0*np.pi/Lt
    k = kn*0.95
    kd=k*dd
    s=(2.0*np.pi/T)**2.0
    # Newton-Rapson Iteration
    while ( abs(kn-k)/kn > eps ):
        k = kn
        kd = min(k*dd,250)
        ff  = s - 9.81*k*np.tanh(kd)
        ffp = -9.81*np.tanh(kd) - 9.81*kd/(np.cosh(kd)*np.cosh(kd))
        kn = k - ff/ffp
        
    L = 2.0*np.pi/k
    return L

#=====================================================================

def plot_pos(df,var,method="weibull"):
    """
    Compute plotting positions for a variable in a dataframe.

    Parameters
    ----------
    df     : pandas.DataFrame 
    var    : Column name for which plotting positions are computed
    method : str, optional
        Method for plotting positions. Options include:
        - "weibull": r/(n+1)
        - "gringorten": (r-0.44)/(n+0.12)
        - "hazen": (r-0.5)/n
        - "beard": (r-0.31)/(n+0.38)
        Default is "weibull"    
    Returns
    -------
    pp     : Plotting position values
    """

    dfs = df.sort_values(by=var, ascending=True)
    n=len(dfs)
    rank= dfs[var].rank(method='first')

    formulas = {
        "weibull": lambda r, n: r / (n + 1),
        "gringorten": lambda r, n: (r - 0.44) / (n + 0.12),
        "hazen": lambda r, n: (r - 0.5) / n,
        "beard": lambda r, n: (r - 0.31) / (n + 0.38),
    }

    if method.lower() not in formulas:
        raise ValueError("method must be one of: 'weibull', 'gringorten', 'hazen', 'beard'")

    pp = formulas[method.lower()](rank, n)

    return pp

#=====================================================================


def am(df,var):
    """
    am(df,var) - find annual maxima for variable var
    df - pandas dataframe
    var - variable to find maxima
    T  - wave period
    """

    r_max = df.loc[df[var].groupby(df.index.year).idxmax()]
    
    return r_max

#=====================================================================

def return_ci(data, T_fit, dist="gev",method="bootstrap", alpha=0.05, n_boot=1000, n_jobs=-1,lam=1, trld=None):
    """
    Compute GEV return levels and confidence intervals.

    Parameters
    ----------
    data : array-like
        Observed extremes
    T_fit : array-like
        Return periods where the curve is evaluated
    dist: "gev" or "gpd"    
    method : str
        "bootstrap", "delta", or "profile"
    alpha : float
        Significance level
    n_boot : int
        Bootstrap samples (if bootstrap method)
    n_jobs : int
        Number of parallel jobs for bootstrap (default=-1 uses all CPUs)
        Set to 1 to disable parallelization
    lam : float
        Average number of events per year (for GPD return level calculation)
    trld : float
        Threshold for GPD fitting (if dist="gpd")

    Returns
    -------
    rl : return level curve
    lower : lower CI
    upper : upper CI
    params : fitted GEV parameters
    """

    data = np.asarray(data)
    P = 1 - 1/(np.asarray(T_fit)*lam)

    if dist=="gev":   # Fit GEV
        c, loc, scale = stats.genextreme.fit(data)
        rl = stats.genextreme.ppf(P, c, loc=loc, scale=scale)
    elif dist=="gpd": # Fit GPD
        u=trld if trld is not None else 0.0
        c, loc, scale, _ = gpd_fit(data, u, method="l-mom")
        print(c,loc,scale)
        rl = stats.genpareto.ppf(P, c, loc=loc, scale=scale)

    # --------------------------------------------------
    # 1) BOOTSTRAP (Parallelized)
    # --------------------------------------------------

    if method == "bootstrap":

        def bootstrap_iteration(seed,cdfP):
            """Single bootstrap iteration"""
            np.random.seed(seed)
            if cdfP=="gev":   # Fit GEV
                sim = stats.genextreme.rvs(c, loc=loc, scale=scale, size=len(data))
                c_b, loc_b, scale_b = stats.genextreme.fit(sim)
                return stats.genextreme.ppf(P, c_b, loc=loc_b, scale=scale_b)
            elif cdfP=="gpd": # Fit GPD
                sim = stats.genpareto.rvs(c, loc=loc, scale=scale, size=len(data))
                c_b, loc_b, scale_b, _ = gpd_fit(sim, u, method="l-mom")
                return stats.genpareto.ppf(P, c_b, loc=loc_b, scale=scale_b)

        # Parallel execution
        boot_levels = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(bootstrap_iteration)(i, dist) for i in range(n_boot)
                                                             )
        boot_levels = np.array(boot_levels)

        lower = np.percentile(boot_levels, 100*alpha/2, axis=0)
        upper = np.percentile(boot_levels, 100*(1-alpha/2), axis=0)

    # --------------------------------------------------
    # 2) DELTA METHOD
    # --------------------------------------------------

    elif method == "delta":

        params = np.array([c, loc, scale])
        z = stats.norm.ppf(1 - alpha/2)

        # crude covariance estimate
        cov = np.eye(3) * np.var(data)/len(data)

        lower = np.zeros_like(rl)
        upper = np.zeros_like(rl)

        for i,p in enumerate(P):

            grad = np.zeros(3)
            eps = 1e-6

            for j in range(3):

                p1 = params.copy()
                p2 = params.copy()

                p1[j] += eps
                p2[j] -= eps

                if dist=="gev":   # Fit GEV
                    rl1 = stats.genextreme.ppf(p, p1[0], loc=p1[1], scale=p1[2])
                elif dist=="gpd": # Fit GPD
                    rl1 = stats.genpareto.ppf(p, p1[0], loc=p1[1], scale=p1[2])

                if dist=="gev":   # Fit GEV
                    rl2 = stats.genextreme.ppf(p, p2[0], loc=p2[1], scale=p2[2])
                elif dist=="gpd": # Fit GPD
                    rl2 = stats.genpareto.ppf(p, p2[0], loc=p2[1], scale=p2[2])

                grad[j] = (rl1 - rl2)/(2*eps)

            var_rl = grad @ cov @ grad
            se = np.sqrt(var_rl)

            lower[i] = rl[i] - z*se
            upper[i] = rl[i] + z*se

    # --------------------------------------------------
    # 3) PROFILE LIKELIHOOD
    # --------------------------------------------------

    elif method == "profile":

        def negloglik(params):
            c, loc, scale = params
            if scale <= 0:
                return np.inf
            if dist=="gev":
                return -np.sum(stats.genextreme.logpdf(data, c, loc=loc, scale=scale))
            elif dist=="gpd":
                return -np.sum(stats.genpareto.logpdf(data, c, loc=loc, scale=scale))

        if dist=="gev":
            bounds=[(-1,1),(None,None),(1e-6,None)]
        elif dist=="gpd":
            bounds=[(-1,1),(None,None),(1e-6,None)]
            
        res = minimize(
            negloglik,
            x0=[c, loc, scale],
            method="L-BFGS-B",
            bounds=bounds
        )

        ll_max = -res.fun
        chi = stats.chi2.ppf(1-alpha, df=1)

        lower = np.zeros_like(rl)
        upper = np.zeros_like(rl)

        for i,p in enumerate(P):

            z_hat = rl[i]

            grid = np.linspace(0.5*z_hat, 1.5*z_hat, 80)
            ll_vals = []

            for z in grid:

                def constrained_nll(params):
                    c, loc, scale = params
                    if scale <= 0:
                        return np.inf

                    if dist=="gev":
                        z_model = stats.genextreme.ppf(p, c, loc=loc, scale=scale)
                    elif dist=="gpd":
                        z_model = stats.genpareto.ppf(p, c, loc=loc, scale=scale)
                        
                    penalty = 1e6*(z_model - z)**2

                    if dist=="gev":
                        return -np.sum(
                            stats.genextreme.logpdf(data, c, loc=loc, scale=scale)
                        ) + penalty
                    elif dist=="gpd":
                        return -np.sum(
                            stats.genpareto.logpdf(data, c, loc=loc, scale=scale)
                        ) + penalty

                res = minimize(
                    constrained_nll,
                    x0=[c, loc, scale],
                    method="L-BFGS-B",
                    bounds=bounds
                )

                ll_vals.append(-res.fun)

            ll_vals = np.array(ll_vals)

            mask = 2*(ll_max - ll_vals) <= chi

            if np.any(mask):
                lower[i] = grid[mask].min()
                upper[i] = grid[mask].max()
            else:
                lower[i] = np.nan
                upper[i] = np.nan

    else:
        raise ValueError("method must be 'bootstrap', 'delta', or 'profile'")

    return rl, lower, upper, (c, loc, scale)

#=====================================================================

# Function to calculate return period from a given value
def value_to_return_period(value, c, loc, scale):
    """
    Calculate return period for a given value using fitted GEV parameters
    
    Parameters:
    value: the value of the variable (e.g., Hs in meters)
    c, loc, scale: GEV distribution parameters
    
    Returns:
    return_period: return period in years
    """
    # Calculate probability of non-exceedance using GEV CDF
    P = stats.genextreme.cdf(value, c, loc=loc, scale=scale)
    
    # Convert to return period: T = 1/(1-P)
    return_period = 1 / (1 - P)
    
    return return_period

#=====================================================================

 # Calculate autocorrelation function
def calculate_acf(data, nlags=200, n_jobs=-1):
    """
    Calculate autocorrelation function (ACF) using parallel processing.
    
    Parameters
    ----------
    data : array-like
        Input data array
    nlags : int, default=200
        Number of lags to compute
    n_jobs : int, default=-1
        Number of jobs for parallel processing. -1 means using all processors.
    
    Returns
    -------
    acf_values : np.ndarray
        Autocorrelation values for lags 0 to nlags
    """
    mmn = np.mean(data)
    c0 = np.sum((data - mmn) ** 2) / len(data)
    
    # Define function to compute ACF for a single lag
    def compute_lag(k, data, mmn, c0):
        c_k = np.sum((data[:-k] - mmn) * (data[k:] - mmn)) / len(data)
        return c_k / c0
    
    # Parallelize lag computation
    acf_lags = Parallel(n_jobs=n_jobs)(
        delayed(compute_lag)(k, data, mmn, c0) for k in range(1, nlags + 1)
    )
    
    # Combine results with ACF at lag 0
    acf_values = np.concatenate([[1.0], acf_lags])
    
    return acf_values

#=====================================================================

# Function to compute Probability Weighted Moments (PWM) for GPD - Parallelized
def compute_pwm(data, r, n_jobs=-1):
    """
    Compute Probability Weighted Moments for GPD using parallel processing.
    
    Parameters
    ----------
    data : array-like
        Input data array
    r : int
        Order of the PWM (0, 1, 2, ...)
    n_jobs : int, default=-1
        Number of jobs for parallel processing. -1 means using all processors.
    
    Returns
    -------
    pwm : float
        Probability weighted moment of order r
    """
    n = len(data)
    x = np.sort(data)
    
    # Define function to compute weight and contribution for single index
    def compute_contribution(j, x, n, r):
        weight = 1.0
        for k in range(r):
            if n - k - 1 > 0:
                weight *= (j - k) / (n - k - 1)
        return x[j] * weight / n
    
    # Parallelize computation across all j indices
    contributions = Parallel(n_jobs=n_jobs)(
        delayed(compute_contribution)(j, x, n, r) for j in range(n)
    )
    
    pwm = np.sum(contributions)
    return pwm

#=====================================================================

# Function to fit GPD using L-moments or MLE
def gpd_fit(data, u, method="l-mom"):
    """
    Fit Generalized Pareto Distribution (GPD) to data above a specified threshold.
    
    Parameters
    ----------
    data : array-like
        Input data array
    u : float
        Threshold value for exceedances
    method : str, default="l-mom"
        Fitting method for GPD parameters. 
        Options include "l-mom" for L-moments and "mle" for maximum likelihood estimation.
    
    Returns
    -------
    c_gp : float
        Shape parameter of the fitted GPD
    loc_gp : float
        Location parameter of the fitted GPD (equal to threshold u)
    scale_gp : float
        Scale parameter of the fitted GPD   
    params : tuple
        Fitted GPD parameters (shape, loc, scale)
    """
    exceedances = data - u
    exceedances = exceedances[exceedances >= 0]
    
    # Fit GPD to exceedances
    if method == "l-mom":
        # Implementation for L-moments fitting
        if len(exceedances) < 3:
            raise ValueError('Need at least 3 exceedances for L-moment GPD fitting.')   
        # Compute sample L-moments following Hosking & Wallis (1997)
        # L_r = E[X * Pr(r-1, n-r)], where Pr are shifted Legendre polynomials
        x = np.sort(exceedances)
        n = len(x)
        # Compute PWMs
        b0 = compute_pwm(x, 0)  # alpha
        b1 = compute_pwm(x, 1)  # beta
        b2 = compute_pwm(x, 2)  # gamma

        # Convert PWMs to L-moments
        # L1 = b0
        # L2 = 2*b1 - b0
        # L3 = 6*b2 - 6*b1 + b0
        l1 = b0
        l2 = 2.0 * b1 - b0
        l3 = 6.0 * b2 - 6.0 * b1 + b0

        if l1 <= 0 or l2 <= 0:
            raise ValueError(f'Invalid sample L-moments: l1={l1:.4f}, l2={l2:.4f}. Both must be > 0.')

        # L-moment ratios
        t2 = l2 / l1  # L-CV
        t3 = l3 / l2  # L-skewness

        if u==0:
            # Working without threshold correction for L-moments, as per Hosking & Wallis (1997)
            c_gp = (1.0-3.0*t3)/(1.0+t3)         # shape parameter
            scale_gp = (1.0+c_gp)*(2.0+c_gp)*l2  # scale parameter
            loc_gp = l1-(2.0+c_gp)*l2            # location (threshold)
            c_gp=-c_gp

        else:   
            # Fixing the threshold, using only exceedances for L-moments
            loc_gp = u    # location (threshold)
            c_gp = 2.0 - 1.0 / t2  # shape parameter (xi)
            scale_gp = l1 * (1.0 - c_gp)  # scale parameter (sigma)
        
        if scale_gp <= 0:
            raise ValueError(f'Computed non-positive GPD scale: {scale_gp:.6f}')

        if c_gp <= -0.5:
            print(f"Warning: shape parameter {c_gp:.4f} is outside typical range for GPD")
        params=(b0, b1, b2, l1, l2, l3, t2, t3)

    elif method == "mle":
        if len(exceedances) < 3:
            raise ValueError('Need at least 3 exceedances for GPD fitting.')
                # Fit on exceedances with location fixed at 0
        c_gp, loc_exc, scale_gp = stats.genpareto.fit(exceedances, floc=u, method="MLE")
        loc_gp = u
        params=0
    else:
        raise ValueError("method must be 'l-mom' or 'mle'")



    return c_gp, loc_gp, scale_gp, params

#=====================================================================












































