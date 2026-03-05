import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import approx_fprime
from scipy.optimize import minimize


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
    method : Method used for plotting position. Options:
             'weibull', 'gringorten', 'hazen', 'beard'.
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

#    y_max = df[var].resample('YE').max()
#    y_idx = df[var].groupby(df.index.year).idxmax().values
#    result= pd.DataFrame({
#    'max_value': y_max,
#    'max_date': y_idx
#    })
    
    return r_max

#=====================================================================


def gev_return_ci(data, T_fit, method="bootstrap", alpha=0.05, n_boot=1000):
    """
    Compute GEV return levels and confidence intervals.

    Parameters
    ----------
    data : array-like
        Observed extremes
    T_fit : array-like
        Return periods where the curve is evaluated
    method : str
        "bootstrap", "delta", or "profile"
    alpha : float
        Significance level
    n_boot : int
        Bootstrap samples (if bootstrap method)

    Returns
    -------
    rl : return level curve
    lower : lower CI
    upper : upper CI
    params : fitted GEV parameters
    """

    data = np.asarray(data)

    # Fit GEV
    c, loc, scale = stats.genextreme.fit(data)

    P = 1 - 1/np.asarray(T_fit)

    rl = stats.genextreme.ppf(P, c, loc=loc, scale=scale)

    # --------------------------------------------------
    # 1) BOOTSTRAP
    # --------------------------------------------------

    if method == "bootstrap":

        boot_levels = np.zeros((n_boot, len(T_fit)))

        for i in range(n_boot):

            sim = stats.genextreme.rvs(c, loc=loc, scale=scale, size=len(data))
            c_b, loc_b, scale_b = stats.genextreme.fit(sim)

            boot_levels[i] = stats.genextreme.ppf(
                P, c_b, loc=loc_b, scale=scale_b
            )

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

                rl1 = stats.genextreme.ppf(p, p1[0], loc=p1[1], scale=p1[2])
                rl2 = stats.genextreme.ppf(p, p2[0], loc=p2[1], scale=p2[2])

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
            return -np.sum(stats.genextreme.logpdf(data, c, loc=loc, scale=scale))

        res = minimize(
            negloglik,
            x0=[c, loc, scale],
            method="L-BFGS-B",
            bounds=[(-1,1),(None,None),(1e-6,None)]
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

                    z_model = stats.genextreme.ppf(p, c, loc=loc, scale=scale)
                    penalty = 1e6*(z_model - z)**2

                    return -np.sum(
                        stats.genextreme.logpdf(data, c, loc=loc, scale=scale)
                    ) + penalty

                res = minimize(
                    constrained_nll,
                    x0=[c, loc, scale],
                    method="L-BFGS-B",
                    bounds=[(-1,1),(None,None),(1e-6,None)]
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
