import numpy as np
import pandas as pd

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
            df[var].interpolate(method='time',inplace=True,limit=12)
#            df.method({var: time}, inplace=True)
#            df[var] = df[var].method(time)

    if iu==2:
        df = pd.read_table(fp, header=None,sep ='\s+', parse_dates=[[0,1,2,3]],index_col=0)
        df.index.rename('YYYY-MM-DD hh:mm:ss',inplace=True)
        df.columns=['Hs','Tm','Tp','Dirm','Dirp','Spr','Lm','Lp','uw','vw','Hsws','Tmws','Dirws','s1Hs','s1Tm','s1Dir','s2Hs','s2Tm','s2Dir']
        for var in ('Tm','Tp'):
            df.loc[df[var]<=0,var]=np.nan
            df[var].interpolate(method='time',inplace=True,limit=12)
    return readWave

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

