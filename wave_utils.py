import numpy as np










def wvlngth(Lt,dd,T):
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
