import numpy as np
import matplotlib.pyplot as plt

ni=0.00001
g=9.81

rhos=2650
rhow=1000
ss=rhos/rhow

d=np.linspace(0.001, 400, 1000, endpoint=True)/1000
D=(g*(ss-1)/ni**2)**(1/3)*d
Rp=np.sqrt((ss-1)*g*d**3)/ni

# Fall velocity Soulsby (1987)
ws_soulsby=ni/d*((10.36**2+1.049*D**3)**0.5-10.36)

# Fall velocity Hallermeier (1981)
dv=np.where(D**3<39)
ws1=ni*D[dv]**3/(18*d[dv])
dv=np.where((D**3>39) & (D**3<10000))
ws2=ni*D[dv]**2.1/(6*d[dv])
#dv=np.where((D**3>10000) & (D**3<300000000))
dv=np.where(D**3>10000)
ws3=1.05*ni*D[dv]**1.5/d[dv]
ws_haller=np.concatenate((ws1, ws2, ws3), axis=None)

# Fall velocity Van Rijn (1984)
dv=np.where(Rp**2<16.187)
ws1=ni*Rp[dv]**2/(18*d[dv])
dv=np.where((Rp**2>16.187) & (Rp**2<16187))
ws2=10*ni/d[dv]*(np.sqrt(1+0.01*Rp[dv]**2) -1)
dv=np.where(Rp**2>16187)
ws3=1.1*ni*Rp[dv]/d[dv]
ws_vanRijn=np.concatenate((ws1, ws2, ws3), axis=None)

# Fall velocity Dietrich (1982)
ws_diet=np.exp(-2.891394 + 0.95296*np.log(Rp) - 0.056835*(np.log(Rp))**2 - 0.002892*(np.log(Rp))**3 + 0.000245*(np.log(Rp))**4)

#np.sqrt((ss-1)*g*d)

plt.figure()
plt.plot(Rp,ws_diet)
plt.plot(Rp,ws_vanRijn)
plt.plot(D**1.5,ws_soulsby)
plt.plot(D**1.5,ws_haller)
#plt.xscale('log')
#plt.yscale('log')
plt.show()

