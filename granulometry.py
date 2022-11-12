import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def add_title(axes):
    for i, ax in enumerate(axes):
        ax.set_title("ax%d" % (i+1), fontsize=18)

df = pd.read_excel(r'./input_granu.xlsx')
print(df)

dpdf=df['W [gr]']/df['W [gr]'].sum()
WW=df['W [gr]'].cumsum()/df['W [gr]'].sum()

d05=np.interp(0.05, WW,df['d [mm]'])
d16=np.interp(0.16, WW,df['d [mm]'])
d25=np.interp(0.25, WW,df['d [mm]'])
d50=np.interp(0.50, WW,df['d [mm]'])
d75=np.interp(0.75, WW,df['d [mm]'])
d84=np.interp(0.84, WW,df['d [mm]'])
d95=np.interp(0.95, WW,df['d [mm]'])

ph05=-np.log2(d05)
ph16=-np.log2(d16)
ph25=-np.log2(d25)
ph50=-np.log2(d50)
ph75=-np.log2(d75)
ph84=-np.log2(d84)
ph95=-np.log2(d95)

# Logarithmic Folk & Ward
MzL=(ph16+ph50+ph84)/3
sigL=np.abs((ph84-ph16)/4+(ph95-ph05)/6.6)
skL=(ph16+ph84-2*ph50)/(2*(ph84-ph16))+(ph05+ph95-2*ph50)/(2*(ph95-ph05))
kuL=(ph95-ph05)/(2.44*(ph75-ph25))

if (sigL< 0.35):
    str_sort="Very well sorted"
elif (sigL>0.35 and sigL< 0.50):
    str_sort="Well sorted"
elif (sigL>0.50 and sigL< 0.70):
    str_sort="Moderately well sorted"
elif (sigL>0.70 and sigL<1.00):
    str_sort="Moderately sorted"
elif (sigL>1.00 and sigL<2.00):
    str_sort="Poorly sorted"
elif (sigL>2.00 and sigL<4.00):
    str_sort="Very poorly sorted"
elif (sigL>4.00):
    str_sort="Extremely poorly sorted"

if (skL>0.3 and skL<1.0):
    str_skew="Very fine skewed"
elif (skL>0.1 and skL<0.3):
    str_skew="Fine skewed"
elif (skL>-0.1 and skL<0.1):
    str_skew="Symmetrical"
elif (skL>-0.3 and skL<-0.1):
    str_skew="Coarse skewed"
elif (skL>-1.0 and skL<-0.3):
    str_skew="Very coarse skewed"

if (kuL<0.67):
    str_kurt="Very playkurtic"
if (kuL>0.67 and kuL<0.90):
    str_kurt="Playkurtic"
if (kuL>0.90 and kuL<1.11):
    str_kurt="Mesokurtic"
if (kuL>1.11 and kuL<1.50):
    str_kurt="Leptokurtic"
if (kuL>1.50 and kuL<3.00):
    str_kurt="Very Leptokurtic"
if (kuL>3.00):
    str_kurt="Extremely Leptokurtic"

#==============================================================================================
# Plotting Results
fig, ax = plt.subplots(figsize=(4,6))
ax.plot(df['d [mm]'],WW*100, lw=2)
ax.set_xscale('log')
plt.yticks(np.arange(0, 101, 10))
plt.grid(True, which="both", ls="-")
plt.xlabel('d [mm]')
plt.ylabel('% in weight')

# Adding info
ax.text(1.05, 1.05, r'SAMPLE '+df['Sample info'][5], transform=ax.transAxes,horizontalalignment='left',fontsize=16)
ax.text(1.05,1.00, r'Sample taken        '+df['Sample info'][2].strftime("%d/%m/%Y"), transform=ax.transAxes,horizontalalignment='left',fontsize=12)
ax.text(1.05,0.95, r'Sample processed '+df['Sample info'][8].strftime("%d/%m/%Y"), transform=ax.transAxes,horizontalalignment='left',fontsize=12)
ax.text(1.05, 0.9, r'Sample Weight [kg] ='+"{:5.3f}".format(df['W [gr]'].sum()/1000), transform=ax.transAxes,horizontalalignment='left',fontsize=12)

ax.text(1.05, 0.825, r'$d_{16}$ [mm] ='+"{:5.2f}".format(d16), transform=ax.transAxes)
ax.text(1.05, 0.775, r'$d_{25}$ [mm] ='+"{:5.2f}".format(d25), transform=ax.transAxes)
ax.text(1.05, 0.725, r'$d_{50}$ [mm] ='+"{:5.2f}".format(d50), transform=ax.transAxes)
ax.text(1.05, 0.675, r'$d_{75}$ [mm] ='+"{:5.2f}".format(d75), transform=ax.transAxes)
ax.text(1.05, 0.625, r'$d_{84}$ [mm] ='+"{:5.2f}".format(d84), transform=ax.transAxes)

fig.savefig('granu_01.png', bbox_inches='tight')
#plt.show()

#==============================================================================================
fig = plt.figure(figsize=(10, 7))
ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=2)
ax1.plot(df['d [mm]'],WW*100, lw=2)
ax1.set_xscale('log')
plt.yticks(np.arange(0, 101, 10))
plt.grid(True, which="both", ls="-")
plt.xlabel('d [mm]')
plt.ylabel('% in weight')
ax2 = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)
#ax2.plot(df['d [mm]'],dpdf, lw=2)
#ax2.step(df['d [mm]'],dpdf)
ax2.bar(df['d [mm]'],dpdf*100)
#ax2.set_xscale('log')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.xlabel('d [mm]')
plt.ylabel('% in weight')

# Adding info
ax1.text(1.05,1.00, r'SAMPLE '+df['Sample info'][5],transform=ax1.transAxes,horizontalalignment='left',fontsize=16)
ax1.text(1.05,0.95, r'Sample taken         '+df['Sample info'][2].strftime("%d/%m/%Y"), transform=ax1.transAxes,horizontalalignment='left',fontsize=12)
ax1.text(1.05,0.90, r'Sample processed  '+df['Sample info'][8].strftime("%d/%m/%Y"), transform=ax1.transAxes,horizontalalignment='left',fontsize=12)
ax1.text(1.05,0.85, r'Sample Weight [kg] ='+"{:5.3f}".format(df['W [gr]'].sum()/1000), transform=ax1.transAxes,horizontalalignment='left',fontsize=12)

ax1.text(1.05, 0.8, r'$d_{16}$ [mm] ='+"{:5.2f}".format(d16), transform=ax1.transAxes)
ax1.text(1.05, 0.76, r'$d_{25}$ [mm] ='+"{:5.2f}".format(d25), transform=ax1.transAxes)
ax1.text(1.05, 0.72, r'$d_{50}$ [mm] ='+"{:5.2f}".format(d50), transform=ax1.transAxes)
ax1.text(1.05, 0.68, r'$d_{75}$ [mm] ='+"{:5.2f}".format(d75), transform=ax1.transAxes)
ax1.text(1.05, 0.64, r'$d_{84}$ [mm] ='+"{:5.2f}".format(d84), transform=ax1.transAxes)

ax1.text(1.55, 0.8, r'Folk & Ward Logaritmic', transform=ax1.transAxes,fontsize=12)
ax1.text(1.55, 0.76, r'$\sigma$ ='+"{:5.2f}".format(sigL), transform=ax1.transAxes)
ax1.text(1.55, 0.72, r'$skew$ ='+"{:5.2f}".format(skL), transform=ax1.transAxes)
ax1.text(1.55, 0.68, r'$kurt$ ='+"{:5.2f}".format(kuL), transform=ax1.transAxes)

ax1.text(1.55, 0.63, str_sort, transform=ax1.transAxes)
ax1.text(1.55, 0.59, str_skew, transform=ax1.transAxes)
ax1.text(1.55, 0.55, str_kurt, transform=ax1.transAxes)

fig.savefig('granu_02.png', bbox_inches='tight')
