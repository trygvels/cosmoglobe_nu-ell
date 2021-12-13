import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.colors as pcol

from cycler import cycler
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import matplotlib.patches
import matplotlib.gridspec
import matplotlib.ticker
import matplotlib.colors
# Style setup
pcolors=getattr(pcol.qualitative,"Plotly")
pcolors=getattr(pcol.qualitative,"D3")
#pcolors=plt.cm.get_cmap('tab10').colors
cm = plt.cm.get_cmap('tab20c')
colors=cm.colors

import matplotlib as mpl

pgf_without_latex = {
    "text.usetex": True,
    "text.latex.preamble": [
        r'\usepackage{txfonts}',
        r'\usepackage[T1]{fontenc}',
        r'\boldmath'],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
    "font.size": 20,
    "font.family": "serif",
    "legend.fontsize": 8,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
    }
mpl.rcParams.update(pgf_without_latex)

from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font0.set_weight('bold')
fontweight = 'bold'
fontproperties = {'weight' : fontweight}

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

#fig = plt.figure(figsize=(8,8))
fig = plt.figure(figsize=(16,12))
gs = gridspec.GridSpec(2,2, width_ratios=[5,1], height_ratios=[1,4])

lmin_plot = 1.5
lmax_plot = 11000
numin_plot = 10
numin_plot = 4
numax_plot=1000
fontsize=20

#ax2 = plt.subplot(gs[1])
ax1 = plt.subplot(gs[2])
plt.xscale('log')
plt.yscale('log')
plt.xlim(lmin_plot,lmax_plot)
plt.ylim(numin_plot,numax_plot)
#ax3.yaxis.tick_right()
#ax2.xaxis.tick_top()
#ax1.minorticks_on()


#sns.despine(top=True, right=True, left=True, bottom=True)

# We need to switch from a log scale to a linear scale to get the fancy
# box styles to work.  This transforms from data coordinates to axes
# coordinates.
trans1 = ax1.transData
trans2 = ax1.transAxes.inverted()
def data2axes(x,y):
    temp = np.zeros((1,2), dtype='double')
    temp[0,0] = x
    temp[0,1] = y
    display = trans1.transform(temp)
    axes = trans2.transform(display)
    return axes[0,0], axes[0,1]

def rspectrum(r, sig="BB", scaling=1.0):
    """
    Calculates the CMB amplituded given a value of r and requested modes
    """
    import camb
    from camb import model, initialpower
    import healpy as hp
    #Set up a new set of parameters for CAMB
    lmax=lmax_plot
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=r)
    pars.set_for_lmax(lmax,  lens_potential_accuracy=2)
    pars.WantTensors = True
    pars.AccurateBB = True
    pars.max_l_tensor = lmax
    pars.max_eta_k_tensor = 18000.
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(params=pars, lmax=lmax, CMB_unit='muK', raw_cl=True,)


    ell = np.arange(2,lmax+1)
    if sig == "TT":
        cl = powers['unlensed_scalar']
        signal = 0
    elif sig == "EE":
        cl = powers['unlensed_scalar']
        signal = 1
    elif sig == "BB":
        cl = powers['tensor']
        signal = 2

    #cl = cl[2:,signal] * 2 * np.pi / (ell*(ell+1))
    #cl = cl * ell * np.sqrt((2*ell+1)/2)
    plt.plot(ell, ell*(ell+1)*cl[2:,signal]/(2*np.pi)*scaling)
    if r==0.01:
        lens = powers['total'] - powers['tensor']
        plt.plot(ell, ell*(ell+1)*lens[2:,signal]/(2*np.pi)*scaling)
    
def lf(nu,Alf=5,betalf=-3.1,nuref=30):
    return Alf*(nu/nuref)**(betalf)

def mbb(nu,Ad=3,betad=1.55,Td=19.,nuref=353):
    h = 6.62607e-34
    k_b  = 1.38065e-23
    gamma = h/(k_b*Td)
    return Ad*(nu/nuref)**(betad+1)*(np.exp(gamma*nuref*1e9)-1)/(np.exp(gamma*nu*1e9)-1)


experiments={
    "GroundBIRD": {"nu": [145,220], "l":[6,300], "sensitivity": 0.9},
    "BICEP/KECK": {"nu": [95,220], "l":[30,400], "sensitivity": 0.2},
    "QUIJOTE": {"nu": [11, 41], "l": [4,100], "sensitivity": 0.77},
    "Polarbear": {"nu": [95, 148], "l": [30,4000], "sensitivity": 0.55},
    "C-BASS": {"nu": [4,6], "l": [2,300], "sensitivity": 0.1},
    "CMB-S4": {"nu": [20,278], "l":[30,5000], "sensitivity": 0.4},
    "SPIDER": {"nu": [94, 280], "l": [10,300], "sensitivity": 0.45},
    "WMAP": {"nu": [23,90], "l":[2,50], "sensitivity": 0.2},
    "LSPE": {"nu": [43,240], "l":[4,150], "sensitivity": 0.3},
    "SO": {"nu": [27, 280], "l": [10,10000], "sensitivity": 0.33},
    "SPT": {"nu": [90, 220], "l": [50, 11000], "sensitivity": 0.90},
    "ACT": {"nu": [28,220], "l":[50,10000], "sensitivity": 0.7},
    "PICO": {"nu": [21,799], "l": [2,3000], "sensitivity": 0.44},
    "Planck": {"nu": [30,353], "l":[2,1200], "sensitivity": 0.2},
    "QUBIC": {"nu": [90, 220], "l": [25,150], "sensitivity": 0.57},
    "LiteBIRD": {"nu": [40,402], "l":[2,200], "sensitivity": 0.1},
}
experiments={
    "BICEP/KECK": {"nu": [95,220], "l":[30,400], "sensitivity": 0.2},
    "QUIJOTE": {"nu": [11, 41], "l": [4,100], "sensitivity": 0.77},
    "Polarbear": {"nu": [95, 148], "l": [30,4000], "sensitivity": 0.55},
    "C-BASS": {"nu": [4,6], "l": [2,300], "sensitivity": 0.1},
    "SPIDER": {"nu": [94, 280], "l": [10,300], "sensitivity": 0.45},
    "WMAP": {"nu": [23,90], "l":[2,50], "sensitivity": 0.2},
    "SPT": {"nu": [90, 220], "l": [50, 11000], "sensitivity": 0.90},
    "ACT": {"nu": [28,220], "l":[50,10000], "sensitivity": 0.7},
    "Planck": {"nu": [30,353], "l":[2,1200], "sensitivity": 0.2},
}

experiments={
    "WMAP": {"nu": [23,90], "l":[2,50], "sensitivity": 0.2},
    "Planck": {"nu": [30,353], "l":[2,1200], "sensitivity": 0.2},
}


########################
###### EXPERIMENTS #####
########################

ax1.tick_params(which="both", direction="in", )
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(numin_plot,numax_plot)
ax1.set_xlim(lmin_plot, lmax_plot)
ax1.set_yticks([10, 30,100,300,])
ax1.set_yticklabels([r"$\mathbf{10}$",r"$\mathbf{30}$",r"$\mathbf{100}$",r"$\mathbf{300}$",],va="center", rotation=90)
xticks=[2,10,100,1000]
ax1.set_xticks(xticks)
ax1.set_xticklabels([r"$\mathbf{"+str(i)+"}$" for i in xticks])
ax1.set_xlabel(r'$\ell$ range') #, fontsize=fontsize)
ax1.set_ylabel(r'{Frequency range (GHz)') #, fontsize=fontsize)
def plot_rect(x1, y1, x2, y2, c, color=None, label=None,ls="-"):

            
    # 0.3 and 0.6 also work for transparencies...

    #boxstyle="roundtooth,pad=0.0,tooth_size=0.01"
    boxstyle="round,pad=0.0,rounding_size=0.03"

    # Set transparency in face color, not for whole BBox

    if c > 6:
        x = 0.99
        y = 0.92-0.035*(c-7)
        color=pcolors[c-7]
        ls="--"
        ha="right"
    else:
        ha="left"
        x = 0.78
        y = 0.92-0.035*c
        color=pcolors[c]
        ls="-"
    va="bottom"
    if label in ["LiteBIRD","LSPE", "GroundBIRD", "WMAP", "BICEP/KECK","SPIDER","C-BASS"]:
        ha="left"
        x_=x1*1.1 
    else:
        ha="right"
        x_=x2*0.90
    if label in ["QUIJOTE", "GroundBIRD","BICEP/KECK", "ACT", "CMB-S4", "SPIDER",]:
        y_=y1*1.02
    else:   
        y_=y2*1.02
        if label in "QUBIC":
            va="top"
            y_=y2*0.95
    ax1.text(x_,y_, r'\textbf{'+label+'}',fontsize=fontsize, color=color, ha=ha, va=va)

    x1, y1 = data2axes(x1, y1)
    x2, y2 = data2axes(x2, y2)

    cc = matplotlib.colors.ColorConverter()
    color_ = cc.to_rgba(color, alpha=0.05)
    #ax1.text(x,y, r'\textbf{'+label+'}', transform=fig.transFigure, fontsize=fontsize,color=color, ha=ha)
    p_fancy = matplotlib.patches.FancyBboxPatch((x1, y1),
        x2 - x1, y2 - y1,
        boxstyle=boxstyle,
        fc=color_, ec=color,
        transform=ax1.transAxes, 
        linewidth=2,
        linestyle=ls,label=label)
    ax1.add_patch(p_fancy)


c=len(experiments)-1
ls="-"
for label, info in experiments.items():
    numin,numax=info["nu"]
    lmin,lmax=info["l"]
    plot_rect(lmin, numin, lmax, numax, c,label=label,)
    c-=1
linecolor = '0.0'

"""
plt.annotate('', xy=(0.8,0.75), xytext=(1.03,1.04), xycoords=ax1.transAxes,
    textcoords=ax1.transAxes,
    arrowprops=dict(arrowstyle="-|>", linewidth=3, color='gray'))

plt.annotate("",xy=(0.78,0.96),xytext=(0.88,0.96), xycoords=fig.transFigure,
    textcoords=fig.transFigure, arrowprops={"color":"gray", "arrowstyle" : "-", "linestyle" : "-",
                         "linewidth" : 2,})
plt.annotate("",xy=(0.895,0.96),xytext=(0.99,0.96), xycoords=fig.transFigure,
    textcoords=fig.transFigure, arrowprops={"color":"gray", "arrowstyle" : "-", "linestyle" : "--",
                         "linewidth" : 2,})

ax1.axhline(y=0.96, xmin=0.8, xmax=0.9, ls="--")
ax1.axhline(y=0.96, xmin=0.9, xmax=0.99, ls="--")
"""
ax1.axvline(x=13, color=linecolor, ls='--',)
ax1.axvline(x=100, color=linecolor, ls='--',)
ax1.axhline(y=70, color=linecolor, ls='--')

########################
######   Plot 2    #####
########################

ax2 = plt.subplot(gs[0])#, sharex=ax1)
ax2.set_xticklabels(ax1.get_xticks(), fontproperties)
ax2.set_yticklabels(ax1.get_yticks(), fontproperties)
plt.ylim(1e-5,1e-0)
plt.xlim(lmin_plot,lmax_plot)

plt.yscale('log')
plt.xscale('log')
y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
ax2.yaxis.set_minor_locator(y_minor)
ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

#plt.minorticks_on()

ax2.axvline(x=13, color=linecolor, ls='--', ymax=0.8)
ax2.axvline(x=100, color=linecolor, ls='--', ymax=0.8)
ticks = [1e-5,1e-4,1e-3,1e-2,1e-1]
tick_string = [r'$\mathbf{10^{-5}}$', '',r'$\mathbf{10^{-3}}$','',\
        r'$\mathbf{10^{-1}}$',]
ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax2.axes.yaxis.set_ticklabels(tick_string,va="center", rotation=90)

rspectrum(r=0.1, sig="BB", scaling=1.0)
rspectrum(r=0.01, sig="BB", scaling=1.0)
fontsize = 16
ax2.text(2, 2.0e-3, r'\boldmath$r = 0.1$', transform=ax2.transData, color='C0',
    fontsize=fontsize) 
ax2.text(2, 2.0e-4, r'\boldmath$r = 0.01$', transform=ax2.transData, color='C1',
    fontsize=fontsize) 
ax2.text(2, 3.0e-5, r'\textbf{Lensing}', transform=ax2.transData,
        color='C2',
    fontsize=fontsize) 


y = 0.2e-1
max_theta = 180/lmin_plot
min_theta = 180/lmax_plot
topaxes = ax2.twiny()
topaxes.set_xlim(max_theta, min_theta)
topaxes.set_xscale('log')
labels = ['90', '10', '1', '0.1']
topaxes.set_xticks([float(x) for x in labels])
topaxes.get_xaxis().set_major_formatter( \
    matplotlib.ticker.FixedFormatter([r'\boldmath${{ {0}^{{\circ}} }}$'.format(x) for x in labels]))
topaxes.set_xlabel(r'\textbf{Angular scale} \boldmath$\Delta\theta\sim180^\circ/\ell$')
topaxes.tick_params(which="both", direction="in")

ax2.tick_params(which="both", direction="in", labelbottom=False)
ax2.set_ylabel(r'$C_\ell^\mathrm{BB}\ (\mu\mathrm{K}^2)$')


ax2.text(2, y, r'\textbf{Reion.}', transform=ax2.transData, color='k', fontsize=fontsize) 
ax2.text(15, y, r'\textbf{Recomb.}', transform=ax2.transData, color='k', fontsize=fontsize) 


y = 0.92
fig.text(0.53, y, r'\textbf{Lensing}', 
        transform=fig.transFigure, color='k', fontsize=fontsize) 
fig.text(0.35, y, r'\textbf{Inflation}', 
        transform=fig.transFigure, color='k', fontsize=fontsize) 

#plt.arrow(0.45, y, 0.50, y, transform=fig.transFigure, color='k') 
plt.annotate('', xy=(0.548-0.035,y+0.005), xytext=(0.50-0.035,y+0.005), xycoords=fig.transFigure,
    textcoords=fig.transFigure,
    arrowprops=dict(arrowstyle="-|>", linewidth=2, color='k'))

plt.annotate('', xytext=(0.50-0.035,y+0.005), xy=(0.452-0.035,y+0.005), xycoords=fig.transFigure,
    textcoords=fig.transFigure,
    arrowprops=dict(arrowstyle="-|>", linewidth=2, color='k'))


########################
###### foregrounds #####
########################

ax3 = plt.subplot(gs[3])#, sharey=ax1)
ax3.set_xticklabels(ax1.get_xticks(), fontproperties)
ax3.set_yticklabels(ax1.get_yticks(), fontproperties)
#ax3.get_yaxis().set_visible(False)
plt.xscale('log')
plt.yscale('log')
plt.ylim(numin_plot,numax_plot)
ax3.minorticks_on()
ax3.tick_params(which="both", direction="in", labelleft=False)
ax3.set_xlim(0.3,30)
ax3.set_xticks([1,10])
ax3.set_xticklabels([r"$\mathbf{1}$",r"$\mathbf{10}$",])

nu  = np.logspace(np.log10(0.1),np.log10(1000),1000)#*1e9
ax3.plot(lf(nu),nu, lw=3)
ax3.plot(mbb(nu),nu, lw=3)
ax3.axhline(y=70, color=linecolor, ls='--')
ax3.text(0.6,700, r'\textbf{Thermal}   \textbf{Dust}', color="C1",)# path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1, -1))])
ax3.text(0.60,75, r'\textbf{Fg. min.}', color="grey",)# path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1, -1))])
ax3.text(0.6,18, r'\textbf{Synchrotron}', color="C0",)# path_effects=[path_effects.withSimplePatchShadow(alpha=0.8, offset=(1, -1))])
ax3.set_xlabel(r'Foregrounds $(\mu\mathrm{K}^2)$')
ax3.xaxis.set_minor_locator(y_minor)
ax3.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())



"""/
COSMOGLOBE
"""
from PIL import Image
im = Image.open('/Users/svalheim/Drive/Bilder/Jobb/Material/CMB-logos/Cosmoglobe/Cosmoglobe-logo-square.png')
width = im.size[0]
height = im.size[1]
im = np.array(im).astype(np.float) / 255

#fig.figimage(im, fig.bbox.xmax, fig.bbox.ymax)
newax = fig.add_axes([0.82, 0.78, 0.16, 0.16], anchor='NE', zorder=-1)
newax.imshow(im)
newax.axis('off')

plt.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.94, 
    hspace=0.02*8.0/6.0, wspace=0.02)

    
if True:
    plt.savefig('Cosmoglobe-experiment-span_BP.pdf', bbox_inches="tight")
    plt.savefig('Cosmoglobe-experiment-span_BP.png', dpi=300, bbox_inches="tight")
    plt.savefig('Cosmoglobe-experiment-span_BP_small.png', dpi=50, bbox_inches="tight")
plt.show()