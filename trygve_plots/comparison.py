import numpy as np
import matplotlib as mpl

def figsize(scale):
   fig_width_pt = 440. # Get this from LaTeX using \the\textwidth
   inches_per_pt = 1.0/72.27 # Convert pt to inch
   golden_mean = (np.sqrt(5.0)-1.0)/2.0 # Aesthetic ratio (you could change this)
   fig_width = fig_width_pt*inches_per_pt*scale # width in inches
   fig_height = fig_width*golden_mean # height in inches
   fig_size = [fig_width,fig_height]
   return fig_size


pgf_without_latex = {
    "text.usetex": True,
    "text.latex.preamble": [
        r'\usepackage{txfonts}',
        r'\usepackage[T1]{fontenc}',
        r'\boldmath'],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
    "font.size": 16,
    "font.family": "serif",
    "legend.fontsize": 8,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
    }
mpl.rcParams.update(pgf_without_latex)

from matplotlib.font_manager import FontProperties
font0 = FontProperties()
font0.set_weight('bold')


import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.gridspec
import matplotlib.ticker
import matplotlib.colors
import numpy as np

import thermo


def foo():
    import pylab as plt
    # Project = [low-ell,high-ell,low-freq,high-freq]
    CLASS = [2,150,30,230]
    ACT = [50,8000,25,230]
    PIPER = [2,300,200,600]
    Polarbear = [70,3000,80,240]
    BICEP = [50, 300, 80, 160]
    SPT = [50, 9000, 80, 230]
    Spider = [15, 300, 80 , 230]
    EBEX = [50, 1000, 130, 420]

    def bar_args(exp):
        return [exp[0],exp[3]-exp[2],exp[1]-exp[0],exp[2]]
    #plt.bar(*bar_args(CLASS))
    #plt.bar(*bar_args(ACT), color='red',alpha=0.5)
    #plt.bar(*bar_args(PIPER), color='green', alpha=0.5)
    def err_args(exp):
        return [(exp[0]+exp[1])/2.,(exp[3]+exp[2])/2.,(exp[3]-exp[2])/2.,(exp[0]-exp[1])/2.]
    ms=10
    lw=2
    plt.errorbar(*err_args(PIPER), fmt='gs', markersize=ms, linewidth=lw, label='PIPER')
    plt.errorbar(*err_args(ACT), fmt='rs', markersize=ms, linewidth=lw, label='AdvACT')
    plt.errorbar(*err_args(Polarbear), fmt='bs', markersize=ms, linewidth=lw, label='Polarbear')
    plt.errorbar(*err_args(BICEP), fmt='ms', markersize=ms, linewidth=lw, label='BICEP/Keck')
    plt.errorbar(*err_args(SPT), fmt='cs', markersize=ms, linewidth=lw, label='SPT 3G')
    plt.errorbar(*err_args(EBEX), fmt='ys', markersize=ms, linewidth=lw, label='EBEX')
    plt.errorbar(*err_args(Spider), fmt='s', color='orange', markersize=ms, linewidth=lw, label='Spider')
    plt.errorbar(*err_args(CLASS), fmt='ks',linewidth=4, markersize=20, capsize=8)
    ax=plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.text(2.5,140,r'\textbf{CLASS}',fontsize=32)
    plt.ylim([20,700])
    # plt.legend(loc='lower left', numpoints = 1)
    plt.xlabel(r'Multipole ($\ell$) Coverage', fontsize=16, fontweight='bold')
    plt.ylabel(r'Frequency Coverage', fontsize=16, fontweight='bold')
    plt.savefig('comparison.png', bbox_inches='tight')


def foo2():
    # order is:
    # name, lmin, lmax, freq min, freq max, color,
    # figure coordinate x for label, figure coordinate y
    col1 = 0.79
    col2 = 0.90
    experiments = [
            ['ACT', 50,8000,25,260,'c', col1, 0.90, 'solid'],
            ['PIPER', 2,300,200,600,'b', col1,0.85, 'dashed'],
            ['SPT', 50, 9000, 80, 260, 'y', col1,0.80, 'dashed'],
            # SO numbers come from https://arxiv.org/pdf/1808.07445.pdf
            #['Simons Observatory', 30,300,27,280,'g',col1,0.75, 'dotted'],
            ['Simons Array', 70,3000,80,240,'g',col1,0.75, 'dotted'],
            ['EBEX', 50, 1000, 130, 420, 'orange',col2,0.90, 'dashdot'],
            #['Spider', 30, 300, 80 , 230, 'm',col2,0.85, 'dashdot'],
            ['Spider', 15, 300, 90 , 280, 'm',col2,0.85, 'dashdot'],
            ['BICEP', 40, 300, 30, 270, 'r',col2,0.80, 'dotted']
        ]
    for e in experiments:
        e[7] = e[7] - 0.05
    CLASS = [r'CLASS', 2, 150, 30, 230, 'k',0,0,'solid']

    lmin = 1
    lmax = 1e4
    ghz_min = 20
    ghz_max = 700

    #font = {'family' : 'cm',
    #    'weight' : 'bold',
    #    'size'   : 22}
    #mpl.rc('font', **font)
    #mpl.rc('text', usetex=True)

    fig = plt.figure(figsize=(8,6))
    gs = matplotlib.gridspec.GridSpec(2,2, width_ratios=[3,1], height_ratios=[1,2])

    
    ax = plt.subplot(gs[2])
    fontweight = 'bold'
    fontproperties = {'weight' : fontweight}
    ax.set_xticklabels(ax.get_xticks(), fontproperties)
    ax.set_yticklabels(ax.get_yticks(), fontproperties)


    # Set up the axes scaling.
    plt.xlim(lmin,lmax)
    plt.ylim(ghz_min,ghz_max)
    plt.xscale('log')
    plt.yscale('log')

    # We need to switch from a log scale to a linear scale to get the fancy
    # box styles to work.  This transforms from data coordinates to axes
    # coordinates.
    trans1 = ax.transData
    trans2 = ax.transAxes.inverted()
    def data2axes(x,y):
        temp = np.zeros((1,2), dtype='double')
        temp[0,0] = x
        temp[0,1] = y
        display = trans1.transform(temp)
        axes = trans2.transform(display)
        return axes[0,0], axes[0,1]


    def plot_rect(x1, y1, x2, y2, color):
        x1, y1 = data2axes(x1, y1)
        x2, y2 = data2axes(x2, y2)

               
        # 0.3 and 0.6 also work for transparencies...
        alpha = 0.1
        hatch = None
        #if color == 'y':
        #    hatch = '//'
        #if color == 'g':
        #    hatch = r'\\'

        if color == 'k':
            alpha = 0.5

        #boxstyle="roundtooth,pad=0.0,tooth_size=0.01"
        boxstyle="round,pad=0.0,rounding_size=0.03"

        # Set transparency in face color, not for whole BBox
        cc = matplotlib.colors.ColorConverter()
        color = cc.to_rgba(xx[4], alpha=alpha)

        p_fancy = matplotlib.patches.FancyBboxPatch((x1, y1),
            x2 - x1, y2 - y1,
            boxstyle=boxstyle,
            fc=color, ec=xx[4],
            transform=ax.transAxes, 
            linewidth=2,
            linestyle=xx[7],
            hatch=hatch)
            #alpha=alpha)
        ax.add_patch(p_fancy)

        #p_fancy = matplotlib.patches.FancyBboxPatch((x1, y1),
        #    x2 - x1, y2 - y1,
        #    boxstyle=boxstyle,
        #    fc=(1,1,1,0), ec=xx[4],
        #    linestyle=xx[7],
        #    linewidth=2,
        #    transform=ax.transAxes) 
        #ax.add_patch(p_fancy)
        

    for e in range(len(experiments)):
        xx = experiments[e][1:]
        #for i in range(4):
        #    xx[i] = xx[i] * (100 + 5*np.random.randn())/100.0

        x1, x2, y1, y2 = xx[0:4]
    
        plot_rect(x1, y1, x2, y2, xx[4])

    xx = CLASS[1:]
    x1, x2, y1, y2 = xx[0:4]
    plot_rect(x1, y1, x2, y2, xx[4])

    exp = [x[0] for x in experiments]
    colors = {x[0]:x[5] for x in experiments}

    for e in range(len(experiments)):
        x = experiments[e]
        ax.text(x[6], x[7], r'\textbf{'+x[0]+'}', transform=fig.transFigure, fontsize=16,
            color=x[5])
        
    ax.text(2.5, 35, r'\textbf{CLASS}', ha='left', fontsize=22, 
            transform=ax.transData, color=(1,1,1)) 

    plt.annotate('', xy=(0.8,0.75), xytext=(1.03,1.04), xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", linewidth=3, color='gray'))

    linecolor = '0.0'

    plt.axvline(x=13, color=linecolor, ls='--')
    plt.axvline(x=100, color=linecolor, ls='--')
    plt.axhline(y=70, color=linecolor, ls='--')
    
    plt.xlabel(r'\boldmath$\ell$ \textbf{range}') #, fontsize=16)
    plt.ylabel(r'\textbf{Frequency range (GHz)}') #, fontsize=16)

    ticks = [2,10,100,1000]
    tick_string = [r'\textbf{%d}'%x for x in ticks]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.axes.xaxis.set_ticklabels(tick_string)

    ticks = [20,40,100,200,400]
    tick_string = [r'\textbf{%d}'%x for x in ticks]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.axes.yaxis.set_ticklabels(tick_string)

    # ----------------------------------------------------------------------
    ax0 = plt.subplot(gs[0])
    ax0.set_xticklabels(ax.get_xticks(), fontproperties)
    ax0.set_yticklabels(ax.get_yticks(), fontproperties)
    ax0.get_xaxis().set_visible(False)
    plt.yscale('log')
    plt.xscale('log')
    ax0.axes.xaxis.set_ticklabels([])
    #ax0.axes.yaxis.set_ticklabels([])
    ticks = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    tick_string = [r'$\mathbf{10^{-6}}$', '',r'$\mathbf{10^{-4}}$','',\
            r'$\mathbf{10^{-2}}$','']
    ax0.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax0.axes.yaxis.set_ticklabels(tick_string)


    data1 = np.loadtxt('../bb_plot/camb_jul2013/wmap9_ecmb_with_r0p01_tau0p1_tensCls.dat')
    data2 = np.loadtxt('../bb_plot/camb_jul2013/wmap9_ecmb_with_r0p0_tau0p1_lensedCls.dat')

    def plot_cl(data, scale=1.0, color=None):
        ell = data[:,0]
        bb = data[:,3]
        bb = bb * 2 * np.pi / (ell*(ell+1))
        bb = bb * ell * np.sqrt((2*ell+1)/2)
        if color is None:
            plt.plot(ell, bb*scale)
        else:
            plt.plot(ell, bb*scale)

    plot_cl(data1, scale=10, color='b')
    plot_cl(data1, color='g')
    plot_cl(data2, color='r')
    plt.ylim(1e-6,1e-0)
    plt.xlim(lmin,lmax)
    plt.ylabel(r'\boldmath$\sqrt{(2\ell+1)/2}\,\ell C_\ell^\mathrm{BB}\ (\mu\mathrm{K}^2)$')

    plt.axvline(x=13, color=linecolor, ls='--', ymax=0.8)
    plt.axvline(x=100, color=linecolor, ls='--', ymax=0.8)

    fontsize = 16
    y = 0.2e-1
    if False:
        x = 1700
        scale = 0.08
        ax0.text(x, y, r'\boldmath$r = 0.1$', transform=ax0.transData, color='r',
            fontsize=fontsize) 
        ax0.text(x, y*scale, r'\boldmath$r = 0.01$', transform=ax0.transData, color='b',
            fontsize=fontsize) 
        ax0.text(x, y*scale**2, r'\textbf{Lensing}', transform=ax0.transData, color='g',
            fontsize=fontsize) 

    #ax0.text(15, 2e-4, '$r = 0.1$', transform=ax0.transData, color='b',
        #fontsize=fontsize) 
    ax0.text(1.3, 2.0e-3, r'\boldmath$r = 0.1$', transform=ax0.transData, color='C0',
        fontsize=fontsize) 
    #ax0.text(1.3, 1e-5, '$r = 0.01$', transform=ax0.transData, color='g',
        #fontsize=fontsize) 
    ax0.text(1.3, 2.0e-4, r'\boldmath$r = 0.01$', transform=ax0.transData, color='C1',
        fontsize=fontsize) 
    ax0.text(1.3, 3.0e-5, r'\textbf{Lensing}', transform=ax0.transData,
            color='C2',
        fontsize=fontsize) 



    ax0.text(2, y, r'\textbf{Reion.}', transform=ax0.transData, color='k', fontsize=fontsize) 
    ax0.text(15, y, r'\textbf{Recomb.}', transform=ax0.transData, color='k', fontsize=fontsize) 


    fig.text(0.55, 0.9, r'\textbf{Lensing}', 
            transform=fig.transFigure, color='k', fontsize=fontsize) 
    fig.text(0.25, 0.9, r'\textbf{Inflation}', 
            transform=fig.transFigure, color='k', fontsize=fontsize) 

    y = 0.91
    #plt.arrow(0.45, y, 0.50, y, transform=fig.transFigure, color='k') 
    plt.annotate('', xy=(0.50,y), xytext=(0.452,y), xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops=dict(arrowstyle="-|>", linewidth=2, color='k'))

    plt.annotate('', xytext=(0.448,y), xy=(0.40,y), xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops=dict(arrowstyle="-|>", linewidth=2, color='k'))

    ax0.set_xlim(lmin, lmax)
    max_theta = 180/lmin
    min_theta = 180/lmax
    topaxes = ax0.twiny()

    topaxes.set_xlim(max_theta, min_theta)
    topaxes.set_xscale('log')
    labels = ['90', '10', '1', '0.1']
    topaxes.set_xticks([float(x) for x in labels])
    topaxes.get_xaxis().set_major_formatter( \
        matplotlib.ticker.FixedFormatter([r'\boldmath${{ {0}^{{\circ}} }}$'.format(x) for x in labels]))
    topaxes.set_xlabel(r'\textbf{Angular scale} \boldmath$\Delta\theta\sim180^\circ/\ell$')

    # ----------------------------------------------------------------------
    ax3 = plt.subplot(gs[3])
    ax3.set_xticklabels(ax.get_xticks(), fontproperties)
    ax3.set_yticklabels(ax.get_yticks(), fontproperties)
    ax3.get_yaxis().set_visible(False)

    #ax3.fill_between(np.logspace(0,5),33,43,   alpha=0.3, color='m')
    #ax3.fill_between(np.logspace(0,5),77,108,  alpha=0.3, color='m')
    #ax3.fill_between(np.logspace(0,5),127,163, alpha=0.3, color='m')
    #ax3.fill_between(np.logspace(0,5),200,234, alpha=0.3, color='m')

    ghz = np.logspace(np.log10(20),3,num=200)
    beta_dust = 1.5
    beta_sync = -2.9
    dust = 0.65 * (ghz/65.0)**(beta_dust*2)
    sync = 0.64 * (ghz/65.0)**(beta_sync*2)
    cross = 0.46 * 0.65 * 0.64 * 2 * (ghz/65.0)**(beta_dust + beta_sync)
    fg = dust + sync + cross
    #fg = fg * thermo.a2t(ghz*1e9)**2
    #plt.plot(fg, ghz)
    #print ghz[np.where(fg == min(fg))[0]]

    ## Directly from Figure 51 of Planck X 2015
    ghz, fg = np.loadtxt('scraped_planckplot.txt').T
    fg = fg**2 * thermo.a2t(ghz*1e9)**2
    plt.plot(fg, ghz)


    #ghz, fg = np.loadtxt('scraped_planckplot_low.txt').T
    #fg = fg**2 * thermo.a2t(ghz*1e9)**2
    #plt.plot(fg, ghz)
    #print ghz[np.where(fg == min(fg))[0]] # 80 GHz

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'\textbf{Foregrounds} \boldmath$(\mu\mathrm{K}^2)$')
    ax3.axes.yaxis.set_ticklabels([])
    #ax3.axes.xaxis.set_ticklabels([])
    ticks = [1e1,1e2,1e3,1e4,1e5]
    tick_string = [r'$\mathbf{10^{\phantom{1}}}$','',\
            r'$\mathbf{10^3}$','',r'$\mathbf{10^5}$']
    ax3.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax3.axes.xaxis.set_ticklabels(tick_string)
    plt.ylim(ghz_min,ghz_max)
    plt.xlim(1,1e5)

    ax3.text(10, 300, r'\textbf{Dust}', transform=ax3.transData, color='k',
        fontsize=fontsize) 
    ax3.text(10, 23, r'\textbf{Synchrotron}', transform=ax3.transData, color='k',
        fontsize=fontsize) 
    ax3.text(10, 75, r'\textbf{Minimum}', transform=ax3.transData, color='k',
        fontsize=fontsize) 


    plt.axhline(y=70, color=linecolor, ls='--')

    plt.subplots_adjust(left=0.13, right=0.99, bottom=0.14, top=0.94, 
        hspace=0.02*8.0/6.0, wspace=0.02)

    plt.savefig('comparison.pdf', bbox_inches="tight")
    plt.savefig('comparison.png', dpi=100, bbox_inches="tight")
    plt.savefig('comparison_small.png', dpi=50, bbox_inches="tight")
    #plt.show()


if __name__ == "__main__":
    #foo()
    foo2()
