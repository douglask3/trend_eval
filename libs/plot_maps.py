import iris
import numpy as np
import cartopy.crs as ccrs

import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from libs.to_precision import *
from pdb import set_trace as browser
from numpy import inf
import math

from   libs              import git_info

   
def plot_annual_mean(cube, levels, cmap, plot_name = None, scale = None, 
                     Nrows = 1, Ncols = 1, plot_n = 1, *args, **kw):

   
    aa = cube.collapsed('time', iris.analysis.MEAN)
    if plot_name is not None: aa.long_name = plot_name
    if scale is not None: aa.data = aa.data * scale
    
    plot_lonely_cube(aa, Nrows, Ncols, plot_n, levels = levels, cmap = cmap, 
                     colourbar = True, grayMask = True, *args, **kw)



def plot_lonely_cube(cube, N = None, M = None, n = None, levels = [0], extend = 'neither', colourbar = True, *args, **kw):

    cf, levels, extend = plot_cube(cube, N,  M, n, levels = levels, extend = extend, *args, **kw)
    if colourbar: 

        addColorbar(cf, levels, extend = extend)
    plt.tight_layout()
    return cf
    
def addColorbar(cf, ticks, *args, **kw):
    cb = plt.colorbar(cf, orientation='horizontal', ticks = ticks, *args, **kw)
    cb.ax.set_xticklabels(ticks)
    return cb

def plot_cube(cube, N, M, n, cmap, levels = None, extend = 'neither', projection = ccrs.Robinson(),
              grayMask = False):
    if levels is None:
        levels, extend = hist_limits(cube, levels, 6)

    if n is None:
        ax = plt.axes(projection = projection)
    else:
        ax = plt.subplot(N, M, n, projection = projection)

    ax.set_title(cube.long_name)

    cmap = plt.get_cmap(cmap)
    
    
    levelsi = [i for i in levels]
    
    if extend == "max" or extend == "both": levelsi += [9E9]
    if extend == "min" or extend == "both": levelsi = [-9E9] + levelsi

    if extend == "max" or extend == "min":
        norm = BoundaryNorm(levelsi, ncolors=cmap.N)
    else:
        norm = BoundaryNorm(levelsi, ncolors=cmap.N)
    
    if grayMask: plt.gca().patch.set_color('.9')
    try:
        cf = iplt.pcolormesh(cube, cmap = cmap, norm = norm) 
    except:
        cf = iplt.pcolormesh(cube, cmap = cmap) 
    
    plt.gca().coastlines()

    return cf, levels, extend


def plot_cubes_map(cubes, nms, cmap, levels, extend = 'neither',
                   figName = None, units = '', nx = None, ny = None, 
                   cbar_yoff = 0.0, figXscale = 1.0, figYscale = 1.0, 
                   totalMap = None, *args, **kw):
    
    try:
        cubeT =cubes.collapsed('time', totalMap)
        nms = [i for i in nms]
        nms.append('Total')
    except: cubeT = None  

    try: cubes = [cubes[i] for i in range(0, cubes.shape[0])]
    except: pass
    
    if cubeT is not None: cubes.append(cubeT)
    
    for i in range(0, len(cubes)):  cubes[i].long_name = nms[i]
    nplts = len(cubes)
    if nx is None and ny is None:
        nx = int(math.sqrt(nplts))
        ny = math.ceil(nplts / float(nx))
        nx = nx + 1.0
    elif nx is None:   
        nx = math.ceil(nplts / float(ny)) + 1
    elif ny is None:
        ny = math.ceil(nplts / float(nx))
    
    plt.figure(figsize = (nx * 2 * figXscale, ny * 4 * figYscale))

    for i in range(0, len(cubes)):         
        cmapi = cmap if (type(cmap) is str) else cmap[i]
        cf = plot_cube(cubes[i], nx, ny, i + 1, cmapi, levels, extend, *args, **kw)

    colorbar_axes = plt.gcf().add_axes([0.15, cbar_yoff + 0.5 / nx, 0.7, 0.15 / nx])
    cb = addColorbar(cf, levels, colorbar_axes, extend = extend)
    cb.set_label(units)

    plt.tight_layout()
    if (figName is not None):
        if figName == 'show':
            plt.show()
        else :
            print(figName)
            git = 'rev:  ' + git_info.rev + '\n' + 'repo: ' + git_info.url
            plt.gcf().text(.05, .95, git, rotation = 270, verticalalignment = "top")
            plt.savefig(figName, bbox_inches='tight')
            plt.clf()

def hist_limits(dat, lims = None, nlims = 5, symmetrical = True):
    def select_lims(prec, nlims):
        nlims0 = nlims
        for p in range(0,100 - nlims0):
            nlims = nlims0 + p
            lims  = np.percentile(dat.data[~np.isnan(dat.data)], range(0, 100, int(100/nlims)))
            
            if (lims[0]==-inf): lims.pop(0)
            
            lims = [to_precision(i, prec) for i in lims]
            lims = np.unique(lims)
            if (len(lims) >= nlims0): break
        return lims

    if (lims is None):
        for prec in range(1,5):
            lims = select_lims(prec, nlims)
            if len(lims) > 3: break

        new_lims = True
    else:
        new_lims = False
    if (lims[0] < 0.0):
        if (new_lims): 
            # are the more levels less than zero or gt  then zero  
            if (sum(i < 0.0 for i in lims) > sum(i > 0.0 for i  in lims)):
                # if more gt zero
                lims = [i for i in lims if i < 0.0]
                lims = np.concatenate((lims,[-i for i in lims[::-1]]))  

            else:
                # if more lt zero
                lims = [i for i in lims if i > 0.0]
                lims = np.concatenate(([-i for i in lims[::-1]], lims))     
        extend = 'both'
        
    else:
        extend = 'max'

    if len(lims) == 1: 
        lims = [-0.0001, -0.000001, 0.000001, 0.0001] if lims == 0.0 else [lims[0] * (1 + i) for i in [-0.1, -0.01, 0.01, 0.1]]
        extend = 'neither'

    if np.log10(lims[0]/lims[1]) > 3: lims[0] = 1000 * lims[1]
    if np.log10(lims[-1] / lims[-2]) > 3: lims[-1] = 1000 * lims[-2]
    if len(lims) < 4:
        lims = np.interp(np.arange(0, len(lims), len(lims)/6.0), range(0, len(lims)), lims)
    
    return (lims, extend)


