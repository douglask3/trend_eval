Dir = 'outputs/jules_isimip_processed/'
FileStart = 'fire_on_JULES_ISIMIP-_'

varnames2plot = ['tree_cover', 'grass_cover', 'burnt_area']
cmaps = ['BuGn', 'YlGn', 'OrRd']
levels = [[0, 0.1, 0.5, 1, 5, 10, 20, 40, 60, 80, 100], 
          [0, 0.1, 0.5, 1, 5, 10, 20, 40, 60, 80, 100], 
          [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]]

scales = [1, 1, 12]


import matplotlib
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris.plot as iplt
import cartopy.crs as ccrs
from   libs.plot_maps    import *
import iris


def load_and_plot(varname, levels, cmap, scale, plot_n, Nrows, Ncols,*args, **kw) :
    
    filename = Dir + FileStart + varname + '.nc'
    cube = iris.load_cube(filename) * scale
    aa = cube.collapsed('time', iris.analysis.MEAN) 
    aa.long_name = varname
    
    plot_lonely_cube(aa, Nrows, Ncols, plot_n + 1, levels = levels, cmap = cmap, 
                     colourbar = True, grayMask = True, *args, **kw)

    ## could add other maps - value during height of fire season might be relevant?

Nrows = int(np.ceil(np.sqrt(3)))
Ncols = int(np.ceil(len(varnames2plot)/Nrows))

for var, lev, col, sc, i in \
        zip(varnames2plot, levels, cmaps, scales, range(len(varnames2plot))):
    load_and_plot(var, lev, col, sc, i, Nrows, Ncols)

plt.show()


