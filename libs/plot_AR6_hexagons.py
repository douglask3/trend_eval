import geopandas as gpd
import geoplot
from matplotlib.colors import TwoSlopeNorm
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

def plot_AR6_hexagons(result, resultID = 1, vmin = 0.0, vcenter = 50.0, vmax = 100.0,
                      cmap = 'RdYlBu', add_colorbar = True, colorbar_label = '',
                      annotate_hexagons = True):  
    """plots AR6 region based results in a hexagon map
        Assumes the AR6 region id is in the first element of "result" input
        Colorbar is just linear at the mo. I'll change this at some point.
    Arguments:
        result -- a 2-d list of results you want to plot. Each element in list has at least 2 
                    entries. The first is the region ID as a string and the 2nd is the 
                    numeric result you want to plotin that region
        resultID -- integer. The entry poistion each each result element that you want to plot.                     Default set to one assuming most the the time, result will be region ID 
                    and result only.
        vmin --- the minimum value in the colourmap
        vcenter -- the minimum value in the colourmap
        vmax -- the maximum value in teh colourmap
        cmap -- colourmap
        add_colorbar -- Boolean. If True, add colourbar legend to the bottom of the map
        colorbar_label -- String. If 'add_colorbar' is True, then this is the label added to 
                    the colorbar
        annotate_hexagons -- Boolean. If True, addes region IDs to hexagons.
    Returns:
        Hexogan map of results
    """
    hexagons = gpd.read_file('data/WeightedZones.gpkg').copy()
    hexagons['Model_results'] = np.nan
    for res in result:
        i_reg = hexagons['label'] == res[0]
        if np.sum(i_reg) == 0: continue
        if np.sum(i_reg) >  1: set_trace()
        hexagons['Model_results'][i_reg] = res[resultID]*100

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap,)

    ax = hexagons.plot()
    hexagons.plot(column='Model_results', cmap=cmap, norm=norm, 
                  linewidth=0.8, edgecolor='gray', legend=False, ax=ax)
    if annotate_hexagons:
        hexagons.apply(lambda x: ax.annotate(text=x['label'], 
                       xy=x.geometry.centroid.coords[0], ha='center'), axis=1);
    if add_colorbar:
        plt.colorbar(cbar,ticks=[0, 25, 50, 75, 100],orientation='horizontal', 
                     label= colorbar_label )
    plt.axis('off')     

