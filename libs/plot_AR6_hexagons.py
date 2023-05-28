import geopandas as gpd
import geoplot
from matplotlib.colors import TwoSlopeNorm
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

def plot_AR6_hexagons(result, resultID = 1, vmin = 0.0, vcenter = 50.0, vmax = 100.0,
                      cmap = 'RdYlBu', add_colorbar = True, colorbar_label = '',
                      annotate_hexagons = True):  
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

