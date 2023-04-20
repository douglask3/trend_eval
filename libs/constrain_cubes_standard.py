
import iris
import cartopy.io.shapereader as shpreader
from ascend import shape
import numpy as np

def constrain_olson(cube, ecoregions):
    biomes = iris.load_cube('data/wwf_terr_ecos_0p5.nc')
    mask = biomes.copy()
    mask.data[:] = 0.0
    for ecoreg in ecoregions: mask.data += biomes.data == ecoreg
    mask.data[mask.data.mask] = 0.0
    mask = mask.data == 0

    for layer in cube.data:
        layer.mask[mask] = False
        layer[mask] = np.nan
    return cube

def constrain_natural_earth(cube, Continent, Country):
    shpfilename = shpreader.natural_earth(resolution='110m', 
                                          category='cultural', name='admin_0_countries')
    natural_earth_file = shape.load_shp(shpfilename)
    if Country is not None:
        NAMES = [i.attributes.get('NAME') for i in natural_earth_file]
        NAME = [s for s in NAMES if Country in s][0]
        CountrySelect = shape.load_shp(shpfilename, NAME=NAME)
    elif Continent is not None:
        CountrySelect = shape.load_shp(shpfilename, Continent='South America')
        CountrySelect = Country.unary_union()
    cube = CountrySelect[0].constrain_cube(cube)
    
    cube = CountrySelect[0].mask_cube(cube)
    return cube

def constrain_region(cube, ecoregions, Continent, Country):
    if ecoregions is not None:
        cube = constrain_olson(cube, ecoregions)

    if Continent is not None or Country is not None:
        cube = constrain_natural_earth(cube, Continent, Country)

    return(cube)
