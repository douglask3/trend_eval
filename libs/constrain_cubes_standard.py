
#Region mask notes:
#https://regionmask.readthedocs.io/en/v0.9.0/
#https://regionmask.readthedocs.io/en/v0.9.0/notebooks/mask_2D.html
#https://regionmask.readthedocs.io/en/v0.9.0/defined_scientific.html

import iris
import iris.coord_categorisation as icc
import cartopy.io.shapereader as shpreader
from ascend import shape
import numpy as np

import cartopy.crs as ccrs
import geopandas as gp
import regionmask
from pdb import set_trace


def ar6_region(cube,year_range, *args, **kw):
    mask = regionmask.defined_regions.ar6
    set_trace()

def sub_year_range(cube, year_range, *args, **kw):
    """Selects months of a year from data   
    Arguments:
        cube -- iris cube with time array with year information.
        year_range -- numeric list of first to last year to cut
    Returns:
        cube of just years between to years provided.
    """
    return cube.extract(iris.Constraint(year=lambda cell: year_range[0] < cell <= year_range[1]))

def sub_year_months(cube, months_of_year):
    """Selects months of a year from data   
    Arguments:
        data -- iris cube with time array we can add add_month_number too.
        months_of_year -- numeric, month of the year you are interested in
                from 0 (Jan) to 11 (Dec)
    Returns:
        cube of just months we are interested in.
    """
    try: 
        icc.add_month_number(cube, 'time')
    except:
        pass  
    
    months_of_year = np.array(months_of_year)+1
    season = iris.Constraint(month_number = lambda cell, mnths = months_of_year: \
                             np.any(np.abs(mnths - cell[0])<0.5))
    return cube.extract(season)

def constrain_olson(cube, ecoregions):
    """constrains a cube to Olson ecoregion
    Assumes that the cube is iris and on a 0.5 defree grid

    Arguments:

    cube -- an iris cube at 0.5 degrees
    ecoregions -- numeric list (i.e [3, 7, 8]) where numbers pick Olson biomes.
        You can pick more than one:
            1 Tropical and subtropical moist broadleaf forests
            2 Tropical and subtropical dry broadleaf forests
            3 Tropical and suptropical coniferous forests
            4 Temperate broadleaf and mixed forests
            5 Temperate Coniferous Forest
            6 Boreal forests / Taiga
            7 Tropical and subtropical grasslands, savannas and shrublands
            8 Temperate grasslands, savannas and shrublands
            9 Flooded grasslands and savannas
            10 Montane grasslands and shrublands
            11 Tundra
            12 Mediterranean Forests, woodlands and scrubs
            13 Deserts and xeric shrublands
            14 Mangroves

    Returns:
    Input cube with areas outside of selected Olson biomes masked out.
    """
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

def constrain_natural_earth(cube, Country, Continent = None, shpfilename = None, *args, **kw):
    
    """constrains a cube to Natural Earth Country or continent.
    Assumes that the cube is iris
    If Country is defined, it wont select Continent.

    Arguments:

    cube -- an iris cube
    Continent -- name of continent. Options are:
        'South America'
        'Oceania'
        'Europe'
        'Afria'
        'North America'
        'Asia' 
    shpfilename -- path and filename of natural earth shapefile.
                   If set to None, it will look for temp. file version and 
                   download if it does not exist.
    Returns:
    Input cube constrained to the extent to that country or continent, 
    and mask areas outside of it. Uses Natural Earth
    """
    if shpfilename is None:
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

def constrain_region(cube, ecoregions = None, Country = None, Continent = None, *args, **kw):
    """ checks if any spatial constrains are set and contrains according to comments in functions above

    Arguments:

    cube -- an iris cube
    ecoregions -- see constrain_olson help
    Country, Continent -- see constrain_natural_earth help
      
    Returns:
    Input cube constrained to the extent to defined country or continent, 
    and mask areas outside of it and defined ecoregions.
    """
    if ecoregions is not None:
        cube = constrain_olson(cube, ecoregions)

    if Continent is not None or Country is not None:
        cube = constrain_natural_earth(cube, Country, Continent, *args, **kw)

    return(cube)
