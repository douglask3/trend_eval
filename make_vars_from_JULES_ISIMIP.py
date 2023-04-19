#python3

# -*- coding: iso-8859-1 -*-

#Get jules.py from here: https://code.metoffice.gov.uk/trac/jules/browser/main/branches/pkg/karinawilliams/r6715_python_packages/share/jules.py

#See iris help pages for more plotting and analysis tips here: 
# https://scitools.org.uk/iris/docs/v2.2/
# https://scitools-iris.readthedocs.io/en/stable/index.html

##################################
## Set what you want to extract ##
##################################

## Which variables you want to import. Each element is a list. More than one item in teh list, and they get added together.
varnames_in = [['pft-bdldcd', 'pft-bdlevgtrop', 'pft-bdlevgtemp', 'pft-ndldcd', 'pft-ndlevg'],
               ['pft-c3grass', 'pft-c4grass'], 
               ['burntarea-total']]

## What your going to call the saved output
varnames_out = ['tree_cover', 'grass_cover', 'burnt_area']

## The start and end of the range of years you want
year_range = [2000, 2003]

## Where are you getting it from. List and change the directory depending on what you want.
## for isimip stuff, varname is saved in teh file. So this is the bit of file before and after the variable name.
Jules_ISIMIP_dir = "/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/"
Jules_ISIMIP_fileID = ["jules-vn6p3_20crv3-era5_obsclim_histsoc_default_", "_global_monthly_1901_2021.nc"]

outDir = 'outputs/jules_isimip_processed/'
outFile = 'fire_on_JULES_ISIMIP-'

## Which country or continent. If you define a country, it one look at the continent. Also clips the data to the extent of that country or continent. Use None is you don't want any
Country = 'Brazil'
Continent = 'South America'
#Continent options: 'South America', 'Oceania', 'Europe', 'Afria', 'North America', 'Asia' 

## Which ecoregion from Olson.  Use None is you don't want any
ecoregions = [3, 7, 8]

#Ecoregion options. You can pick more than one
#1 Tropical and subtropical moist broadleaf forests
#2 Tropical and subtropical dry broadleaf forests
#3 Tropical and suptropical coniferous forests
#4 Temperate broadleaf and mixed forests
#5 Temperate Coniferous Forest
#6 Boreal forests / Taiga
#7 Tropical and subtropical grasslands, savannas and shrublands
#8 Temperate grasslands, savannas and shrublands
#9 Flooded grasslands and savannas
#10 Montane grasslands and shrublands
#11 Tundra
#12 Mediterranean Forests, woodlands and scrubs
#13 Deserts and xeric shrublands
#14 Mangroves



#some useful imports:
import os
import glob
import sys
sys.path.append('/net/home/h03/kwilliam/other_fcm/jules_py/trunk/jules/')
#~kwilliam/other_fcm/jules_py/trunk/jules/jules.py
import jules
import numpy.ma as ma
import iris
import numpy as np
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import iris.plot as iplt
from scipy import stats
import iris.analysis.stats
import scipy.stats
import matplotlib
import numpy.ma as ma
import cf_units
from pdb import set_trace as browser
import iris.coord_categorisation
import cartopy.io.shapereader as shpreader
from ascend import shape

#Function to sort out the time dimension
def sort_time(cube, field, filename):
    cube.coord("time").bounds=None
    tcoord = cube.coord("time")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="gregorian")
    tcoord.convert_units("days since 1661-01-01 00:00:00")
    tcoord.units = cf_units.Unit(tcoord.units.origin, calendar="proleptic_gregorian")
    cube.remove_coord("time")
    cube.add_dim_coord(tcoord, 0) # might need to find this dimension
    iris.coord_categorisation.add_year(cube, 'time')
    iris.coord_categorisation.add_month(cube, 'time')

    return(cube)

def add_bounds(cube):
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    return(cube)


def constrain_olson(cube):
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

def constrain_natural_earth(cube):
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

def constrain_region(cube):
    if ecoregions is not None:
        cube = constrain_olson(cube)

    if Continent is not None or Country is not None:
        cube = constrain_natural_earth(cube)

    return(cube)



try: os.mkdir(outDir)
except: pass

def load_variable(variable, i = 0, multi = False):
    file_in = Jules_ISIMIP_dir + Jules_ISIMIP_fileID[0] + variable + Jules_ISIMIP_fileID[1]
    cube_in = iris.load(file_in, callback=sort_time)[0]
    cube = cube_in.copy()
    cube = cube.extract(iris.Constraint(year=lambda cell: year_range[0] < cell <= year_range[1]))
    cube = add_bounds(cube)
    cube = constrain_region(cube)
    if multi:
        coord = iris.coords.DimCoord(i, 'realization')
        cube.add_aux_coord(coord)
        cube.var_name = 'None'  
        del cube.attributes['history']  
    return cube

def load_variables(varname_in, varname_out = None):
    if len(varname_in) > 1:
        cubes = [load_variable(var, i, True) for var, i in zip(varname_in, range(len(varname_in)))]
        cubes = iris.cube.CubeList(cubes).merge_cube()
        try:
            cube = cubes.collapsed('realization', iris.analysis.SUM)
        except:
            browser()
    else:
        cube = load_variable(varname_in[0])
    
    cube.var_name = varname_out

    out_file = outDir + '/' + outFile + '_' + varname_out + '.nc'
    print(out_file)
    iris.save(cube, out_file)
    return cube

all_variables = [load_variables(var_in, var_out) for var_in, var_out in zip(varnames_in, varnames_out)]

#Region mask notes:
#https://regionmask.readthedocs.io/en/v0.9.0/
#https://regionmask.readthedocs.io/en/v0.9.0/notebooks/mask_2D.html
#https://regionmask.readthedocs.io/en/v0.9.0/defined_scientific.html


