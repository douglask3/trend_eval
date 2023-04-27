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
Country = 'Brazil'
Continent = 'South America'

ecoregions = [3, 7, 8]


#some useful imports:
import os
import glob
import sys
sys.path.append('/net/home/h03/kwilliam/other_fcm/jules_py/trunk/jules/')
#~kwilliam/other_fcm/jules_py/trunk/jules/jules.py
import jules
import numpy.ma as ma
import iris
from   libs.iris_plus import *
from   libs.constrain_cubes_standard import *
import iris.analysis.stats

import numpy.ma as ma
import numpy as np

from pdb import set_trace as browser
import iris.coord_categorisation

############################
## Open up cube           ##
############################
def load_variable(variable, i = 0, multi = False):
    file_in = Jules_ISIMIP_dir + Jules_ISIMIP_fileID[0] + variable + Jules_ISIMIP_fileID[1]
    cube_in = iris.load(file_in, callback=sort_time)[0]
    cube = cube_in.copy()
    cube = cube.extract(iris.Constraint(year=lambda cell: year_range[0] < cell <= year_range[1]))
    cube = add_bounds(cube)
    cube = constrain_region(cube, ecoregions, Continent, Country)
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
        cube = cubes.collapsed('realization', iris.analysis.SUM)
    else:
        cube = load_variable(varname_in[0])
    
    cube.var_name = varname_out

    out_file = outDir + '/' + outFile + '_' + varname_out + '.nc'
    print(out_file)
    iris.save(cube, out_file)
    return cube


try: os.mkdir(outDir)
except: pass

all_variables = [load_variables(var_in, var_out) for var_in, var_out in zip(varnames_in, varnames_out)]





