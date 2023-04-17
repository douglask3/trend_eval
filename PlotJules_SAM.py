#python3

# -*- coding: iso-8859-1 -*-

#Get jules.py from here: https://code.metoffice.gov.uk/trac/jules/browser/main/branches/pkg/karinawilliams/r6715_python_packages/share/jules.py

#See iris help pages for more plotting and analysis tips here: 
# https://scitools.org.uk/iris/docs/v2.2/
# https://scitools-iris.readthedocs.io/en/stable/index.html


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
    #cube = cube.extract(iris.Constraint(year=lambda cell: 2090 > cell))
    return(cube)



#If loading in raw ISIMIP data:
pwdproc="/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/"

burnt_area_total=iris.load(pwdproc+"jules-vn6p3_20crv3-era5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2021.nc", callback=sort_time)[0]
burnt_area=iris.cube.Cube([])


bfiles=sorted( filter( os.path.isfile,
                        glob.glob(pwdproc+"*burnt*") ) )
ffiles=sorted( filter( os.path.isfile,
                        glob.glob(pwdproc+"*_pft*monthly*2021*") ) )
print(len(ffiles),len(bfiles))
icount=0
for ffile,bfile in zip(ffiles, bfiles):
    if "burntarea-total" not in bfile:
        print(ffile,bfile)
        cube = iris.load(bfile, callback=sort_time)[0]*iris.load(ffile, callback=sort_time)[0]
        #cube.data=cube.data/100.  
        if icount == 0:
            icount=1
            burnt_area_pft = cube
        else:
            burnt_area_pft = cube + burnt_area_pft
#iris.coord_categorisation.add_year(burnt_area_pft, 'time')
#iris.coord_categorisation.add_year(burnt_area_total, 'time')
burnt_area_pft=burnt_area_pft.extract(iris.Constraint(year=lambda cell: 2000 < cell < 2002 ))
burnt_area_pft=burnt_area_pft.collapsed(["time"],iris.analysis.SUM)/100.0
burnt_area_total=burnt_area_total.extract(iris.Constraint(year=lambda cell: 2000 < cell < 2002 ))
burnt_area_total=burnt_area_total.collapsed(["time"],iris.analysis.SUM)
print(burnt_area_pft.data)
print(burnt_area_total.data)
qplt.contourf(burnt_area_pft)
iris.save(burnt_area_pft,"burnt_area_pft.nc")
plt.show()
qplt.contourf(burnt_area_total)
iris.save(burnt_area_total,"burnt_area_total.nc")
plt.show()
try:
    burnt_area_total.coord('latitude').guess_bounds()
    burnt_area_total.coord('longitude').guess_bounds()
except:
    print("Already have weights")
weights= iris.analysis.cartography.area_weights(burnt_area_total)
print("from pft frac", burnt_area_pft.collapsed(['latitude','longitude'], iris.analysis.SUM, weights=weights).data/1e12)
print("from global", burnt_area_total.collapsed(['latitude','longitude'], iris.analysis.SUM, weights=weights).data/1e12)


## Some other basic functions

#Load a file to see what's there
cube = jules.load('/point/to/file/location/JULESannualfile.nc')
print (cube)

#load burnt area from the input file
var_constraint = iris.Constraint(cube_func=lambda x: x.var_name == 'burnt_area_gb') 
cube = jules.load_cube('/point/to/file/location/JULESannualfile.nc', var_constraint, missingdata=np.ma.masked)*86400*360 #multiply by seconds in a day and days in a JULES year
cube.data[np.where(np.isnan(cube.data))] = 0.0
print (cube)

#plot a map of annual mean burnt area fraction 
BAmap = cube.collapsed(['time'], iris.analysis.MEAN)
qplt.pcolormesh(BAmap)
plt.show()

#load the landmask
var_constraintx = iris.Constraint(cube_func=lambda x: x.var_name == 'field36')
landfrac = iris.load_cube('/data/cr1/cburton/IMPORTANT/CRU-NCEPv7.landfrac.nc', var_constraintx)
landfrac = landfrac.regrid(cube, iris.analysis.Linear()) 

#Plot a timeseries of global annual total burnt area in Mkm2
cube.data = cube.data * landfrac.data #multiply by land fraction to account for coastal gridboxes
coords = ('longitude', 'latitude')
for coord in coords:
    if not cube.coord(coord).has_bounds():
        cube.coord(coord).guess_bounds()
weights = iris.analysis.cartography.area_weights(cube)
cube = cube.collapsed(coords, iris.analysis.SUM, weights = weights) / 1E12 
qplt.plot(cube)
plt.show()


#Other variables of interest:
#veg_c_fire_emission_gb + burnt_carbon_rpm + burnt_carbon_dpm = total carbon emissions from vegetation (find the sum over lat and lon and divide by 1E12 to get GtC, no need to multiply by 86400 and 360 for these)
#emission factors (check units):
#fire_em_CO2_gb
#fire_em_NOx_gb
#fire_em_BC_gb
#fire_em_CH4_gb
#cv = vegetation cabon
#precip
#t1p5m_gb

#Constrain to a region using natural earth continents or countries
import cartopy.io.shapereader as shpreader
from ascend import shape
shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
natural_earth_file = shape.load_shp(shpfilename)
CountrySelect = shape.load_shp(shpfilename, Continent='South America')
CountrySelect = Country.unary_union()
CountrySelect.show()
#or
CountrySelect = shape.load_shp(shpfilename, Name='Brazil')
CountryData = CountrySelect.mask_cube(cube)

coords = ('longitude', 'latitude')
for coord in coords:
    if not cube.coord(coord).has_bounds():
        cube.coord(coord).guess_bounds()
grid_weights = iris.analysis.cartography.area_weights(CountrySelect)
#Collapse to find timeseries
cube = CountrySelect.collapsed(coords, iris.analysis.SUM, weights = grid_weights1) / 1E12

#Continent options = 'South America', 'Oceania', 'Europe', 'Afria', 'North America', 'Asia' 
#Region mask notes:
#https://regionmask.readthedocs.io/en/v0.9.0/
#https://regionmask.readthedocs.io/en/v0.9.0/notebooks/mask_2D.html
#https://regionmask.readthedocs.io/en/v0.9.0/defined_scientific.html


#You can also cut out a section of the globe just using lat and long - this will give you a square rather than a specific country or biome:
cube=cube.extract(iris.Constraint(latitude=lambda cell: (-60.0) < cell < (15.0), longitude=lambda cell: (275) < cell < (330))) # Whole South America
cube=cube.extract(iris.Constraint(latitude=lambda cell: (-60.0) < cell < (10.0), longitude=lambda cell: (287) < cell < (326)))  #Brazil
cube=cube.extract(iris.Constraint(latitude=lambda cell: (1.0) < cell < (5.0), longitude=lambda cell: (298) < cell < (302)))  #Manaus
cube=cube.extract(iris.Constraint(latitude=lambda cell: (-17.0) < cell < (-13.0), longitude=lambda cell: (302) < cell < (306)))  #Primavera do Leste


#total burnt area:
try:
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
except:
    print("Already have weights")
weights= iris.analysis.cartography.area_weights(cube)
cube_out=cube.copy()
cube_out.data=cube.data*weights
print("total_burnt_area from total", cube_out.collapsed(['latitude','longitude'], iris.analysis.SUM).data/1e12, "million sq km2")


# derived from pft fraction
#print(cube_cf)
#print(frac)
cube_pft = cube_cf*frac
try:
    cube_pft.coord('latitude').guess_bounds()
    cube_pft.coord('longitude').guess_bounds()
except:
    print("Already have weights")
cube_pft = cube_pft.collapsed(['pft'], iris.analysis.SUM)
#print(cube_pft)
cube_out=cube_pft.copy()
cube_out.data=cube_pft.data*weights
print("from pft frac", cube_out.collapsed(['latitude','longitude'], iris.analysis.SUM).data/1e12)


