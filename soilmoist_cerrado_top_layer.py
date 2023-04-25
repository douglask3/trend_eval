import os
import glob
import sys
sys.path.append('/net/home/h03/kwilliam/other_fcm/jules_py/trunk/jules/')
#~kwilliam/other_fcm/jules_py/trunk/jules/jules.py
import jules
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
import cartopy.io.shapereader as shpreader
from ascend import shape
import netCDF4
import iris.coord_categorisation 

#redefine time units from mmyyyy to Julian year
def UpdateTime(cube):
    timeco = cube.coord('time')
    assert timeco.units == cf_units.Unit('months since 1901-01-01 00:00:00', calendar='proleptic_gregorian')
    timeco.units = cf_units.Unit('days since 1901-01-01 00:00:00', calendar='360_day')
    timeco.points = timeco.points * 30.
    iris.coord_categorisation.add_month(cube, 'time', name='month')
    return cube


#contraint time to the period of interest
def ConstrainTime(cube):
    date = iris.Constraint(time=lambda cell: 2019 <= cell.point.year <= 2019)
    cube = cube.extract(date)
    return cube

#load soil file and select top layer (layer 1)
cube = iris.load_cube('/net/home/h04/rveiga/Documents/soilmoist_cerrado_monthly_1901_2019.nc') [:,0,:,:] #select layer 1
#print (cube)
#exit()

cube = UpdateTime(cube)
cube = ConstrainTime(cube)
#print (cube)
#exit()

#collapse cube by mean of the time period and depth
soil_layer1 = cube.collapsed(['time'], iris.analysis.MEAN)
qplt.pcolormesh(soil_layer1)
plt.show()

#load the landmask
var_constraintx = iris.Constraint(cube_func=lambda x: x.var_name == 'field36')
landfrac = iris.load_cube('/data/cr1/cburton/IMPORTANT/CRU-NCEPv7.landfrac.nc', var_constraintx)
landfrac = landfrac.regrid(cube, iris.analysis.Linear()) 

#Plot timeseries
cube.data = cube.data * landfrac.data 
coords = ('longitude', 'latitude')
for coord in coords:
    if not cube.coord(coord).has_bounds():
      cube.coord(coord).guess_bounds()
weights = iris.analysis.cartography.area_weights(cube)
cube = cube.collapsed(coords, iris.analysis.MEAN, weights = weights) 

fig, ax = plt.subplots()
qplt.plot(cube.coord("month"),cube)
ax.set_title("Soil moisture content for the top layer")
ax.set_ylabel("Soil moisture content / kg m-2")
ax.set_xlabel('Year 2019')

plt.show()

#save plot as figure
#plt.savefig('Means_Cerrado/BurntArea_Cerrado_monthly_1990.jpeg')
#plt.close()


