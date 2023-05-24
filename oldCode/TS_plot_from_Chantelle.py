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


#You can also cut out a section of the globe just using lat and long - this will give you a square rather than a specific country or biome:
cube=cube.extract(iris.Constraint(latitude=lambda cell: (-60.0) < cell < (15.0), longitude=lambda cell: (275) < cell < (330))) # Whole South America
cube=cube.extract(iris.Constraint(latitude=lambda cell: (-60.0) < cell < (10.0), longitude=lambda cell: (287) < cell < (326)))  #Brazil
cube=cube.extract(iris.Constraint(latitude=lambda cell: (1.0) < cell < (5.0), longitude=lambda cell: (298) < cell < (302)))  #Manaus
cube=cube.extract(iris.Constraint(latitude=lambda cell: (-17.0) < cell < (-13.0), longitude=lambda cell: (302) < cell < (306)))  #Primavera do Leste
