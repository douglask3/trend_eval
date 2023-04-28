
import cf_units
import iris
from pdb import set_trace

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
    coords = ('time', 'longitude', 'latitude')
    for coord in coords:
        try: 
            cube.coord(coord).guess_bounds()
        except:
            pass
            #if not cube.coord(coord).has_bounds():
            
    return(cube)
