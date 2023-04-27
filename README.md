# Coding-python


## Notes on constraining area


### Political areas
When opening data, setting ''Country'' or ''Continent'' will constrain the extent to that country or continent, and mask areas outside of it. Uses Natural Earth. If you define a country, it one look at the continent. Use None is you don't want any. Continent options are:
* 'South America'
* 'Oceania'
* 'Europe'
* 'Afria'
* 'North America'
* 'Asia'

### Ecoregions
''ecoregions'' is a numeric list (i.e [3, 7, 8]) where numbers pick Olson bomes and mask out everywhere else. If you  pick more than one, it returns a map of all of them.
* **None** Return all areas.
* **1** Tropical and subtropical moist broadleaf forests
* **2** Tropical and subtropical dry broadleaf forests
* **3** Tropical and suptropical coniferous forests
* **4** Temperate broadleaf and mixed forests
* **5** Temperate Coniferous Forest
* **6** Boreal forests / Taiga
* **7** Tropical and subtropical grasslands, savannas and shrublands
* **8** Temperate grasslands, savannas and shrublands
* **9** Flooded grasslands and savannas
* **10** Montane grasslands and shrublands
* **11** Tundra
* **12** Mediterranean Forests, woodlands and scrubs
* **13** Deserts and xeric shrublands
* **14** Mangroves

