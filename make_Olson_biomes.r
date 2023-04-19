library(raster)
library(rgdal)

shape = readOGR("data/official/wwf_terr_ecos.shp")

egRaster = raster("/scratch/hadea/isimip3a/u-cc669_isimip3a_fire/20CRv3-ERA5_obsclim/jules-vn6p3_20crv3-era5_obsclim_histsoc_default_pft-bdlevgtrop_global_monthly_1901_2021.nc")

rr = rasterize(shape, egRaster, 'BIOME')

writeRaster(rr, file = 'data/wwf_terr_ecos_0p5.nc', overwrite = TRUE)
