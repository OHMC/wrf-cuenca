import datetime
import os

from optparse import OptionParser

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import wrf
import netCDF4
import xarray
import pandas as pd
import pangaea_lib as pa
import rasterio
import xarray as xr
from gazar.grid import ArrayGrid
from matplotlib.colors import LinearSegmentedColormap
from rasterstats import zonal_stats

from pyproj import CRS, Proj, transform, Transformer
from osgeo.osr import SpatialReference
from pyproj.crs import CRS

from osgeo import osr, gdalconst

import gdal
from gazar.grid import geotransform_from_yx, resample_grid, utm_proj_from_latlon, ArrayGrid

from affine import Affine

from gazar.grid import GDALGrid

import cuencas_wrf

wrfout = '/home/amontero/datos_wrfout/wrfout_A_d01_2020-05-16_06:00:00'

nc = netCDF4.Dataset(wrfout, mode='r')
rainnc = wrf.getvar(nc, 'RAINNC', timeidx=wrf.ALL_TIMES)
rainc = wrf.getvar(nc, 'RAINC', timeidx=wrf.ALL_TIMES)

# rainnc.values = rainnc.values + 1000
# rainc.values = rainc.values + 1000

arrs = {}
for t in range(1, len(rainnc.coords['Time'])):
    arrs[t] = rainnc[t].values[:, :] + rainc[t].values[:, :]
    arrs[t][arrs[t] == 0] = np.nan

rundate, xds = cuencas_wrf.corregir_wrfout(wrfout)
xds.lsm.rainc = rainc
xds.lsm.rainnc = rainnc

# https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.LambertConformal.html#wrf.LambertConformal
# rainc.projection.proj4()
# rainnc.projection.proj4()
# sorted(nc.variables)

# Projections
rainc_projection = osr.SpatialReference()
rainc_projection.ImportFromProj4(rainc.projection.proj4())

rainnc_projection = osr.SpatialReference()
rainnc_projection.ImportFromProj4(rainnc.projection.proj4())

lat, lon = xds.lsm.latlon
x, y = wrf.ll_to_xy(nc, lat, lon, meta=False)

# geotransform = geotransform_from_yx(y, x)  # -> No funciona, consume toda la memoria del sistema + swap

# driver = gdal.GetDriverByName('HDF5')
dataset = gdal.Open(wrfout, gdal.GA_ReadOnly)

# print("Driver: {}/{}".format(dataset.GetDriver().ShortName, dataset.GetDriver().LongName))
# print("Size is {} x {} x {}".format(dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount))
# print("Projection is {}".format(dataset.GetProjection()))
geotransform = dataset.GetGeoTransform()
# if geotransform:
#     print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
#     print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))
affine = Affine.from_gdal(*geotransform)

g = GDALGrid(wrfout)
# g.geotransform
# g.affine
# y_coords, x_coords = g.coords

# Cuando se exporta con wrf a proj4(desde la variable o como lo hace pangaea en el metodo _load_wrf_projection) el
# string esta mal, y por eso al exportarla con projection.ExportToWkt(), por ejemplo, devuelve un string vacio
# En https://proj.org/operations/projections/lcc.html se pueden encontrar los parametro validos
proj_str = "+proj=lcc +lat_1=-60.0 +lat_2=-30.0 +lat_0=-32.50000762939453 +lon_0=-62.70000076293945"
