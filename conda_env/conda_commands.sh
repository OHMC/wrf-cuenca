#!/bin/bash

conda install -c conda-forge geopandas -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge rasterio -y
conda install -c conda-forge gazar -y
conda install -c conda-forge rasterstats -y
conda install -c conda-forge gdal -y
conda install -c conda-forge xarray dask netCDF4 bottleneck -y
conda install -c conda-forge pangaea -y

#############################################
#conda install -c conda-forge pandas=0.25.0
#conda install -c conda-forge matplotlib=3.1.2
#conda install -c conda-forge xarray=0.12.3 dask netCDF4 bottleneck


# 22-5-2020
conda create -n wrfcuenca3 python=3.8.2
conda activate wrfcuenca3
pip install matplotlib==3.1.2
conda install -c conda-forge pandas=0.25.3
conda install -c conda-forge dask netCDF4 bottleneck
conda install -c conda-forge pyproj=2.4.1
conda install -c conda-forge geos
conda install -c conda-forge shapely
conda install -c conda-forge rtree

conda install -c conda-forge gdal
#The following packages will be DOWNGRADED:
#
#  curl                                    7.69.1-h33f0ec9_0 --> 7.68.0-hf8cf82a_0
#  geos                                     3.8.1-he1b5a44_0 --> 3.7.2-he1b5a44_2
#  hdf5                            1.10.6-nompi_h3c11f04_100 --> 1.10.5-nompi_h3c11f04_1104
#  krb5                                    1.17.1-h173b8e3_0 --> 1.16.4-h173b8e3_0
#  libcurl                                 7.69.1-hf7181ac_0 --> 7.68.0-hda55be3_0
#  libnetcdf                        4.7.4-nompi_h84807e1_104 --> 4.7.1-nompi_h94020b1_102
#  netcdf4                      1.5.3-nompi_py38hfd55d45_105 --> 1.5.3-nompi_py38hfee4bf2_101
#  shapely                              1.7.0-py38hd168ffb_3 --> 1.6.4-py38hec07ddf_1006

conda install -c conda-forge fiona
#The following packages will be UPDATED:
#
#  gdal                                 2.4.3-py38h5f563d9_9 --> 3.0.2-py38hbb6b9fb_2
#  libgdal                                  2.4.3-h2f07a13_9 --> 3.0.2-hc7cfd23_2
#
#The following packages will be DOWNGRADED:
#
#  lz4-c                                    1.9.2-he1b5a44_1 --> 1.8.3-he1b5a44_1001
#  zstd                                     1.4.4-h6597ccf_3 --> 1.4.4-h3b9ef0a_2

conda install -c conda-forge geopandas

conda install -c conda-forge rasterio
#The following NEW packages will be INSTALLED:
#
#  affine             conda-forge/noarch::affine-2.3.0-py_0
#  rasterio           conda-forge/linux-64::rasterio-1.1.2-py38h900e953_0
#  snuggs             conda-forge/noarch::snuggs-1.4.7-py_0

conda install -c conda-forge rasterstats
#The following NEW packages will be INSTALLED:
#
#  rasterstats        conda-forge/noarch::rasterstats-0.14.0-py_0
#  simplejson         conda-forge/linux-64::simplejson-3.17.0-py38h1e0a361_1

conda install -c conda-forge gazar
#The following NEW packages will be INSTALLED:
#
#  appdirs            conda-forge/noarch::appdirs-1.4.3-py_1
#  brotlipy           conda-forge/linux-64::brotlipy-0.7.0-py38h1e0a361_1000
#  cffi               pkgs/main/linux-64::cffi-1.14.0-py38he30daa8_1
#  chardet            conda-forge/linux-64::chardet-3.0.4-py38h32f6830_1006
#  cryptography       conda-forge/linux-64::cryptography-2.9.2-py38h766eaa4_0
#  gazar              conda-forge/noarch::gazar-0.0.4-py_1
#  idna               conda-forge/noarch::idna-2.9-py_1
#  mapkit             conda-forge/noarch::mapkit-1.2.6-py_1
#  pycparser          conda-forge/noarch::pycparser-2.20-py_0
#  pyopenssl          conda-forge/noarch::pyopenssl-19.1.0-py_1
#  pysocks            conda-forge/linux-64::pysocks-1.7.1-py38h32f6830_1
#  requests           conda-forge/noarch::requests-2.23.0-pyh8c360ce_2
#  sqlalchemy         conda-forge/linux-64::sqlalchemy-1.3.17-py38h1e0a361_0
#  urllib3            conda-forge/noarch::urllib3-1.25.9-py_0
#  utm                conda-forge/noarch::utm-0.5.0-py_0

conda install -c conda-forge pandas

