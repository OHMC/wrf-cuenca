import argparse
import datetime
import os
import time
from threading import Thread
# from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ray
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray
import xarray as xr
from gazar.grid import ArrayGrid
from matplotlib.colors import LinearSegmentedColormap
from osgeo import osr, gdalconst
from rasterstats import zonal_stats

import pangaea_lib as pa
from config.constants import PROG_VERSION, COLUM_REPLACE, RAY_ADDRESS
from config.logging_conf import CUENCAS_LOGGER_NAME, get_logger_from_config_file

ray.init(address=RAY_ADDRESS)
logger = get_logger_from_config_file(CUENCAS_LOGGER_NAME)
pa.register()  # Solo para utilizar el import y que se registre el accessor en xarray


def corregir_wrfout(ruta_wrfout: str) -> (datetime.datetime, xarray.Dataset):
    """Fixes variables dimensions
        Parameters
        ----------
        ruta_wrfout: str
            route to the nc file to fix

    """
    xds = xr.open_dataset(ruta_wrfout)
    variables = ['XLAT', 'XLONG', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V']
    if len(xds.coords['XLAT'].shape) > 2:
        for var in variables:
            xds.coords[var] = xds.coords[var].mean(axis=0)
    rundate = datetime.datetime.strptime(xds.START_DATE, '%Y-%m-%d_%H:%M:%S')
    return rundate, xds


def to_projection(_plsm, variable) -> xr.Dataset:
    """Convert Grid to New Projection.
        Parameters
        ----------
        _plsm
        variable: :obj:`str`
            Name of variable in dataset.
    """
    projection = osr.SpatialReference()
    projection.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

    new_data = []
    ggrid = None
    for band in range(_plsm.xarr_obj.dims[_plsm.time_dim]):
        ar_gd = ArrayGrid(in_array=_plsm.xarr_obj[variable][band].values[::-1, :],
                          wkt_projection=_plsm.projection.ExportToWkt(),
                          geotransform=_plsm.geotransform)
        ggrid = ar_gd.to_projection(projection, gdalconst.GRA_Average)
        new_data.append(ggrid.np_array())

    _plsm.to_datetime()
    return _plsm.export_dataset(variable, np.array(new_data), ggrid)


@ray.remote
def guardar_tif(vari: xr.Dataset, arr: np.ndarray, out_path: str):
    nw_ds = rasterio.open(out_path, 'w', driver='GTiff',
                          height=arr.shape[0],
                          width=arr.shape[1],
                          count=1, dtype=str(arr.dtype),
                          crs=vari.lsm.projection.ExportToWkt(),
                          transform=vari.lsm.affine)
    nw_ds.write(arr, 1)
    nw_ds.close()


@ray.remote
def convertir_variable(plsm: xr.Dataset, variable: str) -> xr.Dataset:
    pa.register()
    vari = to_projection(plsm.lsm, variable)
    vari['lat'] = vari['lat'].sel(x=1)
    vari['lon'] = vari['lon'].sel(y=1)
    vari = vari.rename({'lat': 'y', 'lon': 'x'})
    return vari


def genear_tif_prec(plsm: xr.Dataset, out_path: str = None):
    if out_path is None:
        out_path = f"geotiff/ppn_{plsm.START_DATE[:-6]}"
    plsm.variables['RAINNC'].values = plsm.variables['RAINNC'].values + 1000
    plsm.variables['RAINC'].values = plsm.variables['RAINC'].values + 1000
    rainnc_id = convertir_variable.remote(plsm, 'RAINNC')
    rainc_id = convertir_variable.remote(plsm, 'RAINC')
    rainnc = ray.get(rainnc_id)
    rainc = ray.get(rainc_id)
    arrs = {}
    for t in range(len(plsm.coords['Time'])):
        arrs[t] = rainnc.RAINNC[t].values[:, :] + rainc.RAINC[t].values[:, :]
        arrs[t][arrs[t] == 0] = np.nan
    gtiff_id_list = []
    for t in range(1, len(plsm.coords['Time'])):
        gtiff_id_list.append(guardar_tif.remote(rainnc_id, arrs[t] - arrs[t - 1], f"{out_path}_{t}.tif"))
    gtiff_id_list.append(guardar_tif.remote(rainnc_id, arrs[33] - arrs[9], f"{out_path}.tif"))
    for g_id in gtiff_id_list:
        ray.get(g_id)



def integrar_en_cuencas(cuencas_shp: str) -> gpd.GeoDataFrame:
    cuencas_gdf: gpd.GeoDataFrame = gpd.read_file(cuencas_shp)
    df_zonal_stats = pd.DataFrame(zonal_stats(cuencas_shp, "geotiff/ppn.tif"))

    cuencas_gdf_ppn = pd.concat([cuencas_gdf, df_zonal_stats], axis=1).dropna(subset=['mean'])
    cuencas_gdf_ppn = cuencas_gdf_ppn.rename(columns=COLUM_REPLACE)
    return cuencas_gdf_ppn[['subcuenca', 'cuenca', 'geometry', 'count', 'max', 'mean', 'min']]


def generar_imagen(cuencas_gdf_ppn: gpd.GeoDataFrame, outdir: str, rundate: datetime.datetime, configuracion: str):
    path = (outdir + rundate.strftime('%Y_%m/%d/') + 'cuencas_' + configuracion + '.png')
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    f, ax = plt.subplots(1, figsize=(9, 12), frameon=False)
    cm_riesgos = LinearSegmentedColormap.from_list(
        'cmap_name',
        [
            (200 / 256, 255 / 256, 200 / 256),
            (255 / 256, 255 / 256, 0 / 256),
            (256 / 256, 0, 0)
        ],
        N=10
    )
    cuencas_gdf_ppn.dropna(subset=['mean']).plot(
        column='mean',
        vmin=0,
        vmax=100,
        edgecolor='#FFFFFF',
        linewidth=0.2,
        cmap=cm_riesgos,
        legend=False,
        ax=ax
    )
    gdf_cba = gpd.read_file('shapefiles/dep.shp')
    gdf_cba = gdf_cba[gdf_cba.PROVINCIA == 'CORDOBA']
    gdf_cba.plot(color='None', edgecolor='#333333', alpha=0.3, linewidth=0.5, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.savefig(path, bbox_inches='tight')


def guardar_tabla(cuencas_gdf_ppn: gpd.GeoDataFrame, outdir: str, rundate: datetime.datetime, configuracion: str):
    path = f"{outdir}{rundate.strftime('%Y_%m/%d/cordoba/')}cuencas_{configuracion}.csv"
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    cuencas_gdf_ppn = cuencas_gdf_ppn[['subcuenca', 'cuenca', 'count', 'max', 'mean', 'min']]
    cuencas_gdf_ppn = cuencas_gdf_ppn.round(2)
    cuencas_gdf_ppn.to_csv(path, index=False, mode='a')


@ray.remote
def tabla_por_hora(gdf_path, tabla_path, d_range, gdf_index, drop_na, c_rename=''):
    if drop_na:
        cuencas_gdf = gpd.read_file(gdf_path).dropna(subset=[gdf_index])
    else:
        cuencas_gdf = gpd.read_file(gdf_path)
    if c_rename:
        cuencas_gdf = cuencas_gdf.rename(columns=c_rename)

    cuencas_gdf = cuencas_gdf.rename(columns=COLUM_REPLACE)
    tabla_hora = pd.DataFrame(columns=cuencas_gdf[gdf_index], index=d_range)
    tabla_hora.index.name = 'fecha'

    for i in range(1, len(tabla_hora)):
        df_zonal_stats = pd.DataFrame(zonal_stats(cuencas_gdf, f"geotiff/ppn_{i}.tif"))
        cuencas_gdf_concat = pd.concat([cuencas_gdf[gdf_index], df_zonal_stats['mean']], axis=1)
        cuencas_gdf_concat = cuencas_gdf_concat.dropna(subset=['mean']).set_index(gdf_index)
        tabla_hora.iloc[i] = cuencas_gdf_concat['mean']

    tabla_hora = tabla_hora.astype(float).round(2)
    tabla_hora.index = tabla_hora.index + datetime.timedelta(hours=-3)
    tabla_hora.to_csv(tabla_path)
    return True


def generar_tabla_por_hora(outdir: str, rundate: datetime.datetime, configuracion: str):
    rundate_str = rundate.strftime('%Y_%m/%d')
    path_dict = {
        'base': Path(f"{outdir}{rundate_str}/cordoba/cuencas/ppn_por_hora_{configuracion}.csv"),
        'la_quebrada': Path(f"{outdir}{rundate_str}/cordoba/cuencas/la_quebrada/ppn_por_hora_lq_{configuracion}.csv"),
        'san_antonio': Path(f"{outdir}{rundate_str}/cordoba/cuencas/san_antonio/ppn_por_hora_sa_{configuracion}.csv")
    }

    for p in path_dict.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    d_range = pd.date_range(start=rundate, end=(rundate + datetime.timedelta(hours=48 + 9)), freq='H')

    base_shp = 'shapefiles/Cuencas hidrográficas.shp'
    # cuenca san antonio
    sa_shp = 'shapefiles/cuencas_sa.shp'
    # cuenca la quebrada
    lq_shp = 'shapefiles/cuenca_lq.shp'
    t_list = [
        tabla_por_hora.remote(base_shp, path_dict['base'], d_range, 'subcuenca', False, COLUM_REPLACE),
        tabla_por_hora.remote(lq_shp, path_dict['la_quebrada'], d_range, 'NAME', True),
        tabla_por_hora.remote(sa_shp, path_dict['san_antonio'], d_range, 'NAME', True)
    ]
    for t in t_list:
        ray.get(t)


def generar_producto_cuencas(wrfout, outdir_productos, outdir_tabla, configuracion):
    start = time.time()
    rundate, xds = corregir_wrfout(wrfout)
    print(f"Tiempo corregir_wrfout = {time.time() - start}")
    # nc = netCDF4.Dataset(wrfout)
    # xds.lsm.rainc = wrf.getvar(nc, 'RAINC', timeidx=wrf.ALL_TIMES)
    # xds.lsm.rainnc = wrf.getvar(nc, 'RAINNC', timeidx=wrf.ALL_TIMES)
    start = time.time()
    genear_tif_prec(xds, out_path='geotiff/ppn')
    xds.close()
    print(f"Tiempo genear_tif_prec = {time.time() - start}")
    # nc.close()
    start = time.time()
    cuencas_gdf_ppn: gpd.GeoDataFrame = integrar_en_cuencas('shapefiles/cuencas.shp')
    print(f"Tiempo integrar_en_cuencas = {time.time() - start}")
    start = time.time()
    guardar_tabla(cuencas_gdf_ppn, outdir_tabla, rundate, configuracion)
    print(f"Tiempo guardar_tabla = {time.time() - start}")
    start = time.time()
    generar_tabla_por_hora(outdir_tabla, rundate, configuracion)
    print(f"Tiempo generar_tabla_por_hora = {time.time() - start}")

    start = time.time()
    generar_imagen(cuencas_gdf_ppn, outdir_productos, rundate, configuracion)
    print(f"Tiempo generar_imagen = {time.time() - start}")


def main():
    parser = argparse.ArgumentParser(prog="WRF Cuencas")
    parser.add_argument("wrfout", help="ruta al wrfout de la salida del WRF")
    parser.add_argument("outdir_productos", help="ruta donde se guardan los productos")
    parser.add_argument("outdir_tabla", help="ruta donde se guardan las tablas de datos")
    parser.add_argument("configuracion", help="configuracion de las parametrizaciones")
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {PROG_VERSION}')

    args = parser.parse_args()

    generar_producto_cuencas(args.wrfout, args.outdir_productos, args.outdir_tabla, args.configuracion)


if __name__ == "__main__":
    main()
