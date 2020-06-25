import argparse
import datetime
import os
import pickle
import re
import time
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
from config.constants import PROG_VERSION, COLUM_REPLACE, RAY_ADDRESS, WRFOUT_REGEX, CUENCAS_API_PICKLE_PATH
from config.logging_conf import CUENCAS_LOGGER_NAME, get_logger_from_config_file
from config.wrf_api_constants import API_ROOT

ray.init(address=RAY_ADDRESS)
logger = get_logger_from_config_file(CUENCAS_LOGGER_NAME)
pa.register()  # Solo para utilizar el import y que se registre el accessor en xarray

cuencas_api_dict = {'meta': {}, 'csv': {'ppn_acum_diario': {}, 'ppn_por_hora': {}}}


def dump_cuencas_api_dict():
    with open(CUENCAS_API_PICKLE_PATH, mode='wb') as f:
        pickle.dump(cuencas_api_dict, f)


def corregir_wrfout(ruta_wrfout: str) -> xarray.Dataset:
    """
    Fixes variables dimensions, saves the fixed data to a new netCDF file and gets the run date of the desired
    wrfout file.
    Parameters
    ----------
    ruta_wrfout:
            route to the nc file to fix
    Returns
    -------
    Dataset
    """
    xds = xr.open_dataset(ruta_wrfout)
    variables = ['XLAT', 'XLONG', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V']
    if len(xds.coords['XLAT'].shape) > 2:
        for var in variables:
            xds.coords[var] = xds.coords[var].mean(axis=0)
    return xds


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


def genear_tif_prec(plsm: xr.Dataset, out_path: str):
    """
    This functions opens a wrfout NC file, gets the vars RAINNC
    and RAINC and saves them as a geotiff tile

    Parameters:
        plsm: path to the wrfout NC file
        out_path: path to the directory where to save
                        the geoptiffs
    """
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

    ray.get(gtiff_id_list)


def integrar_en_cuencas(cuencas_shp: str) -> gpd.GeoDataFrame:
    """
    This functions opens a geotiff with ppn data, converts to a raster,
    integrate the ppn into cuencas and returns a GeoDataFrame object.

    Parameters:
        cuencas_shp: Path to shapefile
    Returns:
        cuencas_gdf_ppn (GeoDataFrame): a geodataframe with cuerncas and ppn
    """
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
    """
    Generates a cvs from a GDF with Accumulated PPN and basins

    This functions gets a GeoDataFrame with PPN and basins informations
    and generates a CSV with that information.

    Parameters:
        cuencas_gdf_ppn (GDF): dataframe to be exported
        outdir (str): path to the out dir
        rundate (str): run date of wrfout
        configuracion (str): identifier of the wrf simulation
    """
    rundate_str = rundate.strftime('%Y_%m/%d')
    cuencas_api_dict['csv']['ppn_acum_diario']['path'] = f"{API_ROOT}/{rundate_str}/cordoba/cuencas_{configuracion}.csv"
    cuencas_api_dict['csv']['ppn_acum_diario']['is_image'] = False
    cuencas_api_dict['csv']['ppn_acum_diario']['acumulacion'] = "12:00:00"  # ToDo: Revisar
    path = Path(f"{outdir}{rundate_str}/cordoba/cuencas_{configuracion}.csv")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    cuencas_gdf_ppn = cuencas_gdf_ppn[['subcuenca', 'cuenca', 'count', 'max', 'mean', 'min']]
    cuencas_gdf_ppn = cuencas_gdf_ppn.round(2)
    cuencas_gdf_ppn.to_csv(path, index=False, mode='a')


@ray.remote
def tabla_por_hora(gdf_path, tabla_path, rundate, gdf_index, drop_na, c_rename=''):
    """
    Generates csv pear each basin
    This function opens the GeoTiff generated in genear_tif_prec().
    Then gets the accumulated PPN within an hour, and for each basin
    and stores in a GDF.
    This GDF is then exported to csv files

    Parameters:
        gdf_path (str): path to the geotiff
        tabla_path (str): path to the csv file
        rundate: date of the wrf runout
        gdf_index (str): columns to drop
        drop_na (bool): to drop or not to drop
        c_rename (str): rename column
    """
    if drop_na:
        cuencas_gdf = gpd.read_file(gdf_path).dropna(subset=[gdf_index])
    else:
        cuencas_gdf = gpd.read_file(gdf_path)
    if c_rename:
        cuencas_gdf = cuencas_gdf.rename(columns=c_rename)

    d_range = pd.date_range(start=rundate, end=(rundate + datetime.timedelta(hours=48 + 9)), freq='H')

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
    """ Generates csv pear each basin
    This function opens ppn data and shapefiles with basisn
    of Cordoba. Then calls a tabla_por_hora() to get ppn per hour and basin.

    Parameters:
        outdir (str): path where to store the generated files
        rundate (datetime): date of the wrf runout
        configuracion (str): wrf configuration
    """
    rundate_str = rundate.strftime('%Y_%m/%d')
    # Datos para api web
    cuencas_api_dict['csv']['ppn_por_hora']['path'] = f"{API_ROOT}/{rundate_str}/cordoba/cuencas/" \
                                                      f"ppn_por_hora_{configuracion}.csv"
    cuencas_api_dict['csv']['ppn_por_hora']['is_image'] = False
    cuencas_api_dict['csv']['ppn_por_hora']['acumulacion'] = "12:00:00"  # ToDo: Revisar
    path_dict = {
        'base': Path(f"{outdir}{rundate_str}/cordoba/cuencas/ppn_por_hora_{configuracion}.csv"),
        'la_quebrada': Path(f"{outdir}{rundate_str}/cordoba/cuencas/la_quebrada/ppn_por_hora_lq_{configuracion}.csv"),
        'san_antonio': Path(f"{outdir}{rundate_str}/cordoba/cuencas/san_antonio/ppn_por_hora_sa_{configuracion}.csv")
    }
    for p in path_dict.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    base_shp = 'shapefiles/Cuencas hidrográficas.shp'
    # cuenca san antonio
    sa_shp = 'shapefiles/cuencas_sa.shp'
    # cuenca la quebrada
    lq_shp = 'shapefiles/cuenca_lq.shp'
    rundate_id = ray.put(rundate)
    t_list = [
        tabla_por_hora.remote(base_shp, path_dict['base'], rundate_id, 'subcuenca', False, COLUM_REPLACE),
        tabla_por_hora.remote(lq_shp, path_dict['la_quebrada'], rundate_id, 'NAME', True),
        tabla_por_hora.remote(sa_shp, path_dict['san_antonio'], rundate_id, 'NAME', True)
    ]
    ray.get(t_list)


def get_configuracion(wrfout) -> (str, datetime.datetime):
    """Retorna la parametrizacion y el timestamp a partir del nombre del archivo wrfout"""
    m = re.match(WRFOUT_REGEX, wrfout)
    if not m:
        logger.critical("No se pudo obtener la configuracion, proporcione una desde los parametros de ejecición.")
        raise ValueError
    m_dict = m.groupdict()
    param = m_dict.get('param')
    timestamp = datetime.datetime.strptime(m_dict.get('timestamp'), '%Y-%m-%d_%H:%M:%S')
    return param, timestamp


def generar_producto_cuencas(wrfout, outdir_productos, outdir_tabla, configuracion=None):
    """
    This is the main functions and shoudl be called when you want to generates cuencas product from other Python script.
    """
    wrfout_path = Path(wrfout)
    param, rundate = get_configuracion(wrfout_path.name)
    # noinspection PyTypeChecker
    cuencas_api_dict['meta']['timestamp'] = rundate
    # noinspection PyTypeChecker
    cuencas_api_dict['meta']['param'] = param
    if not configuracion:
        configuracion = f"CBA_{param}_{rundate.hour:02d}"
    start = time.time()
    xds = corregir_wrfout(wrfout_path)
    logger.info(f"Tiempo corregir_wrfout = {time.time() - start}")
    start = time.time()
    genear_tif_prec(xds, out_path='geotiff/ppn')
    xds.close()
    logger.info(f"Tiempo genear_tif_prec = {time.time() - start}")
    # nc.close()
    start = time.time()
    cuencas_gdf_ppn: gpd.GeoDataFrame = integrar_en_cuencas('shapefiles/cuencas.shp')
    logger.info(f"Tiempo integrar_en_cuencas = {time.time() - start}")
    start = time.time()
    guardar_tabla(cuencas_gdf_ppn, outdir_tabla, rundate, configuracion)
    logger.info(f"Tiempo guardar_tabla = {time.time() - start}")
    start = time.time()
    generar_tabla_por_hora(outdir_tabla, rundate, configuracion)
    logger.info(f"Tiempo generar_tabla_por_hora = {time.time() - start}")

    start = time.time()
    generar_imagen(cuencas_gdf_ppn, outdir_productos, rundate, configuracion)
    logger.info(f"Tiempo generar_imagen = {time.time() - start}")
    dump_cuencas_api_dict()


def main():
    parser = argparse.ArgumentParser(prog="WRF Cuencas")
    parser.add_argument("wrfout", help="ruta al wrfout de la salida del WRF")
    parser.add_argument("outdir_productos", help="ruta donde se guardan los productos")
    parser.add_argument("outdir_tabla", help="ruta donde se guardan las tablas de datos")
    parser.add_argument("-c", "--configuracion", help="configuracion de las parametrizaciones", default='')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {PROG_VERSION}')

    args = parser.parse_args()

    generar_producto_cuencas(args.wrfout, args.outdir_productos, args.outdir_tabla, args.configuracion)


if __name__ == "__main__":
    main()
