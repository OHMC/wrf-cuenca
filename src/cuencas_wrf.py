import argparse
import datetime
import os
import pickle
import re
import time
import ray
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import fiona
import shapely.geometry as sgeom
from pathlib import Path
from affine import Affine
from matplotlib.colors import LinearSegmentedColormap
from osgeo import osr, gdal, gdal_array
from rasterstats import zonal_stats
import cartopy.crs as ccrs

from config.constants import (PROG_VERSION, COLUM_REPLACE, RAY_ADDRESS,
                              WRFOUT_REGEX, CUENCAS_API_PICKLE_PATH,
                              CBA_EXTENT, WRF_EXTENT, KM_PER_DEGREE,
                              RECORTE_EXTENT, CLEVS, PRECIP_COLORMAP,
                              RESOLUTION)
from config.logging_conf import (CUENCAS_LOGGER_NAME,
                                 get_logger_from_config_file)
from config.wrf_api_constants import API_ROOT

ray.init(address=RAY_ADDRESS)
logger = get_logger_from_config_file(CUENCAS_LOGGER_NAME)

cuencas_api_dict = {"24": {'meta': {}, 'csv': {'ppn_acum_diario': {}, 'ppn_por_hora': {}}},
                    "36": {'meta': {}, 'csv': {'ppn_acum_diario': {}}},
                    "48": {'meta': {}, 'csv': {'ppn_acum_diario': {}}}}


def dump_cuencas_api_dict():
    with open(CUENCAS_API_PICKLE_PATH, mode='wb') as f:
        pickle.dump(cuencas_api_dict, f)


def getGeoT(extent, nlines, ncols):
    # Compute resolution based on data dimension
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0], resx, 0, extent[3], 0, -resy]


# TodDo: ray to this
def cambiar_projection_ppn(in_array: np.ndarray):
    """Convert Grid to New Projection.
        Parameters
        ----------
        in_array

    """
    # WRF Spatial Reference System
    source_prj = osr.SpatialReference()
    source_prj.ImportFromProj4('+proj=lcc +lat_0=-32.500008 +lon_0=-62.7 '
                               '+lat_1=-60 +lat_2=-30 +x_0=0 +y_0=0 +R=6370000'
                               ' +units=m +no_defs')
    # Lat/lon WSG84 Spatial Reference System
    target_prj = osr.SpatialReference()
    target_prj.ImportFromProj4('+proj=longlat +ellps=WGS84 '
                               '+datum=WGS84 +no_defs')

    # se configura la matriz destino
    sizex = int(((CBA_EXTENT[2] - CBA_EXTENT[0]) * KM_PER_DEGREE) / RESOLUTION)
    sizey = int(((CBA_EXTENT[3] - CBA_EXTENT[1]) * KM_PER_DEGREE) / RESOLUTION)

    out_array = np.ndarray(shape=(len(in_array.keys()), sizey, sizex))

    for t in in_array.keys():
        # loar gdal array y se le asigna la projección y transofrmación
        raw = gdal_array.OpenArray(np.flipud(in_array[t]))
        raw.SetProjection(source_prj.ExportToWkt())
        raw.SetGeoTransform(getGeoT(WRF_EXTENT,
                                    raw.RasterYSize,
                                    raw.RasterXSize))

        grid = gdal.GetDriverByName('MEM').Create("tmp_ras",
                                                  sizex, sizey, 1,
                                                  gdal.GDT_Float32)
        # Setup projection and geo-transformation
        grid.SetProjection(target_prj.ExportToWkt())
        grid.SetGeoTransform(getGeoT(CBA_EXTENT,
                                     grid.RasterYSize,
                                     grid.RasterXSize))

        # reprojectamos
        gdal.ReprojectImage(raw,
                            grid,
                            source_prj.ExportToWkt(),
                            target_prj.ExportToWkt(),
                            gdal.GRA_NearestNeighbour,
                            options=['NUM_THREADS=ALL_CPUS'])

        out_array[t] = grid.ReadAsArray()

    return out_array, grid.GetGeoTransform(), grid.GetProjection()


@ray.remote
def guardar_tif(geoTransform: list, target_prj: str,
                arr: np.ndarray, out_path: str):
    nw_ds = rasterio.open(out_path, 'w', driver='GTiff',
                          height=arr.shape[0],
                          width=arr.shape[1],
                          count=1, dtype=str(arr.dtype),
                          crs=target_prj,
                          transform=Affine.from_gdal(*geoTransform))
    nw_ds.write(arr, 1)
    nw_ds.close()


def genear_img_prec(plsm: xr.Dataset, configuracion: str, out_path: str,
                    path_png: str):
    """
    This functions gest a xarray Dataset, gets the vars RAINNC
    and RAINC and saves them as a geotiff tile

    Parameters:
        plsm: path to the wrfout NC file
        out_path: path to the directory where to save
                        the geoptiffs
    """
    try:
        os.makedirs(os.path.dirname(out_path))
    except OSError:
        pass

    arrs = {}
    for t in range(len(plsm.variables['Times'])):
        arrs[t] = (plsm.variables['RAINNC'].values[t, :, :] +
                   plsm.variables['RAINC'].values[t, :, :])

    out_ppn, geoTransform, target_prj = cambiar_projection_ppn(arrs)

    base_path = f"{out_path}{configuracion}_ppn"

    gtiff_id_list = []
    for t in range(1, out_ppn.shape[0]):
        gtiff_id_list.append(guardar_tif.remote(geoTransform, target_prj,
                                                out_ppn[t, :, :] -
                                                out_ppn[t - 1, :, :],
                                                f"geotiff/ppn_{t}.tif"))

    gtiff_id_list.append(guardar_tif.remote(geoTransform, target_prj,
                                            out_ppn[33] - out_ppn[9],
                                            f"{base_path}.tif"))
    gtiff_id_list.append(guardar_tif.remote(geoTransform, target_prj,
                                            out_ppn[45] - out_ppn[9],
                                            f"{base_path}_36.tif"))
    gtiff_id_list.append(guardar_tif.remote(geoTransform, target_prj,
                                            out_ppn[57] - out_ppn[9],
                                            f"{base_path}_48.tif"))

    ray.get(gtiff_id_list)

    gen_png_prec(plsm, out_ppn[33] - out_ppn[9], path_png,
                 configuracion, '')
    gen_png_prec(plsm, out_ppn[45] - out_ppn[9], path_png,
                 configuracion, '_36')
    gen_png_prec(plsm, out_ppn[57] - out_ppn[9], path_png,
                 configuracion, '_48')


def gen_png_prec(plsm: xr.Dataset, arr: np.ndarray, png_path: str,
                 configuracion: str, accum: str):
    """
    Plots de precipitacion acumulada para cuencas

    Parameters: 
        plsm: xarray del nc
        arr: array de acumulacion de precipitacion
        png_path: rundate donde guardar los png
        accum: string para 24, 36 o 48hs
    """
    try:
        os.makedirs(os.path.dirname(png_path))
    except OSError:
        pass

    deltax = -(CBA_EXTENT[0] - CBA_EXTENT[2])/(arr.shape[0]*arr.shape[1])
    deltay = -(CBA_EXTENT[1] - CBA_EXTENT[3])/(arr.shape[0]*arr.shape[1])

    # Compute the lon/lat coordinates with rasterio.warp.transform
    ny, nx = (arr.shape[0],arr.shape[1])
    lon = np.arange(CBA_EXTENT[0], CBA_EXTENT[2], deltax)
    lat = np.flip(np.arange(CBA_EXTENT[1], CBA_EXTENT[3], deltay))

    # Rasterio works with 1D arrays
    lon = np.asarray(lon).reshape((ny, nx))
    lat = np.asarray(lat).reshape((ny, nx))

    plsm.coords['lon'] = (('y', 'x'), lon)
    plsm.coords['lat'] = (('y', 'x'), lat)

    # Plot on a map
    plt.figure(figsize=(6, 8), frameon=False)
    ax = plt.subplot(projection=ccrs.PlateCarree())

    norm = mpl.colors.BoundaryNorm(CLEVS, len(CLEVS))

    cba_extent = [CBA_EXTENT[0], CBA_EXTENT[2], CBA_EXTENT[1], CBA_EXTENT[3]]
    img_plot = ax.imshow(np.flipud(arr), origin='upper', extent=cba_extent,
                         cmap=PRECIP_COLORMAP, norm=norm,
                         transform=ccrs.PlateCarree())

    ax.set_extent([RECORTE_EXTENT[0], RECORTE_EXTENT[2],
                   RECORTE_EXTENT[1], RECORTE_EXTENT[3]])
    # Abro archivo shapefile con departamentos
    shpfile = "shapefiles/dep.shp"
    with fiona.open(shpfile) as records:
        geometries = [sgeom.shape(shp['geometry'])
                      for shp in records]
    # Agrego a la figura cada uno de los departamentos
    ax.add_geometries(geometries, ccrs.PlateCarree(),
                      edgecolor='slategrey', facecolor='none', linewidth=0.35)

    # Abro archivo shapefile con cuencas
    shpfile = "shapefiles/cuencas.shp"
    with fiona.open(shpfile) as records:
        geometries = [sgeom.shape(shp['geometry'])
                      for shp in records]
    # Agrego a la figura cada uno de los departamentos
    ax.add_geometries(geometries, ccrs.PlateCarree(),
                      edgecolor='lightgrey', facecolor='none', linewidth=0.2)

    gl = ax.gridlines( draw_labels=True, alpha=0.5)

    plt.savefig(f'{png_path}ppn{configuracion}{accum}.png',
                bbox_inches='tight', dpi=160, pad_inches=0)


def integrar_en_cuencas(cuencas_shp: str, out_path: str,
                        configuracion: str) -> gpd.GeoDataFrame:
    """
    This functions opens a geotiff with ppn data, converts to a raster,
    integrate the ppn into cuencas and returns a GeoDataFrame object.

    Parameters:
        cuencas_shp: Path to shapefile
    Returns:
        cuencas_gdf_ppn (GeoDataFrame): a geodataframe with cuerncas and ppn
    """
    base_path = f"{out_path}{configuracion}_ppn"

    cuencas_gdf: gpd.GeoDataFrame = gpd.read_file(cuencas_shp)
    df_zs = pd.DataFrame(zonal_stats(cuencas_shp, f"{base_path}.tif"))
    df_zs_36 = pd.DataFrame(zonal_stats(cuencas_shp, f"{base_path}_36.tif"))
    df_zs_48 = pd.DataFrame(zonal_stats(cuencas_shp, f"{base_path}_48.tif"))

    df_zs_36 = df_zs_36.rename(columns={"mean": "mean_36",
                                        "max": "max_36",
                                        "min": "min_36"})
    df_zs_48 = df_zs_48.rename(columns={"mean": "mean_48",
                                        "max": "max_48",
                                        "min": "min_48"})

    cuencas_gdf_ppn = pd.concat([cuencas_gdf, df_zs,
                                 df_zs_36['mean_36'], df_zs_36['max_36'],
                                 df_zs_36['min_36'], df_zs_48['mean_48'],
                                 df_zs_48['max_48'], df_zs_48['min_48']],
                                axis=1).dropna(subset=['mean'])

    cuencas_gdf_ppn = cuencas_gdf_ppn.rename(columns=COLUM_REPLACE)

    return cuencas_gdf_ppn[['subcuenca', 'cuenca', 'geometry', 'count',
                            'max', 'min', 'mean', 'max_36', 'min_36', 'mean_36',
                            'max_48', 'min_48', 'mean_48']]


def generar_imagen(cuencas_gdf_ppn: gpd.GeoDataFrame, outdir: str,
                   rundate: datetime.datetime, configuracion: str):
    path = (f"{outdir }cuencas_{configuracion}")

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

    for hour in ('', '_36', '_48'):
        cuencas_gdf_ppn.dropna(subset=[f'mean{hour}']).plot(
            column=f'mean{hour}',
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
        gdf_cba.plot(color='None', edgecolor='#333333', alpha=0.3,
                     linewidth=0.5, ax=ax)
        ax.set_axis_off()
        plt.axis('equal')
        plt.savefig(f'{path}{hour}.png', bbox_inches='tight')


def guardar_tabla(cuencas_gdf_ppn: gpd.GeoDataFrame, outdir: str,
                  rundate: datetime.datetime, configuracion: str):
    """
    Generates a cvs from a GDF with Accumulated PPN and basins

    This functions gets a GeoDataFrame with PPN and basin
        out_ppn[t] = out_ppn[t] + 1000
        out_ppn[t][out_ppn[t] == 0] = np.nan
    Parameters:
        cuencas_gdf_ppn (GDF): dataframe to be exported
        outdir (str): path to the out dir
        rundate (str): run date of wrfout
        configuracion (str): identifier of the wrf simulation
    """
    rundate_str = rundate.strftime('%Y_%m/%d')
    cuencas_api_dict['24']['csv']['ppn_acum_diario']['path'] = f"{API_ROOT}/{rundate_str}/cordoba/cuencas_{configuracion}.csv"

    path = Path(f"{outdir}{rundate_str}/cordoba/cuencas_{configuracion}.csv")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    cuencas_gdf_ppn = cuencas_gdf_ppn[['subcuenca', 'cuenca', 'count',
                                       'max', 'mean', 'min',
                                       'max_36', 'mean_36', 'min_36',
                                       'max_48', 'mean_48', 'min_48']]

    cuencas_gdf_ppn_24 = cuencas_gdf_ppn[['subcuenca', 'cuenca', 'count',
                                          'max', 'mean', 'min']]
    cuencas_gdf_ppn_24 = cuencas_gdf_ppn_24.round(2)
    cuencas_gdf_ppn_24.to_csv(path, index=False, mode='a')

    cuencas_api_dict['36']['csv']['ppn_acum_diario']['path'] = f"{API_ROOT}/{rundate_str}/cordoba/cuencas_{configuracion}_36.csv"

    path_36 = Path(f"{outdir}{rundate_str}/cordoba/cuencas_{configuracion}_36.csv")

    cuencas_gdf_ppn_36 = cuencas_gdf_ppn[['subcuenca', 'cuenca', 'count',
                                          'max_36', 'mean_36', 'min_36']]
    cuencas_gdf_ppn_36 = cuencas_gdf_ppn_36.rename(columns={"mean_36": "mean",
                                                            "max_36": "max",
                                                            "min_36": "min"})
    cuencas_gdf_ppn_36 = cuencas_gdf_ppn.round(2)
    cuencas_gdf_ppn_36.to_csv(path_36, index=False, mode='a')

    cuencas_api_dict['48']['csv']['ppn_acum_diario']['path'] = f"{API_ROOT}/{rundate_str}/cordoba/cuencas_{configuracion}_48.csv"

    path_48 = Path(f"{outdir}{rundate_str}/cordoba/cuencas_{configuracion}_48.csv")

    cuencas_gdf_ppn_48 = cuencas_gdf_ppn[['subcuenca', 'cuenca', 'count',
                                          'max_48', 'mean_48', 'min_48']]
    cuencas_gdf_ppn_48 = cuencas_gdf_ppn_36.rename(columns={"mean_48": "mean",
                                                            "max_48": "max",
                                                            "min_48": "min"})
    cuencas_gdf_ppn_48 = cuencas_gdf_ppn.round(2)
    cuencas_gdf_ppn_48.to_csv(path_48, index=False, mode='a')


def tabla_por_hora(gdf_path: str, tabla_path: str, rundate: datetime.datetime,
                   gdf_index, drop_na, configuracion: str, c_rename=''):
    """
    Generates csv pear each basin
    This function opens the GeoTiff generated in genear_tif_prec().
    Then gets the accumulated PPN within an hour, and for each basin
    and stores in a GDF.
    This GDF is then exported to csv files

    Parameters:
        gdf_path (str): path to the shapefile
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

    d_range = pd.date_range(start=rundate,
                            end=(rundate + datetime.timedelta(hours=48 + 9)),
                            freq='H')

    cuencas_gdf = cuencas_gdf.rename(columns=COLUM_REPLACE)
    tabla_hora = pd.DataFrame(columns=cuencas_gdf[gdf_index], index=d_range)
    tabla_hora.index.name = 'fecha'

    for i in range(1, len(tabla_hora)):
        df_zonal_stats = pd.DataFrame(zonal_stats(cuencas_gdf,
                                                  f"geotiff/ppn_{i}.tif"))
        cuencas_gdf_concat = pd.concat([cuencas_gdf[gdf_index],
                                        df_zonal_stats['mean']],
                                       axis=1)
        cuencas_gdf_concat = cuencas_gdf_concat.dropna(subset=['mean']).set_index(gdf_index)
        tabla_hora.iloc[i] = cuencas_gdf_concat['mean']

    tabla_hora = tabla_hora.astype(float).round(2)
    tabla_hora.index = tabla_hora.index + datetime.timedelta(hours=-3)
    tabla_hora.to_csv(tabla_path)
    return True


def generar_tabla_por_hora(outdir: str, rundate: datetime.datetime,
                           shapefile: str, param: str, path_gtif: str,
                           configuracion: str):
    """ Generates csv pear each basin
    This function opens ppn data and shapefiles with basisn
    of Cordoba. Then calls a tabla_por_hora() to get ppn per hour and basin.

    Parameters:
        outdir (str): path where to store the generated files
        rundate (datetime): date of the wrf runout
        oaram (str): wrf configuration
    """
    rundate_str = rundate.strftime('%Y_%m/%d')
    # Datos para api web
    cuencas_api_dict['24']['csv']['ppn_por_hora']['path'] = (f"{API_ROOT}/{rundate_str}/cordoba/cuencas/"
                                                             f"ppn_por_hora_{param}.csv")

    path_dict = {
        'base': Path(f"{outdir}{rundate_str}/cordoba/cuencas/"
                     f"ppn_por_hora_{param}.csv"),
    }
    for p in path_dict.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    tabla_por_hora(shapefile,
                   path_dict['base'],
                   rundate,
                   'subcuenca',
                   False,
                   configuracion,
                   COLUM_REPLACE),


def get_configuracion(wrfout) -> (str, datetime.datetime):
    """Retorna la parametrizacion y el timestamp a partir del
    nombre del archivo wrfout"""
    m = re.match(WRFOUT_REGEX, wrfout)
    if not m:
        logger.critical("No se pudo obtener la configuracion, proporcione"
                        "una desde los parametros de ejecición.")
        raise ValueError
    m_dict = m.groupdict()
    param = m_dict.get('param')
    timestamp = datetime.datetime.strptime(m_dict.get('timestamp'),
                                           '%Y-%m-%d_%H:%M:%S')
    return param, timestamp


def generar_producto_cuencas(wrfout, outdir_productos, outdir_tabla,
                             configuracion=None):
    """
    This is the main functions and shoudl be called when you want to
    generates cuencas product from other Python script.
    """
    shapefile = 'shapefiles/cuencas_hidro_new.shp'
    wrfout_path = Path(wrfout)
    param, rundate = get_configuracion(wrfout_path.name)
    # noinspection PyTypeChecker
    cuencas_api_dict['24']['meta']['timestamp'] = rundate
    cuencas_api_dict['36']['meta']['timestamp'] = rundate
    cuencas_api_dict['48']['meta']['timestamp'] = rundate
    # noinspection PyTypeChecker
    cuencas_api_dict['24']['meta']['param'] = param
    cuencas_api_dict['36']['meta']['param'] = param
    cuencas_api_dict['48']['meta']['param'] = param
    if not configuracion:
        configuracion = f"CBA_{param}_{rundate.hour:02d}"
    start = time.time()
    xds = xr.open_dataset(wrfout_path)
    logger.info(f"Tiempo corregir_wrfout = {time.time() - start}")
    start = time.time()
    path_gtiff = (f'{outdir_productos}/geotiff/'
                  f'{rundate.strftime("%Y_%m/%d/")}')
    path_png = (f'{outdir_productos}/plots/'
                f'{rundate.strftime("%Y_%m/%d/")}')
    genear_img_prec(xds, configuracion, path_gtiff, path_png)
    xds.close()
    logger.info(f"Tiempo genear_img_prec = {time.time() - start}")
    # nc.close()
    start = time.time()
    cuencas_gdf_ppn: gpd.GeoDataFrame = integrar_en_cuencas(shapefile,
                                                            path_gtiff,
                                                            configuracion)
    logger.info(f"Tiempo integrar_en_cuencas = {time.time() - start}")
    start = time.time()
    guardar_tabla(cuencas_gdf_ppn, outdir_tabla, rundate, configuracion)
    logger.info(f"Tiempo guardar_tabla = {time.time() - start}")
    start = time.time()
    generar_tabla_por_hora(outdir_tabla, rundate, shapefile, param,
                           path_gtiff, configuracion)
    logger.info(f"Tiempo generar_tabla_por_hora = {time.time() - start}")

    start = time.time()
    generar_imagen(cuencas_gdf_ppn, path_png, rundate, configuracion)
    logger.info(f"Tiempo generar_imagen = {time.time() - start}")
    dump_cuencas_api_dict()


def main():
    parser = argparse.ArgumentParser(prog="WRF Cuencas")
    parser.add_argument("wrfout",
                        help="ruta al wrfout de la salida del WRF")
    parser.add_argument("outdir_productos",
                        help="ruta donde se guardan los productos")
    parser.add_argument("outdir_tabla",
                        help="ruta donde se guardan las tablas de datos")
    parser.add_argument("-c", "--configuracion",
                        help="configuracion de las parametrizaciones",
                        default='')
    parser.add_argument('-v', '--version',
                        action='version', version=f'%(prog)s {PROG_VERSION}')

    args = parser.parse_args()

    generar_producto_cuencas(args.wrfout, args.outdir_productos,
                             args.outdir_tabla, args.configuracion)


if __name__ == "__main__":
    main()
