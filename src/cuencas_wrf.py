import argparse
import datetime
import os
import time

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
from config.constants import PROG_VERSION
from config.logging_conf import CUENCAS_LOGGER_NAME, get_logger_from_config_file

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


def guardar_tif(vari: xr.Dataset, arr: np.ndarray, out_path: str):
    nw_ds = rasterio.open(out_path, 'w', driver='GTiff',
                          height=arr.shape[0],
                          width=arr.shape[1],
                          count=1, dtype=str(arr.dtype),
                          crs=vari.lsm.projection.ExportToWkt(),
                          transform=vari.lsm.affine)
    nw_ds.write(arr, 1)
    nw_ds.close()


def convertir_variable(plsm: xr.Dataset, variable: str) -> xr.Dataset:
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
    rainnc = convertir_variable(plsm, 'RAINNC')
    rainc = convertir_variable(plsm, 'RAINC')
    arrs = {}
    for t in range(len(plsm.coords['Time'])):
        arrs[t] = rainnc.RAINNC[t].values[:, :] + rainc.RAINC[t].values[:, :]
        arrs[t][arrs[t] == 0] = np.nan
    for t in range(1, len(plsm.coords['Time'])):
        guardar_tif(rainnc, arrs[t] - arrs[t - 1], f"{out_path}_{t}.tif")
    guardar_tif(rainnc, arrs[33] - arrs[9], f"{out_path}.tif")


def integrar_en_cuencas(cuencas_shp: str) -> gpd.GeoDataFrame:
    cuencas_gdf: gpd.GeoDataFrame = gpd.read_file(cuencas_shp)
    df_zonal_stats = pd.DataFrame(zonal_stats(cuencas_shp, "geotiff/ppn.tif"))

    cuencas_gdf_ppn = pd.concat([cuencas_gdf, df_zonal_stats], axis=1)
    cuencas_gdf_ppn = cuencas_gdf_ppn.dropna(subset=['mean'])
    cuencas_gdf_ppn = cuencas_gdf_ppn.rename(columns={'Subcuenca': 'subcuenca', 'Cuenca': 'cuenca'})
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


def generar_tabla_por_hora(outdir: str, rundate: datetime.datetime, configuracion: str):
    path = (outdir + rundate.strftime('%Y_%m/%d/cordoba/cuencas/') + 'ppn_por_hora_' + configuracion + '.csv')
    path_sa = (outdir + rundate.strftime('%Y_%m/%d/cordoba/cuencas/') + 'san_antonio/ppn_por_hora_sa_'
               + configuracion + '.csv')  # san antonio
    path_lq = (outdir + rundate.strftime('%Y_%m/%d/cordoba/cuencas/') + 'la_quebrada/ppn_por_hora_lq_'
               + configuracion + '.csv')  # la quebrada

    try:
        os.makedirs(os.path.dirname(path_sa))
        os.makedirs(os.path.dirname(path_lq))
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    cuencas_gdf = gpd.read_file('shapefiles/Cuencas hidrográficas.shp')
    cuencas_gdf = cuencas_gdf.rename(columns={'Subcuenca': 'subcuenca', 'Cuenca': 'cuenca'})

    d_range = pd.date_range(start=rundate, end=(rundate + datetime.timedelta(hours=48 + 9)), freq='H')
    tabla_hora = pd.DataFrame(columns=cuencas_gdf.subcuenca, index=d_range)
    tabla_hora.index.name = 'fecha'
    # cuenca san antonio
    cuencas_gdf_sa = gpd.read_file('shapefiles/cuencas_sa.shp').dropna(subset=['NAME'])
    tabla_hora_sa = pd.DataFrame(columns=cuencas_gdf_sa.NAME, index=d_range)

    tabla_hora_sa.index.name = 'fecha'
    # cuenca la quebrada
    cuencas_gdf_lq = gpd.read_file('shapefiles/cuenca_lq.shp').dropna(subset=['NAME'])
    tabla_hora_lq = pd.DataFrame(columns=cuencas_gdf_lq.NAME, index=d_range)

    tabla_hora_lq.index.name = 'fecha'
    for i in range(1, len(tabla_hora)):
        cuencas_gdf = gpd.read_file('shapefiles/Cuencas hidrográficas.shp')
        cuencas_gdf_sa = gpd.read_file('shapefiles/cuencas_sa.shp').dropna(subset=['NAME'])
        cuencas_gdf_lq = gpd.read_file('shapefiles/cuenca_lq.shp').dropna(subset=['NAME'])
        # ToDo: Revisar
        df_zonal_stats = pd.DataFrame(zonal_stats(cuencas_gdf, f"geotiff/ppn_{i}.tif"))
        df_zonal_stats_sa = pd.DataFrame(zonal_stats(cuencas_gdf_sa, f"geotiff/ppn_{i}.tif"))
        df_zonal_stats_lq = pd.DataFrame(zonal_stats(cuencas_gdf_lq, f"geotiff/ppn_{i}.tif"))

        cuencas_gdf = cuencas_gdf.rename(columns={'Subcuenca': 'subcuenca', 'Cuenca': 'cuenca'})
        cuencas_gdf = pd.concat([cuencas_gdf['subcuenca'], df_zonal_stats['mean']], axis=1)
        cuencas_gdf = cuencas_gdf.dropna(subset=['mean']).set_index('subcuenca')
        tabla_hora.iloc[i] = cuencas_gdf['mean']
        # san antonio
        cuencas_gdf_sa = pd.concat([cuencas_gdf_sa['NAME'], df_zonal_stats_sa['mean']], axis=1)
        cuencas_gdf_sa = cuencas_gdf_sa.dropna(subset=['mean']).set_index('NAME')
        tabla_hora_sa.iloc[i] = cuencas_gdf_sa['mean']
        # la quebrada
        cuencas_gdf_lq = pd.concat([cuencas_gdf_lq['NAME'], df_zonal_stats_lq['mean']], axis=1)
        cuencas_gdf_lq = cuencas_gdf_lq.dropna(subset=['mean']).set_index('NAME')
        tabla_hora_lq.iloc[i] = cuencas_gdf_lq['mean']
    tabla_hora = tabla_hora.astype(float).round(2)
    tabla_hora.index = tabla_hora.index + datetime.timedelta(hours=-3)
    tabla_hora.to_csv(path)
    # san antonio
    tabla_hora_sa = tabla_hora_sa.astype(float).round(2)
    tabla_hora_sa.index = tabla_hora_sa.index + datetime.timedelta(hours=-3)
    tabla_hora_sa.to_csv(path_sa)

    # san antonio
    tabla_hora_lq = tabla_hora_lq.astype(float).round(2)
    tabla_hora_lq.index = tabla_hora_lq.index + datetime.timedelta(hours=-3)
    tabla_hora_lq.to_csv(path_lq)


def generar_producto_cuencas(wrfout, outdir_productos, outdir_tabla, configuracion):
    rundate, xds = corregir_wrfout(wrfout)
    # nc = netCDF4.Dataset(wrfout)
    # xds.lsm.rainc = wrf.getvar(nc, 'RAINC', timeidx=wrf.ALL_TIMES)
    # xds.lsm.rainnc = wrf.getvar(nc, 'RAINNC', timeidx=wrf.ALL_TIMES)

    genear_tif_prec(xds, out_path='geotiff/ppn')
    xds.close()
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
