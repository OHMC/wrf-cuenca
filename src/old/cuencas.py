import datetime
import os

from optparse import OptionParser

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray
import pandas as pd
import pangaea_lib as pa
import rasterio
import xarray as xr
from gazar.grid import ArrayGrid
from matplotlib.colors import LinearSegmentedColormap
from rasterstats import zonal_stats


def corregir_wrfout(ruta_wrfout):
    """Fixes variables dimensions
        Parameters
        ----------
        ruta_wrfout: str
            route to the nc file to fix
        Returns
        -------
        rundate

    """
    xds = xr.open_dataset(ruta_wrfout)
    variables = ['XLAT', 'XLONG', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V']
    for var in variables:
        if len(xds.coords['XLAT'].shape) > 2:
            xds.coords[var] = xds.coords[var].mean(axis=0)
    xds.to_netcdf(ruta_wrfout + '.nc')
    rundate = datetime.datetime.strptime(xds.START_DATE, '%Y-%m-%d_%H:%M:%S')
    xds.close()
    return rundate


def abrir_plsm(ruta_wrfout) -> xarray.Dataset:
    return pa.open_mfdataset(
        ruta_wrfout,
        lat_var='XLAT',
        lon_var='XLONG',
        time_var='XTIME',
        lat_dim='south_north',
        lon_dim='west_east',
        time_dim='Time'
    )


def to_projection(plsm, variable, projection=None):
    """Convert Grid to New Projection.
        Parameters
        ----------
        variable: :obj:`str`
            Name of variable in dataset.
        projection: :func:`osr.SpatialReference`
            Projection to convert data to.
        Returns
        -------
        :func:`xarray.Dataset`
    """
    if projection is None:
        from osgeo import osr, gdalconst
        projection = osr.SpatialReference()
        projection.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 \
                                    +no_defs")
    new_data = []
    for band in range(plsm._obj.dims[plsm.time_dim]):
        ar_gd = ArrayGrid(in_array=plsm._obj[variable][band].values[::-1, :],
                          wkt_projection=plsm.projection.ExportToWkt(),
                          geotransform=plsm.geotransform)
        ggrid = ar_gd.to_projection(projection, gdalconst.GRA_Average)
        new_data.append(ggrid.np_array())

    plsm.to_datetime()
    return plsm._export_dataset(variable, np.array(new_data), ggrid)


def guardar_tif(vari, arr, out_path):
    nw_ds = rasterio.open(out_path, 'w', driver='GTiff',
                          height=arr.shape[0],
                          width=arr.shape[1],
                          count=1, dtype=str(arr.dtype),
                          crs=vari.lsm.projection.ExportToWkt(),
                          transform=vari.lsm.affine)
    nw_ds.write(arr, 1)
    nw_ds.close()


def convertir_variable(plsm, variable):
    vari = to_projection(plsm.lsm, variable)
    vari['lat'] = vari['lat'].sel(x=1)
    vari['lon'] = vari['lon'].sel(y=1)
    vari = vari.rename({'lat': 'y', 'lon': 'x'})
    return vari


def genear_tif(ruta_wrfout, variable, time_idx, out_path):
    plsm: xarray.Dataset = abrir_plsm(ruta_wrfout)
    vari = convertir_variable(plsm, variable)
    arr = vari[variable][time_idx].values[:, :]
    arr[arr == 0] = np.nan
    guardar_tif(vari, arr, out_path)
    plsm.close()


def genear_tif_prec(ruta_wrfout, out_path=None):
    plsm = abrir_plsm(ruta_wrfout)
    if out_path is None:
        out_path = 'geotiff/ppn_' + plsm.START_DATE[:-6]
    plsm.variables['RAINNC'].values = plsm.variables['RAINNC'].values + 1000
    plsm.variables['RAINC'].values = plsm.variables['RAINC'].values + 1000
    rainnc = convertir_variable(plsm, 'RAINNC')
    rainc = convertir_variable(plsm, 'RAINC')
    arrs = {}
    for t in range(len(plsm.coords['time'])):
        arrs[t] = rainnc.RAINNC[t].values[:, :] + rainc.RAINC[t].values[:, :]
        arrs[t][arrs[t] == 0] = np.nan
    for t in range(1, len(plsm.coords['time'])):
        guardar_tif(rainnc, arrs[t] - arrs[t - 1], out_path + '_' + str(t)
                    + '.tif')
    guardar_tif(rainnc, arrs[33] - arrs[9], out_path + '.tif')
    plsm.close()


def integrar_en_cuencas(cuencas_gdf):
    with rasterio.open("geotiff/ppn.tif") as src:
        affine = src.transform
        array = src.read(1)
        df_zonal_stats = pd.DataFrame(zonal_stats(cuencas_gdf, array,
                                                  affine=affine, all_touched=True))

    cuencas_gdf_ppn = pd.concat([cuencas_gdf, df_zonal_stats], axis=1)
    cuencas_gdf_ppn = cuencas_gdf_ppn.dropna(subset=['mean'])
    cuencas_gdf_ppn = cuencas_gdf_ppn.rename(columns={'Subcuenca': 'subcuenca',
                                                      'Cuenca': 'cuenca'})
    return cuencas_gdf_ppn[['subcuenca', 'cuenca', 'geometry', 'count', 'max',
                            'mean', 'min']]


def generar_imagen(cuencas_gdf_ppn, outdir, rundate, configuracion):
    path = (outdir
            + rundate.strftime('%Y_%m/%d/')
            + 'cuencas_'
            + configuracion + '.png')
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
    cuencas_gdf_ppn.dropna(subset=['mean']).plot(column='mean',
                                                 vmin=0,
                                                 vmax=100,
                                                 edgecolor='#FFFFFF',
                                                 linewidth=0.2,
                                                 cmap=cm_riesgos,
                                                 legend=False,
                                                 ax=ax)
    gdf_cba = gpd.read_file('../wrfplot/shapefiles/dep.shp')
    gdf_cba = gdf_cba[gdf_cba.PROVINCIA == 'CORDOBA']
    gdf_cba.plot(color='None', edgecolor='#333333', alpha=0.3,
                 linewidth=0.5, ax=ax)
    ax.set_axis_off()
    plt.axis('equal')
    plt.savefig(path, bbox_inches='tight')


def guardar_tabla(cuencas_gdf_ppn, outdir, rundate, configuracion):
    path = (outdir + rundate.strftime('%Y_%m/%d/cordoba/') + 'cuencas_'
            + configuracion + '.csv')
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    cuencas_gdf_ppn = cuencas_gdf_ppn[['subcuenca', 'cuenca', 'count', 'max',
                                       'mean', 'min']]
    cuencas_gdf_ppn = cuencas_gdf_ppn.round(2)
    cuencas_gdf_ppn.to_csv(path, index=False, mode='a')


def generar_tabla_por_hora(outdir, rundate, configuracion):
    path = (outdir + rundate.strftime('%Y_%m/%d/cordoba/cuencas/')
            + 'ppn_por_hora_' + configuracion + '.csv')
    path_sa = (outdir + rundate.strftime('%Y_%m/%d/cordoba/cuencas/')
               + 'san_antonio/ppn_por_hora_sa_'
               + configuracion + '.csv')  # san antonio
    path_lq = (outdir + rundate.strftime('%Y_%m/%d/cordoba/cuencas/')
               + 'la_quebrada/ppn_por_hora_lq_'
               + configuracion + '.csv')  # la quebrada

    try:
        os.makedirs(os.path.dirname(path_sa))
        os.makedirs(os.path.dirname(path_lq))
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    cuencas_gdf = gpd.read_file('shapefiles/Cuencas hidrográficas.shp')
    cuencas_gdf = cuencas_gdf.rename(columns={'Subcuenca': 'subcuenca',
                                              'Cuenca': 'cuenca'})
    tabla_hora = pd.DataFrame(columns=cuencas_gdf.subcuenca,
                              index=pd.DatetimeIndex(start=rundate, end=(rundate + datetime.timedelta(hours=48 + 9)),
                                                     freq='H'))
    tabla_hora.index.name = 'fecha'
    # cuenca san antonio
    cuencas_gdf_sa = gpd.read_file('shapefiles/cuencas_sa.shp').dropna(
        subset=['NAME'])
    tabla_hora_sa = pd.DataFrame(columns=cuencas_gdf_sa.NAME,
                                 index=pd.DatetimeIndex(start=rundate, end=(rundate + datetime.timedelta(hours=48 + 9)),
                                                        freq='H'))

    tabla_hora_sa.index.name = 'fecha'
    # cuenca la quebrada
    cuencas_gdf_lq = gpd.read_file('shapefiles/cuenca_lq.shp').dropna(
        subset=['NAME'])
    tabla_hora_lq = pd.DataFrame(columns=cuencas_gdf_lq.NAME,
                                 index=pd.DatetimeIndex(start=rundate, end=(rundate + datetime.timedelta(hours=48 + 9)),
                                                        freq='H'))

    tabla_hora_lq.index.name = 'fecha'
    for i in range(1, len(tabla_hora)):
        cuencas_gdf = gpd.read_file('shapefiles/Cuencas hidrográficas.shp')
        cuencas_gdf_sa = gpd.read_file('shapefiles/cuencas_sa.shp').dropna(
            subset=['NAME'])
        cuencas_gdf_lq = gpd.read_file('shapefiles/cuenca_lq.shp').dropna(
            subset=['NAME'])
        with rasterio.open("geotiff/ppn_" + str(i) + ".tif") as src:
            affine = src.transform
            array = src.read(1)
            df_zonal_stats = pd.DataFrame(zonal_stats(cuencas_gdf, array,
                                                      affine=affine, all_touched=True))
            df_zonal_stats_sa = pd.DataFrame(zonal_stats(cuencas_gdf_sa, array,
                                                         affine=affine, all_touched=True))
            df_zonal_stats_lq = pd.DataFrame(zonal_stats(cuencas_gdf_lq, array,
                                                         affine=affine, all_touched=True))
        cuencas_gdf = cuencas_gdf.rename(columns={'Subcuenca': 'subcuenca',
                                                  'Cuenca': 'cuenca'})
        cuencas_gdf = pd.concat([cuencas_gdf['subcuenca'],
                                 df_zonal_stats['mean']], axis=1)
        cuencas_gdf = cuencas_gdf.dropna(subset=['mean']).set_index(
            'subcuenca')
        tabla_hora.iloc[i] = cuencas_gdf['mean']
        # san antonio
        cuencas_gdf_sa = pd.concat([cuencas_gdf_sa['NAME'],
                                    df_zonal_stats_sa['mean']], axis=1)
        cuencas_gdf_sa = cuencas_gdf_sa.dropna(subset=['mean']).set_index(
            'NAME')
        tabla_hora_sa.iloc[i] = cuencas_gdf_sa['mean']
        # la quebrada
        cuencas_gdf_lq = pd.concat([cuencas_gdf_lq['NAME'],
                                    df_zonal_stats_lq['mean']], axis=1)
        cuencas_gdf_lq = cuencas_gdf_lq.dropna(subset=['mean']).set_index(
            'NAME')
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


def generar_producto_cuencas(wrfout, outdir_productos,
                             outdir_tabla, configuracion):
    # Abrimos DataFrame con las cuentas
    cuencas_gdf = gpd.read_file('shapefiles/cuencas.shp')
    # cuencas_lq = gpd.read_file('shapefiles/cuenca_lq.shp')
    # cuencas_sa = gpd.read_file('shapefiles/cuencas_sa.shp')

    rundate = corregir_wrfout(wrfout)

    genear_tif_prec(wrfout + '.nc', out_path='geotiff/ppn')

    cuencas_gdf_ppn = integrar_en_cuencas(cuencas_gdf)
    # cuencas_gdf_ppn_lq = integrar_en_cuencas(cuencas_lq)
    # cuencas_gdf_ppn_sa = integrar_en_cuencas(cuencas_sa)

    guardar_tabla(cuencas_gdf_ppn, outdir_tabla, rundate, configuracion)
    #    guardar_tabla(cuencas_gdf_ppn_lq, outdir_tabla, rundate, configuracion)
    #    guardar_tabla(cuencas_gdf_ppn_sa, outdir_tabla, rundate, configuracion)

    generar_tabla_por_hora(outdir_tabla, rundate, configuracion)

    generar_imagen(cuencas_gdf_ppn, outdir_productos, rundate, configuracion)

    # Eliminamos el wrfout que creamos
    os.remove(wrfout + '.nc')


def main():
    usage = """cuencas.py [--wrfout=pathToDatosWRF] [--outdir_productos=..]
            [--outdir_tabla ..] [--configuracion ..]"""
    parser = OptionParser(usage)
    parser.add_option("--wrfout", dest="wrfout",
                      help="ruta al wrfout de la salida del WRF")
    parser.add_option("--outdir_productos", dest="outdir_productos",
                      help="ruta donde se guardan los productos")
    parser.add_option("--outdir_tabla", dest="outdir_tabla",
                      help="ruta donde se guardan las tablas de datos")
    parser.add_option("--configuracion", dest="configuracion",
                      help="configuracion de las parametrizaciones")

    (opts, args) = parser.parse_args()
    if not opts.wrfout or not opts.outdir_tabla or not opts.outdir_productos \
            or not opts.configuracion:
        print("Faltan parametros!")
        print(usage)
    else:
        wrfout = opts.wrfout
        outdir_tabla = opts.outdir_tabla
        outdir_productos = opts.outdir_productos
        configuracion = opts.configuracion
        generar_producto_cuencas(wrfout, outdir_productos, outdir_tabla,
                                 configuracion)


if __name__ == "__main__":
    main()
