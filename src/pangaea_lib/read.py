#  Author : Alan D Snow, 2017.
#  License: BSD 3-Clause
"""
    This module provides helper functions to read in
    land surface model datasets.
"""
import xarray as xr


def open_mfdataset(path_to_lsm_files,
                   lat_var,
                   lon_var,
                   time_var,
                   lat_dim,
                   lon_dim,
                   time_dim,
                   lon_to_180=False,
                   coords_projected=False) -> xr.Dataset:
    """
    Wrapper to open land surface model netcdf files
    using :func:`xarray.open_mfdataset`.

    .. warning:: The time dimension and variable will both be
        renamed to 'time' to enable slicing.

    Parameters
    ----------
    path_to_lsm_files: :obj:`str`
        Path to land surface model files with wildcard.
        (Ex. '/path/to/files/*.nc')
    lat_var: :obj:`str`
        Latitude variable (Ex. lat).
    lon_var: :obj:`str`
        Longitude variable (Ex. lon).
    time_var: :obj:`str`
        Time variable (Ex. time).
    lat_dim: :obj:`str`
        Latitude dimension (Ex. lat).
    lon_dim: :obj:`str`
        Longitude dimension (Ex. lon).
    time_dim: :obj:`str`
        Time dimension (ex. time).
    lon_to_180: bool, optional, default=False
        It True, will convert longitude from [0 to 360]
        to [-180 to 180].
    coords_projected: bool, optional, default=False
        It True, it will assume the coordinates are already
        in the projected coordinate system.
    Read with pangaea example::

        import pangaea as pa

        with pa.open_mfdataset('/path/to/ncfiles/*.nc',
                               lat_var='lat',
                               lon_var='lon',
                               time_var='time',
                               lat_dim='lat',
                               lon_dim='lon',
                               time_dim='time') as xds:
            print(xds.lsm.projection)
    """

    def define_coords(_xds):
        """xarray loader to ensure coordinates are loaded correctly"""
        # remove time dimension from lat, lon coordinates
        if _xds[lat_var].ndim == 3:
            _xds[lat_var] = _xds[lat_var].squeeze(time_dim)
        # make sure coords are defined as coords
        if lat_var not in _xds.coords \
                or lon_var not in _xds.coords \
                or time_var not in _xds.coords:
            _xds.set_coords([lat_var, lon_var, time_var],
                            inplace=True)
        return _xds

    preprocess = define_coords

    # ToDo: Revisar http://xarray.pydata.org/en/stable/combining.html#combining-multi
    # http://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html
    xds = xr.open_mfdataset(
        path_to_lsm_files,
        preprocess=preprocess,
        concat_dim=time_dim,
        combine='by_coords'
    )
    xds.lsm.y_var = lat_var
    xds.lsm.x_var = lon_var
    xds.lsm.y_dim = lat_dim
    xds.lsm.x_dim = lon_dim
    xds.lsm.lon_to_180 = lon_to_180
    xds.lsm.coords_projected = coords_projected

    # make sure time dimensions are same for slicing
    # xds = xds.rename(
    #     {
    #         time_dim: 'time',
    #         time_var: 'time',
    #     }
    # )
    xds.lsm.to_datetime()
    return xds
