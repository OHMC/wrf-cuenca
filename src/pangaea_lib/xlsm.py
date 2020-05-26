#  Author : Alan D Snow, 2017.
#  License: BSD 3-Clause
"""
    This module is an extension for xarray for land surface models.
    (see: http://xarray.pydata.org/en/stable/internals.html#extending-xarray)
"""
import numpy as np
import pandas as pd
import wrf
import xarray as xr
from affine import Affine
from gazar.grid import geotransform_from_yx
from osgeo import osr
from pyproj import Proj, transform


def register():
    pass


@xr.register_dataset_accessor('lsm')
class LSMGridReader:
    """
    This is an extension for xarray specifically
    designed for land surface models.

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

    Read with xarray example::

        import xarray as xr

        with pa.open_dataset('/path/to/file.nc') as xds:
            print(xds.lsm.projection)
    """

    def __init__(self, xarray_obj):
        self.xarr_obj = xarray_obj
        self._projection = None
        self._epsg = None
        self._geotransform = None
        self._affine = None
        self._center = None
        self._y_inverted = None

        # set variable information
        self.y_var = 'XLAT'
        self.x_var = 'XLONG'
        self.time_var = 'XTIME'
        # set dimension information
        self.y_dim = 'south_north'
        self.x_dim = 'west_east'
        self.time_dim = 'Time'
        # convert lon from [0 to 360] to [-180 to 180]
        self.lon_to_180 = False
        # coordinates are projected already
        self.coords_projected = False
        #
        self.rainc = None
        self.rainnc = None

    def to_datetime(self):
        """Converts time to datetime."""
        time_values = self.xarr_obj[self.time_var].values
        if 'datetime' not in str(time_values.dtype):
            try:
                time_values = [time_val.decode('utf-8') for time_val in time_values]
            except AttributeError:
                pass

            try:
                datetime_values = pd.to_datetime(time_values)
            except ValueError:
                # WRF DATETIME FORMAT
                datetime_values = pd.to_datetime(time_values, format="%Y-%m-%d_%H:%M:%S")
            # FixMe: ValueError: Cannot assign to the .values attribute of dimension coordinate a.k.a IndexVariable
            #  'Time'. Please use DataArray.assign_coords, Dataset.assign_coords or Dataset.assign as appropriate.
            # rainc['Time'].values tiene los datetime
            self.xarr_obj[self.time_var].values = datetime_values

    @property
    def y_inverted(self):
        """Is the y-coord inverted"""
        if self._y_inverted is None:
            y_coords = self.xarr_obj[self.y_var].values
            if y_coords.ndim == 3:
                y_coords = y_coords[0]
            if y_coords.ndim == 2:
                self._y_inverted = (y_coords[-1, 0] > y_coords[0, 0])
            else:
                self._y_inverted = (y_coords[-1] > y_coords[0])
        return self._y_inverted

    def _load_wrf_projection(self):
        """Load the osgeo.osr projection for WRF Grid.

        - 'MAP_PROJ': The map projection type as an integer.
        - 'TRUELAT1': True latitude 1.
        - 'TRUELAT2': True latitude 2.
        - 'MOAD_CEN_LAT': Mother of all domains center latitude.
        - 'STAND_LON': Standard longitude.
        - 'POLE_LAT': Pole latitude.
        - 'POLE_LON': Pole longitude.
        """
        # load in params from WRF Global Attributes
        possible_proj_params = ('MAP_PROJ', 'TRUELAT1', 'TRUELAT2',
                                'MOAD_CEN_LAT', 'STAND_LON', 'POLE_LAT',
                                'POLE_LON', 'CEN_LAT', 'CEN_LON', 'DX', 'DY')
        proj_params = dict()
        for proj_param in possible_proj_params:
            if proj_param in self.xarr_obj.attrs:
                proj_params[proj_param] = self.xarr_obj.attrs[proj_param]

        # determine projection from WRF Grid
        # ToDo: Corregir proj con expresion regular para remover parametros innecesarios
        # proj = wrf.getproj(**proj_params)
        proj_str = "+proj=lcc +lat_1=-60.0 +lat_2=-30.0 +lat_0=-32.50000762939453 +lon_0=-62.70000076293945"

        # export to Proj4 and add as osr projection
        self._projection = osr.SpatialReference()
        self._projection.ImportFromProj4(proj_str)

    def _load_grib_projection(self):
        """Get the osgeo.osr projection for Grib Grid.
            - grid_type:  Lambert Conformal
            - Latin1:     True latitude 1.
            - Latin2:     True latitude 2.
            - Lov:        Central meridian.
            - Lo1:        Pole longitude.
            - La1:        Pole latitude.
            - Dx:         [ 3.]
            - Dy:         [ 3.]
        """
        lat_var_attrs = self.xarr_obj[self.y_var].attrs
        if 'Lambert Conformal' in lat_var_attrs['grid_type']:
            mean_lat = self.xarr_obj[self.y_var].mean().values
            proj4_str = ("+proj=lcc "
                         "+lat_1={true_lat_1} "
                         "+lat_2={true_lat_2} "
                         "+lat_0={latitude_of_origin} "
                         "+lon_0={central_meridian} "
                         "+x_0=0 +y_0=0 "
                         "+ellps=WGS84 +datum=WGS84 "
                         "+units=m +no_defs") \
                .format(true_lat_1=lat_var_attrs['Latin1'][0],
                        true_lat_2=lat_var_attrs['Latin2'][0],
                        latitude_of_origin=mean_lat,
                        central_meridian=lat_var_attrs['Lov'][0])
        else:
            raise ValueError("Unsupported projection: {grid_type}"
                             .format(grid_type=lat_var_attrs['grid_type']))

        # export to Proj4 and add as osr projection
        self._projection = osr.SpatialReference()
        self._projection.ImportFromProj4(proj4_str)

    @property
    def projection(self):
        """:func:`osgeo.osr.SpatialReference`
            The projection for the dataset.
        """
        if self._projection is None:
            # read projection information from global attributes
            map_proj4 = self.xarr_obj.attrs.get('proj4')
            if map_proj4 is not None:
                self._projection = osr.SpatialReference()
                self._projection.ImportFromProj4(str(map_proj4))
            elif 'MAP_PROJ' in self.xarr_obj.attrs:
                # logger.info("pass_projection_02")
                self._load_wrf_projection()
            elif 'grid_type' in self.xarr_obj[self.y_var].attrs:
                self._load_grib_projection()
            elif 'ProjectionCoordinateSystem' in self.xarr_obj.keys():
                # national water model
                proj4_str = self.xarr_obj['ProjectionCoordinateSystem'] \
                    .attrs['proj4']
                self._projection = osr.SpatialReference()
                self._projection.ImportFromProj4(str(proj4_str))
            else:
                # default to EPSG 4326
                self._projection = osr.SpatialReference()
                self._projection.ImportFromEPSG(4326)
            # make sure EPSG loaded if possible
            self._projection.AutoIdentifyEPSG()
        return self._projection

    @property
    def epsg(self):
        """str: EPSG code"""
        if self._epsg is None:
            self._epsg = self.projection.GetAuthorityCode(None)
        return self._epsg

    @property
    def geotransform(self):
        """:obj:`tuple`: The geotransform for grid."""
        if self._geotransform is None:
            if self.xarr_obj.attrs.get('geotransform') is not None:
                self._geotransform = [float(g) for g in self.xarr_obj.attrs.get('geotransform')]
            elif str(self.epsg) != '4326':
                proj_y, proj_x = self.coords
                self._geotransform = geotransform_from_yx(proj_y, proj_x)
            else:
                self._geotransform = geotransform_from_yx(*self.latlon)
        return self._geotransform

    @property
    def affine(self):
        """:func:`Affine`: The affine for the transformation."""
        if self._affine is None:
            self._affine = Affine.from_gdal(*self.geotransform)
        return self._affine

    @property
    def _raw_coords(self):
        """Gets the raw coordinated of dataset"""
        x_coords = self.xarr_obj[self.x_var].values
        y_coords = self.xarr_obj[self.y_var].values

        if x_coords.ndim == 3:
            x_coords = x_coords[0]
        if y_coords.ndim == 3:
            y_coords = y_coords[0]

        if x_coords.ndim < 2:
            x_coords, y_coords = np.meshgrid(x_coords, y_coords)

        # WRF & NWM Grids are upside down
        if self.y_inverted:
            x_coords = x_coords[::-1]
            y_coords = y_coords[::-1]

        return y_coords, x_coords

    @property
    def latlon(self):
        """Returns lat,lon arrays

            .. warning:: The grids always be returned with [0,0]
                as Northeast and [-1,-1] as Southwest.
        """
        if 'MAP_PROJ' in self.xarr_obj.attrs:
            lat, lon = wrf.latlon_coords(self.xarr_obj, as_np=True)
            if lat.ndim == 3:
                lat = lat[0]
            if lon.ndim == 3:
                lon = lon[0]
            # WRF Grid is upside down
            lat = lat[::-1]
            lon = lon[::-1]
        else:
            lat, lon = self._raw_coords

        if self.coords_projected:
            lon, lat = transform(Proj(self.projection.ExportToProj4()),
                                 Proj('epsg:4326'),
                                 lon,
                                 lat)

        if self.lon_to_180:
            lon = (lon + 180) % 360 - 180  # convert [0, 360] to [-180, 180]

        return lat, lon

    @property
    def coords(self):
        """Returns y, x coordinate arrays

            .. warning:: The grids always be returned with [0,0]
                as Northeast and [-1,-1] as Southwest.
        """
        if not self.coords_projected:
            lat, lon = self.latlon
            x_coords, y_coords = \
                transform(Proj('epsg:4326'),
                          Proj(self.projection.ExportToProj4()),
                          lon,
                          lat)
            return y_coords, x_coords
        return self._raw_coords

    def export_dataset(self, variable, new_data, grid):
        """Export subset of dataset."""
        lats, lons = grid.latlon

        return xr.Dataset(
            {variable: (['time', 'y', 'x'], new_data, self.xarr_obj[variable].attrs)},
            coords={
                'lat': (['y', 'x'], lats, self.xarr_obj[variable].coords[self.y_var].attrs),
                'lon': (['y', 'x'], lons, self.xarr_obj[variable].coords[self.x_var].attrs),
                'time': (['time'], self.xarr_obj[self.time_var].values, self.xarr_obj[self.time_var].attrs)
            },
            attrs={'proj4': grid.proj4, 'geotransform': grid.geotransform}
        )
