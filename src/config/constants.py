import os
import matplotlib

PROG_VERSION = 'v1.0.0'
DEBUG_CUENCAS = bool(os.getenv('DEBUG_CUENCAS', False))
PICKLE_PATH = 'pickles'
CUENCAS_API_PICKLE_PATH = os.getenv('CUENCAS_PICKLE_PATH', f'{PICKLE_PATH}/cuencas_api_dict.p')

COLUM_REPLACE = {'Subcuenca': 'subcuenca', 'Cuenca': 'cuenca'}

RAY_ADDRESS = os.getenv('RAY_ADDRESS', "localhost:6379")

WRFOUT_REGEX = r"wrfout_(?P<param>[A-Z])_[a-z0-9]{3,4}_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})"

CBA_EXTENT = [-68.91031,
              -37.408794,
              -56.489685,
              -27.518177]

WRF_EXTENT = [-538001.0623448786,
              -538000.0000000792,
              537998.9376551214,
              537999.9999999208]

# recorte para graphs de ppnaccum
RECORTE_EXTENT = [-66.07031,
                  -35.168794,
                  -61.579685,
                  -29.328177]

KM_PER_DEGREE = 111.32
RESOLUTION = 4

nws_precip_colors = [
    "#ffffff",  # 0.01 - 0.10 inches
    "#04e9e7",  # 0.10 - 0.25 inches
    "#019ff4",  # 0.25 - 0.50 inches
    "#0300f4",  # 0.50 - 0.75 inches
    "#02fd02",  # 0.75 - 1.00 inches
    "#01c501",  # 1.00 - 1.50 inches
    "#008e00",  # 1.50 - 2.00 inches
    "#fdf802",  # 2.00 - 2.50 inches
    "#e5bc00",  # 2.50 - 3.00 inches
    "#fd9500",  # 3.00 - 4.00 inches
    "#fd0000",  # 4.00 - 5.00 inches
    "#d40000",  # 5.00 - 6.00 inches
    "#bc0000",  # 6.00 - 8.00 inches
    "#f800fd",  # 8.00 - 10.00 inches
    "#9854c6"]  # 10.00+
PRECIP_COLORMAP = matplotlib.colors.ListedColormap(nws_precip_colors)

CLEVS = [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 80, 100, 200, 300]