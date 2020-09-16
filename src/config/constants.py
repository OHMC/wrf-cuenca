import os

PROG_VERSION = 'v1.0.0'
DEBUG_CUENCAS = bool(os.getenv('DEBUG_CUENCAS', False))
PICKLE_PATH = 'pickles'
CUENCAS_API_PICKLE_PATH = f'{PICKLE_PATH}/cuencas_api_dict.p'

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

KM_PER_DEGREE = 111.32
RESOLUTION = 4
