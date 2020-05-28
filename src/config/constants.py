import os

PROG_VERSION = 'v1.0.0'
DEBUG_CUENCAS = bool(os.getenv('DEBUG_CUENCAS', False))
PICKLE_PATH = 'pickles'
CUENCAS_API_PICKLE_PATH = f'{PICKLE_PATH}/cuencas_api_dict.p'

COLUM_REPLACE = {'Subcuenca': 'subcuenca', 'Cuenca': 'cuenca'}

RAY_ADDRESS = os.getenv('RAY_ADDRESS', "localhost:6379")

WRFOUT_REGEX = r"wrfout_(?P<param>[A-Z])_[a-z0-9]{3,4}_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})"
