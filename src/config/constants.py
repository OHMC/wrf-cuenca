import os

PROG_VERSION = 'v1.0.0'
DEBUG_CUENCAS = bool(os.getenv('DEBUG_CUENCAS', False))
PICKLE_PATH = 'pickles'
CUENCAS_API_PICKLE_PATH = f'{PICKLE_PATH}/cuencas_api_dict.p'

COLUM_REPLACE = {'Subcuenca': 'subcuenca', 'Cuenca': 'cuenca'}

# RAY_TIMELINE_PATH = '/tmp/ray_timeline.json'
RAY_ADDRESS = os.getenv('RAY_ADDRESS', "localhost:6379")
