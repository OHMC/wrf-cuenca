import os
import warnings

API_TOKEN = os.getenv('API_TOKEN', '')
if not API_TOKEN:
    warnings.warn("No se ha seteado un token de autenticacion para la API de WRF(API_TOKEN envirnonment variable)")

API_BASE_URL = 'https://ohmc.com.ar/wrf-beta/api'

API_ROOT = "/datos-meteo"

API_RESPONSES = {
    'post_ok': 201,
    'get_ok': 200
}
