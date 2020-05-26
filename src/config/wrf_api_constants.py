import os
import warnings

API_TOKEN = os.getenv('API_TOKEN', '')
if not API_TOKEN:
    warnings.warn("No se ha seteado un token de autenticacion para la API de WRF(API_TOKEN envirnonment variable)")

API_BASE_URL = 'https://ohmc.com.ar/wrf-beta/api'

WRF_DATA_SOURCES = ['localidades', 'estaciones']
WRF_API_NAMES = {
    'localidades': 'wrf-estacion',
    'estaciones': 'wrf-localidad'
}

NOMBRES_PROVINCIAS = {
    'cordoba': 'Córdoba',
    'chubut': 'Chubut',
    'san_juan': 'San Juan',
    'santiago_del_estero': 'Santiago del Estero'
}

API_RESPONSES = {
    'post_ok': 201,
    'get_ok': 200
}
