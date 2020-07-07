import datetime
import logging
import time

import requests
from requests.exceptions import RequestException

from config.logging_conf import INGESTOR_LOGGER_NAME
from config.wrf_api_constants import API_BASE_URL_DICT, API_RESPONSES

logger = logging.getLogger(INGESTOR_LOGGER_NAME)


def get_wrf_api_object_id(api_base_url, nombre, valor, campo):
    try:
        r = requests.get(f"{api_base_url}/{nombre}/?{campo}={valor}")
        return r.json().get('results')[0].get('id')
    except IndexError:
        logger.exception(f"No se encontro el ID de {valor}")
        raise
    except RequestException:
        logger.exception("Error al obtener objeto: ")
        raise


def create_wrf_object(api_base_url, token, nombre, payload: dict):
    headers_base = {'Authorization': f'Token {token}'}
    try:
        r = requests.post(f"{api_base_url}/{nombre}/", json=payload, headers=headers_base)
    except RequestException:
        logger.exception(f"Error al crear objeto - nombre={nombre},payload={payload}")
        raise
    if r.status_code != API_RESPONSES['post_ok']:
        logger.error(f"bad_request->s_code = {r.status_code} - nombre={nombre},payload={payload}")
        raise RequestException  # TODO: Revisar que sucede si ya existe el objeto
    else:
        wrf_id = r.json().get('id')
        return wrf_id


def ingest_csv_to_db(cuencas_dict: dict):
    logger.info("Iniciando ingestor de datos")
    timestamp = datetime.datetime.strftime(cuencas_dict['meta']['timestamp'], '%Y-%m-%d %H:%M')
    for api_base_url, meta in API_BASE_URL_DICT.items():
        parametrizacion_id = get_wrf_api_object_id(api_base_url, 'parametrizacion', cuencas_dict['meta']['param'],
                                                   campo='search')
        for prod in cuencas_dict['csv'].keys():
            producto_cuenca_id = get_wrf_api_object_id(api_base_url, 'producto-cuencas', prod, campo='nombre')
            path = cuencas_dict['csv'][prod]['path']
            acumulacion = cuencas_dict['csv'][prod]['acumulacion']
            payload = {
                "path": path,
                "acumulacion": acumulacion,
                'producto_cuencas': producto_cuenca_id,
                'parametrizacion': parametrizacion_id,
                'timestamp': timestamp,
            }
            try:
                create_wrf_object(api_base_url, meta['token'], nombre='cuencas', payload=payload)
            except RequestException:
                pass
            time.sleep(0.3)
