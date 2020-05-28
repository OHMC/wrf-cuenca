import datetime
import logging
import time

import requests
from requests.exceptions import RequestException

from config.logging_conf import INGESTOR_LOGGER_NAME
from config.wrf_api_constants import API_TOKEN, API_BASE_URL, API_RESPONSES

headers_base = {'Authorization': f'Token {API_TOKEN}'}

logger = logging.getLogger(INGESTOR_LOGGER_NAME)


def get_wrf_api_object_id(nombre, valor, campo='search'):
    try:
        r = requests.get(f"{API_BASE_URL}/{nombre}/?{campo}={valor}")
        return r.json().get('results')[0].get('id')
    except IndexError:
        logger.exception(f"No se encontro el ID de {valor}")
        raise
    except RequestException:
        logger.exception("Error al obtener objeto: ")
        raise


def create_wrf_object(nombre, payload: dict):
    try:
        r = requests.post(f"{API_BASE_URL}/{nombre}/", json=payload, headers=headers_base)
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
    parametrizacion_id = get_wrf_api_object_id('parametrizacion', cuencas_dict['meta']['param'])

    corrida_payload = {'timestamp': timestamp, 'parametrizacion': parametrizacion_id}
    corrida_id = create_wrf_object('corrida', corrida_payload)
    for prod in cuencas_dict['csv'].keys():
        producto_id = get_wrf_api_object_id('producto', prod, 'nombre')
        path = cuencas_dict['csv'][prod]['path']
        is_image = cuencas_dict['csv'][prod]['is_image']
        acumulacion = cuencas_dict['csv'][prod]['acumulacion']
        payload = {
            "path": path,
            "acumulacion": acumulacion,
            "is_image": is_image,
            "producto": producto_id,
            "corrida": corrida_id
        }
        try:
            create_wrf_object(nombre='cuencas', payload=payload)
        except RequestException:
            pass
        time.sleep(0.3)
