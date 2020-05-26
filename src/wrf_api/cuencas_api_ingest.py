import logging
import time

import requests
from requests.exceptions import RequestException

from config.logging_conf import INGESTOR_LOGGER_NAME
from config.wrf_api_constants import API_TOKEN, API_BASE_URL, WRF_DATA_SOURCES, API_RESPONSES, WRF_API_NAMES

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


def ingest_csv_to_db(wrf_api_dict: dict, wrf_config_dict: dict):
    logger.info("Iniciando ingestor de datos")
    timestamp = wrf_config_dict['timestamp']
    parametrizacion_id = get_wrf_api_object_id('parametrizacion', wrf_config_dict['configuracion']['parametrizacion'])

    corrida_payload = {'timestamp': timestamp, 'parametrizacion': parametrizacion_id}
    corrida_id = create_wrf_object('corrida', corrida_payload)

    producto_id = get_wrf_api_object_id('producto', wrf_config_dict['producto'], 'nombre')
    for provincia in wrf_api_dict.keys():
        # prov_id = get_wrf_api_object_id('provincias', NOMBRES_PROVINCIAS.get(provincia), 'nombre')
        for wrf_data_source in WRF_DATA_SOURCES:

            for nombre, v in wrf_api_dict[provincia][wrf_data_source].items():
                try:
                    # TODO: Revsisar si ademas hay que filtrar por provincia
                    wrf_data_source_id = get_wrf_api_object_id(wrf_data_source, nombre, 'nombre')
                except IndexError:
                    continue
                wrf_payload = {'timestamp': timestamp, 'producto': producto_id, 'corrida': corrida_id,
                               'path': v['path'], 'wrf_data_source': wrf_data_source_id}
                try:
                    time.sleep(0.3)
                    wrf_id = create_wrf_object(WRF_API_NAMES[wrf_data_source], wrf_payload)
                    logger.info(f"Creado wrf_id={wrf_id} -- {nombre}(id={wrf_data_source_id}): {v['path']}")
                except RequestException:
                    continue  # TODO: Revisar si continuar o cortar
