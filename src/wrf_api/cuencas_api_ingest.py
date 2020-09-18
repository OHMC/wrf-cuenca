import datetime
import logging
import time
from json.decoder import JSONDecodeError

import requests
from requests.exceptions import RequestException

from config.logging_conf import INGESTOR_LOGGER_NAME
from config.wrf_api_constants import API_BASE_URL_DICT

logger = logging.getLogger(INGESTOR_LOGGER_NAME)


def get_wrf_api_object_id(api_base_url, nombre, query):
    try:
        r = requests.get(f"{api_base_url}/{nombre}/{query}")
        r.raise_for_status()
        return r.json().get('results')[0].get('id')
    except (IndexError, KeyError, JSONDecodeError):
        logger.exception(f"No se encontro el ID de {query}")
        raise
    except RequestException:
        logger.exception(f"Error al obtener objeto")
        raise


def create_wrf_object(api_base_url, token, nombre, payload: dict):
    headers_base = {'Authorization': f'Token {token}'}
    try:
        r = requests.post(f"{api_base_url}/{nombre}/", json=payload, headers=headers_base)
        r.raise_for_status()
        wrf_id = r.json().get('id')
        return wrf_id
    except RequestException:
        logger.exception(f"Error al crear objeto - nombre={nombre},payload={payload}")
        raise
    except (JSONDecodeError, KeyError):
        logger.exception("Error al decodificar las respuesta")
        raise


def ingest_csv_to_db(cuencas_dict: dict):
    logger.info("Iniciando ingestor de datos")
    for api_base_url, meta in API_BASE_URL_DICT.items():
        cuenca_dict: dict
        for accum, cuenca_dict in cuencas_dict.items():
            timestamp = datetime.datetime.strftime(cuenca_dict['meta']['timestamp'], '%Y-%m-%d %H:%M')
            try:
                parametrizacion_id = get_wrf_api_object_id(api_base_url, 'parametrizacion',
                                                           query=f"?nombre={cuenca_dict['meta']['param']}")
            except Exception:
                logger.warning(f"No se pudo realizar la ingestion para {api_base_url}")
                continue
            acumulacion = f"{accum}:00:00"
            for prod, prod_dict in cuenca_dict['csv'].items():
                try:
                    producto_cuenca_id = get_wrf_api_object_id(api_base_url, 'producto-cuencas',
                                                               query=f"?short_name={prod}&accumulation={acumulacion}")
                except Exception:
                    logger.warning(f"No se pudo realizar la ingestion para {api_base_url}")
                    continue
                try:
                    path = prod_dict['path']
                except KeyError:
                    continue

                payload = {
                    "path": path,
                    'producto_cuencas': producto_cuenca_id,
                    'parametrizacion': parametrizacion_id,
                    'timestamp': timestamp,
                }
                try:
                    create_wrf_object(api_base_url, meta['token'], nombre='csvs-cuencas', payload=payload)
                except Exception:
                    pass
