import pickle

from config.constants import CUENCAS_API_PICKLE_PATH
from config.logging_conf import get_logger_from_config_file, INGESTOR_LOGGER_NAME
from wrf_api.cuencas_api_ingest import ingest_csv_to_db

logger = get_logger_from_config_file(logger_name=INGESTOR_LOGGER_NAME)


def load_wrf_api_dicts():
    """
    Carga los diccionarios generados en tablas_datos con los datos para ingestar en ld DB

    Returns
    -------
    dict

    """
    try:
        with open(CUENCAS_API_PICKLE_PATH, mode='rb') as w_api_f:
            cuencas_api_dict = pickle.load(w_api_f)
    except (OSError, pickle.PickleError):
        logger.exception("Error al cargar diccionarios")
        raise
    return cuencas_api_dict


if __name__ == '__main__':
    logger.info("Iniciando ingestor DB.")
    ingest_csv_to_db(load_wrf_api_dicts())
    logger.info("Fin ingestor.")
