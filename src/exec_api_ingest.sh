#!/bin/bash

export API_BASE_URL_DICT='{"https://ohmc.com.ar/wrf-beta/api": {"token": "9c0d4c51ef7a43c3eab966b5cc96b549b2496caf"}, "https://ohmc.com.ar/wrf/api": {"token": "9c0d4c51ef7a43c3eab966b5cc96b549b2496caf"}}'

python api_ingest_csv.py
