#!/bin/bash
WRFOUT="/home/amontero/datos_wrfout/wrfout_A_d01_2020-05-16_06:00:00"
OUTDIR_PRODUCTOS="temp/productos/"
OUTDIR_TABLA="temp/tablas/"
CONFIGURACION="CBA_A_06"

ray start --head --redis-port=6379 --num-cpus=32 # --object-store-memory 42949672960 --redis-max-memory 21474836480

time python cuencas_wrf.py ${WRFOUT} ${OUTDIR_PRODUCTOS}  ${OUTDIR_TABLA} ${CONFIGURACION}

ray stop
