#!/bin/bash
WRFOUT="/datos/wrf/wrfout/2020_05/wrfout_A_d01_2020-05-27_06:00:00"
OUTDIR_PRODUCTOS="temp/productos/"
OUTDIR_TABLA="temp/tablas/"
# La configuracion ahora es un parametro opcional, si no se pasa, se obtiene automaticamente del wrfout con el formato
# CBA_{PARAMETRIZACION}_{HORA}
#CONFIGURACION="CBA_A_06"

ray start --head --redis-port=6379 --num-cpus=30

time python cuencas_wrf.py ${WRFOUT} ${OUTDIR_PRODUCTOS}  ${OUTDIR_TABLA} # --configuracion ${CONFIGURACION}

ray stop
