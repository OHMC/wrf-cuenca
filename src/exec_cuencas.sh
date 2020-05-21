#!/bin/bash
WRFOUT="/home/amontero/datos_wrfout/wrfout_A_d01_2020-05-16_06:00:00"
OUTDIR_PRODUCTOS="temp/productos/"
OUTDIR_TABLA="temp/tablas/"
CONFIGURACION="CBA_A_06"


python cuencas.py --wrfout ${WRFOUT} --outdir_productos ${OUTDIR_PRODUCTOS}  --outdir_tabla ${OUTDIR_TABLA} --configuracion ${CONFIGURACION}

