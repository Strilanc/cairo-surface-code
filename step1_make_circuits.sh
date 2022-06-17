#!/bin/bash

CIRCUIT_DIR=$1
HCB_PATH=$2

if [ -z "${CIRCUIT_DIR}" ]; then
  echo "First arg must be output directory for circuits."
  exit 1
fi
if [ -z "${HCB_PATH}" ]; then
  echo "Second arg must be '--no-honeycomb' or a directory containing a clone of https://github.com/Strilanc/honeycomb-boundaries"
  exit 1
fi
if [ "${HCB_PATH}" = "--no-honeycomb" ]; then
  pypath="src"
  honeycomb_switch="0"
else
  pypath="src:${HCB_PATH}/src"
  honeycomb_switch="1"
fi

PYTHONPATH="${pypath}" python3 src/parsurf/scripts/generate_circuit_files.py \
    --diam 5 7 9 11 13 15 17 19 \
    --basis X Z \
    --noise 0.0001 0.0002 0.0003 0.0005 0.0008 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 \
    --round_factors 3 \
    --honeycomb "${honeycomb_switch}" \
    --out_dir "${CIRCUIT_DIR}"
