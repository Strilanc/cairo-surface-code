#!/bin/bash

IN_CSV=$1
OUT_DIR=$2
CHUNKING=d

if [ -z "${IN_CSV}" ]; then
  echo "First arg must be input stats csv file path."
  exit 1
fi
if [ -z "${OUT_DIR}" ]; then
  echo "Second arg must be output directory for plot images."
  exit 1
fi

PYTHONPATH=src python3 src/parsurf/scripts/plot_error_rate.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --save "${OUT_DIR}/error_rate_plot.png"

PYTHONPATH=src python3 src/parsurf/scripts/plot_extrapolation.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --save "${OUT_DIR}/extrapolation_plot.png"

PYTHONPATH=src python3 src/parsurf/scripts/plot_footprint.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --save "${OUT_DIR}/footprint_plot.png"

PYTHONPATH=src python3 src/parsurf/scripts/plot_extrapolation.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --semi_systemic_bayesian \
    --save "${OUT_DIR}/semi_systemic_bayesian_extrapolation_plot.png"

PYTHONPATH=src python3 src/parsurf/scripts/plot_footprint.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --semi_systemic_bayesian \
    --save "${OUT_DIR}/semi_systemic_bayesian_footprint_plot.png"
