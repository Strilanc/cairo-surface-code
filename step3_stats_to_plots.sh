#!/bin/bash

set -e

readonly IN_CSV=$1
readonly OUT_DIR=$2
readonly BASIS=$3
readonly CHUNKING=$4

if [ -z "${IN_CSV}" ]; then
  echo "First arg must be input stats csv file path."
  exit 1
fi
if [ -z "${OUT_DIR}" ]; then
  echo "Second arg must be output directory for plot images."
  exit 1
fi
if [ "$BASIS" = "X" ]; then
  readonly SKIP_BASIS="Z"
elif [ "$BASIS" = "Z" ]; then
  readonly SKIP_BASIS="X"
else
  echo "Third arg must be basis to analyze (either 'X' or 'Z')."
  exit 1
fi
if [ -z "${CHUNKING}" ]; then
  echo "Fourth arg must be the chunking to use ('d' for per code distance, 'round' for per round, or 'shot' for per shot)."
  exit 1
fi

PYTHONPATH=src python3 src/parsurf/scripts/plot_error_rate.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --save "${OUT_DIR}/error_rate_plot.png" \
    --skip_b "${SKIP_BASIS}"

PYTHONPATH=src python3 src/parsurf/scripts/plot_extrapolation.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --save "${OUT_DIR}/extrapolation_plot.png" \
    --skip_b "${SKIP_BASIS}"

PYTHONPATH=src python3 src/parsurf/scripts/plot_footprint.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --save "${OUT_DIR}/footprint_plot.png" \
    --skip_b "${SKIP_BASIS}"

PYTHONPATH=src python3 src/parsurf/scripts/plot_extrapolation.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --semi_systemic_bayesian \
    --save "${OUT_DIR}/semi_systemic_bayesian_extrapolation_plot.png" \
    --skip_b "${SKIP_BASIS}"

PYTHONPATH=src python3 src/parsurf/scripts/plot_footprint.py \
    --csv "${IN_CSV}" \
    --chunking "${CHUNKING}" \
    --semi_systemic_bayesian \
    --save "${OUT_DIR}/semi_systemic_bayesian_footprint_plot.png" \
    --skip_b "${SKIP_BASIS}"
