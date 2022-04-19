#!/usr/bin/env bash

ROOT="/home/kamikaze/projects/RTU"

source venv/bin/activate
PYTHONPATH="${ROOT}/rtu-bachelor-tasks/src:${PYTHONPATH}" python3.10 -m benchmark --data-root=$ROOT/datasets
deactivate
