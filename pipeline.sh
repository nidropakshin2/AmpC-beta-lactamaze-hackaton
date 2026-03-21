#!/bin/bash

set -e
set -o pipefail

CONDA_BASE=$(conda info --base)

source "$CONDA_BASE\etc\profile.d\conda.sh"
conda activate sber_hack_test

echo "Preprocessing data..."
python qsar/preprocess_data.py

echo "Building QSAR model..."
python qsar/optimize.py

echo "REINVENT agent learning..."
reinvent -l reinvent/stage1.log reinvent/stage1.toml