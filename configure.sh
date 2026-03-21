#!/bin/bash

CONDA_BASE=$(conda info --base)

source "$CONDA_BASE\etc\profile.d\conda.sh"
	
conda create -n sber_hack_test python=3.11
conda activate sber_hack_test
# pip install -r requirements.txt
pip install poetry==2.3.2

echo "Installing QSARtuna..."
if [ !(-d "QSARtuna") ] ; then
    git clone https://github.com/MolecularAI/QSARtuna.git
fi
cd QSARtuna
poetry install --all-extras
cd ..

echo "Installing REINVENT4..."
if [ !(-d "REINVENT4") ] ; then
    git clone https://github.com/MolecularAI/REINVENT4.git
fi
cd REINVENT4
python install.py cpu
cd ..



