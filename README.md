## Генерация ингибиторов AmpC β-лактамазы с помощью reinforcement learning и суррогатных ML-моделей 

<img align=right src="imgs/rep_image.png" alt="## Генерация ингибиторов AmpC β-лактамазы с помощью reinforcement learning и суррогатных ML-моделей " width="120"/>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.03%2B-green.svg)](https://www.rdkit.org/)
[![REINVENT](https://img.shields.io/badge/REINVENT-4.0-orange.svg)](https://github.com/MolecularAI/ReinventCommunity)

Aim
- This repository implements a computational drug discovery pipeline for generating
 novel AmpC β-lactamase inhibitors using reinforcement learning (RL) combined with  machine learning (ML) models.

The approach integrates:

- REINVENT 4: De novo molecular generation via generative models + RL optimization
- QSAR Modeling: Fast surrogate model (XgBoost) predicting pIC50 activity
- Multi-objective Optimization: Balancing activity, drug-likeness, synthesizability, and diversity

Target: E. coli AmpC β-lactamase  - a Class C β-lactamase conferring antibiotic resistance

## Contents
- [Pipeline](#Pipeline)
- [Dataset](#Dataset)
- [Methods](#Methods)
- [System requirements](#System-requirements)
- [Dependencies](#Dependencies)
- [Installation](#Installation)
- [Usage](#Usage)
- [Results](#Results)
- [Authors](#Authors)

## Pipeline

## Dependencies

General pipeline depends on:
- python >= 3.10
- rdkit >= 2023.03.1
- xgboost >= 
- reinvent >= 4.0


## Authors

- M. Urakov
- M. Belyakov
- M. Mirny
- V. Ishtuganova
