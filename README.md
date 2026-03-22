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
- [Methods](#Methods)
- [System requirements](#System-requirements)
- [Dependencies](#Dependencies)
- [Authors](#Authors)

## Pipeline

![img](https://github.com/nidropakshin2/AmpC-beta-lactamaze-hackaton/blob/main/imgs/pipeline.png)

## Methods

The following tools are used in this project: 

### Core Frameworks & Libraries
- **Python 3.10+** as the primary programming language
- **REINVENT 4** — a framework for de novo molecule generation with Reinforcement Learning 

### Feature Engineering

- **RDKit (2023.09+)** — generation:
- Morgan Fingerprints
- Physicochemical descriptors

### Machine Learning Ecosystem
- **Scikit-learn** (v1.2+) for ML models:
- Gradient Boosting
  - **XGBoost** for QSAR modeling

### Reinforcement Learning

- **REINVENT 4** — the main RL framework

### Visualization Tools

- **RDKit** — visualization of molecules (2D structures)
- **PyMOL** — 3D visualization docking

### Development Environment
- **VS Code** with Python extensions
- **Git** for version control


## Authors

- M. Urakov
- M. Belyakov
- M. Mirny
- V. Ishtuganova
