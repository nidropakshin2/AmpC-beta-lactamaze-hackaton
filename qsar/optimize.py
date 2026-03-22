import apischema
import time
import toml
import pandas as pd
# Start with the imports.
import sklearn
from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.buildconfig import BuildConfig
from optunaz.config.optconfig import (
    OptimizationConfig,
    SVR,
    RandomForestRegressor,
    Ridge,
    XGBRegressor,
    PLSRegression,
    KNeighborsRegressor
)
from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepAllNoDeduplication
from optunaz.utils.preprocessing.splitter import Stratified, NoSplitting
from optunaz.descriptors import ECFP, MACCS_keys, ECFP_counts, PathFP


# Setup basic logging.
import logging
from importlib import reload
reload(logging)
logging.basicConfig(level=logging.INFO)
logging.getLogger("train").disabled = True # Prevent ChemProp from logging
import numpy as np
np.seterr(divide="ignore")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tqdm
from functools import partialmethod, partial
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # Prevent tqdm in ChemProp from flooding log

# Avoid decpreciated warnings from packages etc
import warnings
warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn


def build_QSAR(src, dataset_train_file, dataset_test_file):
    # Prepare hyperparameter optimization configuration.
    config = OptimizationConfig(
        data=Dataset(
            input_column="smiles",  # Typical names are "SMILES" and "smiles".
            response_column="activity",  # Often a specific name (like here), or just "activity".
            training_dataset_file=dataset_train_file,  # The file with training data.
            test_dataset_file=dataset_test_file,  # Hidden during optimization.
            deduplication_strategy=KeepAllNoDeduplication(),
            split_strategy=Stratified(),
            save_intermediate_files=True,
        ),
        descriptors=[
            ECFP.new(),
            ECFP_counts.new(),
            MACCS_keys.new(),
            PathFP.new()
        ],
        algorithms=[
            XGBRegressor.new(n_estimators={"low": 3, "high": 6}),
            SVR.new(),
            PLSRegression.new(),
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            use_cache=True,
            n_splits=5,  # Number of splits for cross-validation.
            n_trials=100,  # Total number of trials.
            n_startup_trials=50,  # Number of startup ("random") trials.
            random_seed=42, # Seed for reproducability
            direction=OptimizationDirection.MAXIMIZATION,
            shuffle=True,  # Shuffle data before splitting
        ),
    ) 
    
    datetime = time.strftime("%Y-%m-%d_%H_%M_%S")
    
    opt_config = apischema.serialize(config)
    toml.dump(opt_config, open(f"{src}/configs/opt_config.toml", "w"))

    study = optimize(config, study_name=f"study_{datetime}")
    
    buildconfig = buildconfig_best(study)
    buildconfig_as_dict = apischema.serialize(buildconfig)

    toml.dump(buildconfig_as_dict, open(f"{src}/configs/best_config_{datetime}.toml", "w"))
    toml.dump(buildconfig_as_dict, open(f"{src}/configs/best_config_latest.toml", "w"))
    

    build_best(buildconfig, f"{src}/model/model_{datetime}.pkl")
    build_best(buildconfig, f"{src}/model/latest.pkl")

    build_merged(buildconfig, f"{src}/model/model_merged_{datetime}.pkl")
    build_merged(buildconfig, f"{src}/model/model_merged_latest.pkl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='qsar', help='source directory (default: qsar)')
    parser.add_argument('--dataset_train_file', required=False, help='dataset train file (CSV)')
    parser.add_argument('--dataset_test_file', required=False, help='dataset test file (CSV)')
    
    args = parser.parse_args()
    
    build_QSAR(args.src, args.dataset_train_file, args.dataset_test_file)