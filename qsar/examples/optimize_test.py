import apischema
import time
import toml
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
    Lasso,
    PLSRegression,
    KNeighborsRegressor
)
from optunaz.datareader import Dataset
from optunaz.utils.preprocessing.deduplicator import KeepAllNoDeduplication
from optunaz.utils.preprocessing.splitter import Stratified
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


def main():
    # parser = argparse.ArgumentParser()  
    # # Добавляем аргументы  
    # parser.add_argument('config', const='config.json')  
    # parser.add_argument('merged-model-outpath', const="model/merged.pkl")  
    # parser.add_argument('best-model-outpath', const="model/best.pkl")  
    
    # # Парсинг аргументов  
    # args = parser.parse_args()  
    
    # Prepare hyperparameter optimization configuration.
    config = OptimizationConfig(
        data=Dataset(
            input_column="smiles",  # Typical names are "SMILES" and "smiles".
            response_column="activity",  # Often a specific name (like here), or just "activity".
            training_dataset_file="qsar/dataset/AID_585_test_str.csv",  # The file with training data.
            # test_dataset_file="qsar/dataset/AID_585_test_str.csv",  # Hidden during optimization.
            deduplication_strategy=KeepAllNoDeduplication(),
            split_strategy=Stratified(),
        ),
        descriptors=[
            ECFP.new(),
            ECFP_counts.new(),
            MACCS_keys.new(),
            PathFP.new()
        ],
        algorithms=[
            SVR.new(),
            RandomForestRegressor.new(n_estimators={"low": 5, "high": 10}),
            Ridge.new(),
            # Lasso.new(),
            PLSRegression.new(),
            KNeighborsRegressor.new()
        ],
        settings=OptimizationConfig.Settings(
            mode=ModelMode.REGRESSION,
            n_splits=3,
            n_trials=100,  # Total number of trials.
            n_startup_trials=50,  # Number of startup ("random") trials.
            random_seed=44, # Seed for reproducability
            direction=OptimizationDirection.MAXIMIZATION,
        ),
    ) 
    
    opt_config = apischema.serialize(config)
    
    toml.dump(opt_config, open(f"qsar/configs/opt_config.toml", "w"))

    study = optimize(config, study_name="my_study")
    
    # Get the best Trial from the Study and make a Build (Training) configuration for it.
    buildconfig = buildconfig_best(study)

    buildconfig_as_dict = apischema.serialize(buildconfig)
    print(buildconfig_as_dict)

    datetime = time.strftime("%Y-%m-%d_%H_%M_%S")
    toml.dump(buildconfig_as_dict, open(f"qsar/configs/best_config_{datetime}.toml", "w"))
    
    best_build = build_best(buildconfig, f"qsar/model/model_{datetime}.pkl")
    best_build = build_best(buildconfig, f"qsar/model/latest.pkl")

main()