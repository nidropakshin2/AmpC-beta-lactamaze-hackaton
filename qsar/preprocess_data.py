from rdkit import Chem
import pandas as pd
from rdkit.Chem import MolStandardize
from optunaz.utils.preprocessing.splitter import Stratified
import os
import argparse

def standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # удаление солей
    mol = MolStandardize.rdMolStandardize.FragmentParent(mol)
    
    # нормализация
    mol = MolStandardize.rdMolStandardize.Cleanup(mol)
    
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def prepare_dataset(df):

    # 1. фильтрация
    df = df[["Smiles", "pIC50"]]
    
    # 2. стандартизация
    df["smiles"] = df["Smiles"].apply(standardize)
    # df = df.dropna(subset=["smiles"])
    
    # 3. target
    df["activity"] = df["pIC50"]

    return df

def process(src, dataset_file):
    dataset_file = dataset_file.split(".")[0]
    df = pd.read_csv(f"{src}/dataset/{dataset_file}.csv")

    df = prepare_dataset(df)
    df.to_csv(f"{src}/dataset/{dataset_file}_preprocessed.csv", index=False)
    df = pd.read_csv(f"{src}/dataset/{dataset_file}_preprocessed.csv")


    train_str, test_str = Stratified(fraction=0.2, seed=42, bins="fd").split(df["smiles"], df["activity"])

    df.loc[train_str].to_csv(f"{src}/dataset/{dataset_file}_train.csv", index=False)
    df.loc[test_str].to_csv(f"{src}/dataset/{dataset_file}_test.csv", index=False)

    print(f"Train: {len(train_str)}", f"Test: {len(test_str)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='qsar', help='source directory (default: qsar)')
    parser.add_argument('--dataset_file', required=True, help='dataset file (CSV)')
    
    args = parser.parse_args()
    
    process(args.src, args.dataset_file)