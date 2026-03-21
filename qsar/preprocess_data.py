from rdkit import Chem
import pandas as pd
from rdkit.Chem import MolStandardize
from optunaz.utils.preprocessing.splitter import Stratified
import os

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
    df = df.dropna(subset=["PUBCHEM_EXT_DATASOURCE_SMILES", "Log of AC50"])
    df = df[df["Curve R2"] > 0.7]
    df = df[df["Curve Class"].isin([1,2,3])]
    
    # 2. стандартизация
    df["smiles"] = df["PUBCHEM_EXT_DATASOURCE_SMILES"].apply(standardize)
    # df = df.dropna(subset=["smiles"])
    
    # 3. target
    df["activity"] = df["Log of AC50"]

    return df

src = "qsar"

df = pd.read_csv(f"{src}/dataset/AID_585_datatable_all.csv")
df.drop(index=[0, 1, 2], inplace=True)
df.drop(columns=["PUBCHEM_RESULT_TAG", "PUBCHEM_ACTIVITY_URL", "PUBCHEM_ASSAYDATA_COMMENT", "Curve Description"], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(f"{src}/dataset/AID_585_datatable_all_cleaned.csv", index=False)
df = pd.read_csv(f"{src}/dataset/AID_585_datatable_all_cleaned.csv")

df = prepare_dataset(df)
df.to_csv(f"{src}/dataset/AID_585_datatable_preprocessed.csv", index=False)
df = pd.read_csv(f"{src}/dataset/AID_585_datatable_preprocessed.csv")

df = pd.read_csv(f"{src}/dataset/AID_585_datatable_preprocessed.csv")
primarydf = df[["smiles", "activity"]]
primarydf.to_csv(f"{src}/dataset/AID_585_datatable_smiles_activity.csv", index=False)
primarydf = pd.read_csv(f"{src}/dataset/AID_585_datatable_smiles_activity.csv")


train_str, test_str = Stratified(fraction=0.2, seed=42, bins="fd").split(primarydf["smiles"], primarydf["activity"])

print("Train (stratified):", len(train_str))
print("Test (stratified):", len(test_str))

primarydf.loc[train_str].to_csv(f"{src}/dataset/AID_585_train_str.csv", index=False)
primarydf.loc[test_str].to_csv(f"{src}/dataset/AID_585_test_str.csv", index=False)