import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from collections import defaultdict


# ======================
# 1. Загрузка модели
# ======================

model = joblib.load("qsar/model/latest.pkl")

# QSARtuna-style
estimator = model.predictor


# ======================
# 2. Получение feature importance
# ======================

importances = estimator.feature_importances_
importances = np.array(importances)

# топ фичи
TOP_K = 20
top_idx = np.argsort(importances)[-TOP_K:][::-1]

print("Top feature indices:", top_idx)


# ======================
# 3. Функция для получения bit_info
# ======================

def get_bit_info(smiles_list, n_bits=2048):
    """
    Для каждой молекулы собираем mapping:
    bit -> [(mol, atom_idx, radius)]
    """
    bit_dict = defaultdict(list)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=n_bits, bitInfo=bit_info
        )

        for bit, occurrences in bit_info.items():
            for atom_idx, radius in occurrences:
                bit_dict[bit].append((mol, atom_idx, radius))

    return bit_dict


# ======================
# 4. Извлечение субструктур
# ======================

def extract_substructure(mol, atom_idx, radius):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    if not env:
        return None
    submol = Chem.PathToSubmol(mol, env)
    return submol


# ======================
# 5. Визуализация топ фичей
# ======================

def visualize_top_bits(bit_dict, top_bits, max_examples=3):
    images = []
    legends = []

    for bit in top_bits:
        examples = bit_dict.get(bit, [])[:max_examples]

        for mol, atom_idx, radius in examples:
            submol = extract_substructure(mol, atom_idx, radius)
            if submol is not None:
                images.append(submol)
                legends.append(f"bit {bit}")

    if images:
        img = Draw.MolsToGridImage(
            images,
            molsPerRow=5,
            legends=legends,
            subImgSize=(200, 200), 
            returnPNG=True
        )
        return img
    else:
        print("No substructures found")
        return None


# ======================
# 6. Основной pipeline
# ======================

def analyze_features(smiles_list):

    print("Building bit dictionary...")
    bit_dict = get_bit_info(smiles_list)

    print("Top important bits:")
    for i, bit in enumerate(top_idx):
        print(f"{i+1}. Bit {bit}, importance={importances[bit]:.4f}")

    print("\nVisualizing substructures...")
    img = visualize_top_bits(bit_dict, top_idx)

    if img:
        img.save("figures/top_features.png")
        img.show()


# ======================
# 7. Пример использования
# ======================

if __name__ == "__main__":
    # ЗАГРУЗИ СВОЙ DATASET
    # smiles_list = [
    #     "CCO",
    #     "CCN",
    #     "c1ccccc1",
    #     "CC(=O)O",
    #     "CCCl",
    #     "CCBr",
    # ]
    smiles_list = pd.read_csv("qsar/dataset/data_rdkit_train.csv")["smiles"].tolist()

    analyze_features(smiles_list)