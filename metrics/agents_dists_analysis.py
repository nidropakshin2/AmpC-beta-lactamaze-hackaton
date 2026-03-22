import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ======================
# Utils
# ======================

def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def compute_properties(df):
    results = []

    for smi in df["smiles"]:
        mol = mol_from_smiles(smi)
        if mol is None:
            continue

        try:
            results.append({
                'smiles': smi,
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': rdMolDescriptors.CalcTPSA(mol),
                'QED': QED.qed(mol),
                'NumRotBonds': Descriptors.NumRotatableBonds(mol)
            })
        except:
            continue

    return pd.DataFrame(results)


# ======================
# Metrics
# ======================

def summary_stats(df):
    stats = {}

    for col in df.columns:
        if col in ['smiles']:
            continue
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median()
        }

    return stats


def kl_divergence(p, q, bins=50):
    hist_p, bin_edges = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bin_edges, density=True)

    hist_p += 1e-8
    hist_q += 1e-8

    return entropy(hist_p, hist_q)


# ======================
# Plotting
# ======================

def plot_distributions(train_df, gen1_df, gen2_df, output_prefix):
    cols = [c for c in train_df.columns if c not in ['smiles']]

    for col in cols:
        plt.figure()
        plt.hist(train_df[col], bins=50, alpha=0.4, label='train', density=True)
        plt.hist(gen1_df[col], bins=50, alpha=0.4, label='agent1', density=True)
        plt.hist(gen2_df[col], bins=50, alpha=0.4, label='agent2', density=True)
        plt.legend()
        plt.title(col)
        plt.savefig(f"{output_prefix}_{col}.png")
        # plt.close()


# ======================
# Main comparison
# ======================

def compare_datasets(train_df, gen1_df, gen2_df, output_file='results.csv'):
    print("Computing properties...")

    train_props = compute_properties(train_df)
    gen1_props = compute_properties(gen1_df)
    gen2_props = compute_properties(gen2_df)

    print("Computing summary stats...")

    stats_train = summary_stats(train_props)
    stats_gen1 = summary_stats(gen1_props)
    stats_gen2 = summary_stats(gen2_props)

    print("Computing KL divergences...")

    kl_results = []

    for col in train_props.columns:
        if col == 'smiles':
            continue

        kl_1 = kl_divergence(train_props[col], gen1_props[col])
        kl_2 = kl_divergence(train_props[col], gen2_props[col])

        kl_results.append({
            'property': col,
            'KL_train_vs_agent1': kl_1,
            'KL_train_vs_agent2': kl_2
        })

    kl_df = pd.DataFrame(kl_results)

    print("Saving results...")

    # flatten stats
    rows = []
    for dataset_name, stats in zip(
        ['train', 'agent1', 'agent2'],
        [stats_train, stats_gen1, stats_gen2]
    ):
        for prop, values in stats.items():
            rows.append({
                'dataset': dataset_name,
                'property': prop,
                **values
            })

    stats_df = pd.DataFrame(rows)

    final_df = stats_df.merge(kl_df, on='property', how='left')
    final_df.to_csv(output_file, index=False)

    print(f"Saved to {output_file}")

    print("Plotting distributions...")
    plot_distributions(train_props, gen1_props, gen2_props, "metrics/figures/agents_comparison/dist")


# ======================
# Example usage
# ======================

if __name__ == "__main__":
    # ожидается CSV с колонками: smiles, activity
    train_df = pd.read_csv("qsar/dataset/data_rdkit_train.csv")
    gen1_df = pd.read_csv("reinvent/data/samples_10000.csv")
    gen2_df = pd.read_csv("reinvent/data/samples_10000_PLS.csv")

    compare_datasets(train_df, gen1_df, gen2_df, output_file='metrics/results/agents_comparison.csv')
