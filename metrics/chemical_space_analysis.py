import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap 
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tqdm import tqdm

# ======================
# Utils
# ======================

def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def smiles_to_fp(smiles, n_bits=2048):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def compute_fps(smiles_list):
    fps = []
    valid_smiles = []
    for s in smiles_list:
        fp = smiles_to_fp(s)
        if fp is not None:
            fps.append(fp)
            valid_smiles.append(s)
    return fps, valid_smiles

# ======================
# Similarity
# ======================

def tanimoto_matrix(fps):
    n = len(fps)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = TanimotoSimilarity(fps[i], fps[j])
            mat[i, j] = sim
            mat[j, i] = sim
    return mat


def nearest_neighbor_similarity(fps_a, fps_b):
    sims = []
    for fp in tqdm(fps_a):
        max_sim = max(TanimotoSimilarity(fp, fp2) for fp2 in fps_b)
        sims.append(max_sim)
    return np.array(sims)

# ======================
# Metrics
# ======================

def internal_diversity(fps):
    mat = tanimoto_matrix(fps)
    tril = mat[np.tril_indices(len(fps), k=-1)]
    return 1 - np.mean(tril)


def novelty(nns, threshold=0.7):
    return np.mean(nns < threshold)


def coverage(fps_train, fps_gen, threshold=0.7):
    covered = 0
    for fp in tqdm(fps_train):
        if any(TanimotoSimilarity(fp, fp2) > threshold for fp2 in fps_gen):
            covered += 1
    return covered / len(fps_train)

# ======================
# Scaffold
# ======================

def get_scaffold(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_stats(smiles_list, train_scaffolds=None):
    scaffolds = [get_scaffold(s) for s in smiles_list]
    scaffolds = [s for s in scaffolds if s is not None]
    unique = set(scaffolds)

    diversity = len(unique) / len(scaffolds)

    novelty = None
    if train_scaffolds is not None:
        novelty = len([s for s in unique if s not in train_scaffolds]) / len(unique)

    return diversity, novelty, unique

# ======================
# Descriptor distributions
# ======================

def compute_descriptors(smiles_list):
    data = []
    for s in smiles_list:
        mol = smiles_to_mol(s)
        if mol is None:
            continue
        data.append({
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol)
        })
    return pd.DataFrame(data)


def kl_divergence(p, q, bins=50):
    hist_p, bin_edges = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bin_edges, density=True)

    hist_p += 1e-8
    hist_q += 1e-8

    return entropy(hist_p, hist_q)

# ======================
# Visualization
# ======================

def embed_and_plot(fps_train, fps_rl, method="umap"):
    X = np.array([np.array(fp) for fp in fps_train + fps_rl])
    labels = ["train"] * len(fps_train) + ["rl"] * len(fps_rl)

    if method == "pca":
        emb = PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        emb = TSNE(n_components=2).fit_transform(X)
    else:
        emb = umap.UMAP().fit_transform(X)

    emb = np.array(emb)

    plt.figure(figsize=(8,6))
    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(emb[idx, 0], emb[idx, 1], label=label, alpha=0.5)

    plt.legend()
    plt.title(f"Embedding: {method}")
    plt.savefig(f"metrics/figures/embedding_{method}.png")
    # plt.show()


def plot_property_distributions(df_train, df_rl):
    for col in df_train.columns:
        plt.figure()
        plt.hist(df_train[col], bins=50, alpha=0.5, label="train", density=True)
        plt.hist(df_rl[col], bins=50, alpha=0.5, label="rl", density=True)
        plt.legend()
        plt.title(col)
        plt.savefig(f"metrics/figures/{col}_dist.png")
        # plt.show()

# ======================
# Main analysis
# ======================

def analyze(train_smiles, rl_smiles):
    print("Computing fingerprints...")
    fps_train, train_smiles = compute_fps(train_smiles)
    fps_rl, rl_smiles = compute_fps(rl_smiles)

    print("Internal diversity...")
    intdiv = internal_diversity(fps_rl)

    print("Nearest neighbor similarity...")
    nns = nearest_neighbor_similarity(fps_rl, fps_train)

    nov = novelty(nns)
    cov = coverage(fps_train, fps_rl)

    print("Scaffold stats...")
    _, _, train_scaffolds = scaffold_stats(train_smiles)
    scaf_div, scaf_nov, _ = scaffold_stats(rl_smiles, train_scaffolds)

    print("Descriptor distributions...")
    desc_train = compute_descriptors(train_smiles)
    desc_rl = compute_descriptors(rl_smiles)

    kl_scores = {}
    for col in desc_train.columns:
        kl_scores[col] = kl_divergence(desc_train[col], desc_rl[col])

    print("\n=== RESULTS ===", file=open("metrics/results.txt", "w"))
    print("Internal diversity:", intdiv, file=open("metrics/results.txt", "a"))
    print("Novelty:", nov, file=open("metrics/results.txt", "a"))
    print("Coverage:", cov, file=open("metrics/results.txt", "a"))
    print("Scaffold diversity:", scaf_div, file=open("metrics/results.txt", "a"))
    print("Scaffold novelty:", scaf_nov, file=open("metrics/results.txt", "a"))
    print("KL divergence:", kl_scores, file=open("metrics/results.txt", "a"))

    print("\nPlotting embeddings...")
    embed_and_plot(fps_train, fps_rl, method="umap")
    embed_and_plot(fps_train, fps_rl, method="tsne")
    embed_and_plot(fps_train, fps_rl, method="pca")
    

    print("Plotting property distributions...")
    plot_property_distributions(desc_train, desc_rl)


# ======================
# Example usage
# ======================

if __name__ == "__main__":
    # Replace with your data
    # train_smiles = ["CCO", "CCN", "CCC"]
    # rl_smiles = ["CCO", "CCCC", "CCCl"]

    train_smiles = pd.read_csv("qsar/dataset/data_rdkit_train.csv")["smiles"].to_list()[:5000]
    rl_smiles = pd.read_csv("reinvent/data/samples_10000.csv")["SMILES"].to_list()[:5000]
    analyze(train_smiles, rl_smiles)