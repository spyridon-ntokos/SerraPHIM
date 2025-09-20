"""
SerraPHIM — Inference pipeline
==============================

Created on 17.07.2025
Author: snt
Based on: PhageHostLearn by @dimiboeckaerts

Short description
-----------------
This script runs the SerraPHIM inference workflow: it annotates new phage and bacterial
genomes (Pharokka + Bakta), detects phage RBPs (PhageRBPdetect v4), computes/loads
ESM-2 embeddings (via helper modules), constructs pairwise features (host × phage),
loads a trained XGBoost model and produces interaction probability predictions.
Predictions are saved as a matrix (hosts × phages) and as ranked lists of top phages per host.

Primary outputs
---------------
- `results/prediction_results{data_suffix}.csv` : matrix of predicted interaction probabilities
  (rows: host accessions, columns: phage IDs).
- `results/ranked_results{data_suffix}.pickle` : per-host ranked phage lists (tuples (phage, score)).
- `{results_path}/top_{n}_ranked_phages{data_suffix}.csv` : CSV-friendly top-N table.

Key assumptions & inputs
-----------------------
- Project path variables near the top must be adjusted to your environment.
- Inference data should be placed under `data_inf_path` (phage and bacterial FASTAs).
- The fine-tuned XGBoost model learned at training time must be available at `model_path`.
- Pharokka, Bakta and other required system binaries should be installed and callable.

Prerequisites
-------------
- Python packages: pandas, numpy, xgboost, biopython, etc.
- System binaries: Pharokka, Bakta and their dependencies for local annotation.
- If you want to compute embeddings here (instead of re-using previously saved CSVs),
  ensure a machine with sufficient memory/GPU is used.

Usage notes
-----------
- This script is meant for inference on new bacterial and/or phage genomes placed in the 
  `data_inf_path` (sub-)directory.
- The script saves intermediate outputs under `bakta_inf_results_path` and writes final
  predictions into `results_path`. Adjust paths accordingly.
- If you already ran Bakta/Pharokka previously and have outputs in `bakta_inf_results_path`
  or `pharokka_results`, the run will reuse them (helper functions may skip re-running).
"""

# 1 - INITIAL SETUP
# --------------------------------------------------
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Local modules: preprocessing and feature construction
import serraphim_processing as sp
import serraphim_features as sf

# -------------------------
# Project and data paths
# -------------------------
# IMPORTANT: Update these paths for your environment before running.
main_path = '/home/snt/PhageHostLearn-main'  # on yakuza server: '/home/sntokos/SerraPHIM'

data_inf_path = main_path + '/data/inference_data'  # root folder for inference inputs & intermediate outputs
code_path = main_path + '/code'
results_path = main_path + '/results'

# Phage & bacterial genome data that you want to make predictions for
phage_genomes_inf_path = data_inf_path + '/phage_genomes'
bact_genomes_inf_path = data_inf_path + '/bacteria_genomes'

# Bakta results path for inference — helper writes Bact_receptors{data_suffix}.json here
bakta_inf_results_path = data_inf_path + '/bakta_results'

# Suffix used so inference artifacts don't mix with training artifacts
data_suffix = '_inference'

# Software & DB paths (adjust for your environment)
phl_path = '/home/snt/PhageHostLearn-main'  # on yakuza server: not needed
pharokka_executable_path = phl_path + '/pharokka/bin/pharokka.py'  # on yakuza server: 'apptainer run /stornext/GL_CARPACCIO/home/HPC/opt/singularity/pharokka-1.7.3.sif pharokka.py'
pharokka_db_path = phl_path + '/pharokka/databases'  # on yakuza server: '/stornext/GL_CARPACCIO/home/HPC/opt/db/pharokka-db'

# Path to the fine-tuned PhageRBPdetect_v4 model (ESM-fine) — used for RBP detection
RBP_detect_model_path = main_path + '/RBPdetect_v4_ESMfine'

# Bakta DB path (used by run_bakta_and_extract_receptors)
bakta_db_path = phl_path + '/bakta_db/db'
bakta_results_path = data_inf_path + '/bakta_results'

# CPU resource tuning — change based on the machine you're using for inference
n_cpus = 16  # on yakuza server: 80


# 2 - DATA PROCESSING
# --------------------------------------------------
# Steps:
#  - run Pharokka on phage genomes -> produces pharokka_results/<phage> directories containing .faa, .gff, etc.
#  - run the RBP detector inference (PhageRBPdetect v4) to detect RBPs and create RBPbase{suffix}.csv
#  - run Bakta on bacterial genomes to annotate and extract candidate receptors -> Bact_receptors{suffix}.json

# Run Pharokka for phage annotation. This will create per-phage output directories under data_inf_path/pharokka_results
sp.pharokka_processing(
    data_inf_path, 
    phage_genomes_inf_path, 
    pharokka_executable_path, 
    pharokka_db_path, 
    data_suffix,
    threads=n_cpus
)

# Run RBP detection using PhageRBPdetect_v4; the helper checks for CUDA and falls back to CPU if necessary.
sp.process_and_detect_rbps_v4(
    data_inf_path, 
    RBP_detect_model_path, 
    data_suffix,
    gpu_index=0,
    threads=n_cpus
)

# Run Bakta to annotate bacterial genomes and extract receptor candidate proteins.
# The same keyword lists used in training are used here to identify potential receptors.
product_keywords = [
    "capsule", "polysaccharide", "-polysaccharide", "LPS",
    "oligosaccharide polymerase",
    "UDP-", "GDP-",
    "mannose",
    "guanylyltransferase",
    "glycosyltransferase",
    "flippase",
    "-antigen", "O-antigen",
    "outer membrane", "outer core", "omp-",
    "mannosyltransferase",
    "fimbria-",
    "pili-", "pilus",
    "adhesin",
    "cellulose",
    "curli",
    "siderophore receptor", "siderophore",
    "enterobactin receptor",
    "aerobactin receptor",
    "yersiniabactin receptor",
    "salmochelin receptor",
    "phage receptor",
    "porin", "-porin",
    "efflux pump",
    "hemagglutinin", "haemagglutinin",
    "lectin",
    "lipid", "-lipid", 
    "beta-barrel assembly",
    "autotransporter",
    "pathway protein",
    "flagell-",
    "chemotaxis",
    "transporter membrane", "transmembrane",
    "TonB-dependent receptor", "TonB", "TonB-",
    "chitinase", "lecithinase", "hemolysin", 
    "lipase", "-lipase", 
    "protease", "-protease",
    "exoenzyme-", "virulence", "invasion"
]
gene_keywords = [
    "wza", # (capsule export)
    "wzb", # (tyrosine-protein phosphatase)
    "wzc", # (capsule synthesis tyrosine kinase)
    "galF", # (UDP-glucose pyrophosphorylase)
    "ugd", # (UDP-glucose dehydrogenase)
    "manC", # (GDP-mannose pyrophosphorylase)
    "manB", # (mannose-1-phosphate guanylyltransferase)
    "Wzx", # (O-antigen flippase)
    "Wzy", # (O-antigen polymerase)
    "WaaL", # (O-antigen ligase)
    "Waa-", # (core LPS biosynthesis)
    "OmpA", "OmpX", "Omp-", # (outer membrane proteins)
    "TolC", # (outer membrane proteins)
    "OprD", "Opr-", # (outer membrane proteins)
    "HasR", "HasB", # (TonB-dependent outer membrane receptor complex)
    "lpx-", # (lipid A biosynthesis: lpxA, lpxC, lpxD)
    "WbbP", # (mannosyltransferase)
    "fimA", "fimH", # (Type 1 fimbriae)
    "mrkA", "mrkB", "mrkC", # (Type 3 fimbriae)
    "PilQ", "PilX", # (Adhesins)
    "csgA", "csgB", # (Curli fibers)
    "FliD", # (Flagellar hook-associated proteins)
    "CheA", "CheB", "CheR", "CheY", "CheW", "CheZ", "Che-", # (Chemotaxis proteins)
    "FepA", "Fiu", # (Enterobactin receptors)
    "IutA", # (Aerobactin receptors)
    "FyuA", # (Yersiniabactin receptors)
    "IroN" # (Salmochelin receptors)
]

# Run Bakta and extract receptors. If Bakta outputs already exist, the helper will skip re-running.
sp.run_bakta_and_extract_receptors(
    bact_genomes_inf_path, 
    bakta_inf_results_path, 
    bakta_db_path, 
    gene_keywords, product_keywords,
    data_suffix,
    threads=n_cpus,
    training=False
)


# 3 - FEATURE CONSTRUCTION
# --------------------------------------------------
# Compute or load ESM-2 embeddings for:
#  - RBPs detected by the RBP detector -> `data_inf_path/esm2_embeddings_rbp{data_suffix}.csv`
#  - receptors extracted by Bakta -> `data_inf_path/esm2_embeddings_receptors{data_suffix}.csv`
#
# If you already have these CSVs from previous runs, these steps can be skipped or run with `add=True`.

# Compute ESM-2 embeddings for RBPs (reads RBPbase{data_suffix}.csv)
sf.compute_esm2_embeddings_rbp_improved(
    data_inf_path, 
    data_suffix, 
    batch_size=n_cpus
)

# Compute ESM-2 embeddings for receptors (reads Bact_receptors{data_suffix}.json)
sf.compute_esm2_embeddings_receptors(
    data_inf_path, 
    bakta_inf_results_path, 
    data_suffix,
    aggregate_by_accession=True,
    batch_size=n_cpus,
    skip_long_seq=False
)

# Paths to the embedding CSVs
rbp_embeddings_path = data_inf_path + '/esm2_embeddings_rbp' + data_suffix + '.csv'
receptors_embeddings_path = data_inf_path + '/esm2_embeddings_receptors' + data_suffix + '.csv'

# Build the inference feature matrix that pairs each host accession with each phage
# Returns:
#   - features: array with concatenated receptor + averaged-per-phage RBP embeddings
#   - groups_hosts: list of host accession ids aligned with rows of features
#   - groups_phages: list of phage ids aligned with rows of features
features, groups_hosts, groups_phages = sf.construct_feature_matrices_inference(
    receptor_embeddings_path=f"{data_inf_path}/esm2_embeddings_receptors{data_suffix}.csv",
    rbp_embeddings_path=f"{data_inf_path}/esm2_embeddings_rbp{data_suffix}.csv"
)


# 4 - PREDICT & RANK NEW INTERACTIONS
# --------------------------------------------------
# Load the pre-trained XGBoost model and predict probabilities for every host×phage pair.
# Then aggregate predictions into a matrix and produce per-host ranked lists of phages.

# Path to saved model (trained earlier)
model_path = f"{results_path}/xgb_receptor_model_tuned_training.json"

# Load XGBoost model and run inference
model = XGBClassifier()
model.load_model(model_path)

# Run prediction (probability of interaction)
scores = model.predict_proba(features)[:, 1]

# Reconstruct unique host & phage sets and create an index mapping
hosts = sorted(set(groups_hosts))
phages = sorted(set(groups_phages))

# Initialize score matrix (hosts × phages)
score_matrix = np.zeros((len(hosts), len(phages)))
phage_to_idx = {ph: i for i, ph in enumerate(phages)}
host_to_idx = {h: i for i, h in enumerate(hosts)}

# Fill the matrix — groups_hosts/groups_phages align with `features` rows
for h, p, s in zip(groups_hosts, groups_phages, scores):
    i = host_to_idx[h]
    j = phage_to_idx[p]
    score_matrix[i, j] = s

# Save the prediction matrix as a CSV (rows=hosts, cols=phages)
results_df = pd.DataFrame(score_matrix, index=hosts, columns=phages)
results_df.to_csv(f"{results_path}/prediction_results{data_suffix}.csv")

# Create ranked lists of phages per host (descending probability)
ranked = {}
for i, host in enumerate(hosts):
    phage_scores = [(ph, score_matrix[i, phage_to_idx[ph]]) for ph in phages]
    phage_scores.sort(key=lambda x: x[1], reverse=True)
    ranked[host] = phage_scores

# Persist ranked results (pickle)
with open(f"{results_path}/ranked_results{data_suffix}.pickle", "wb") as f:
    pickle.dump(ranked, f)

# Load back the ranked results for convenience (optional)
with open(f"{results_path}/ranked_results{data_suffix}.pickle", "rb") as f:
    ranked_results = pickle.load(f)

# Prepare a human-friendly top-N table (phage names + scores per host)
top_n = 15
ranked_table = pd.DataFrame(index=results_df.index)

for rank in range(1, top_n + 1):
    phage_names_at_rank = []
    scores_at_rank = []

    for bact in results_df.index:
        # Sort phages by descending score for this bacterium and pick the rank-th phage
        top_phages = results_df.loc[bact].sort_values(ascending=False).head(top_n)
        phage = top_phages.index[rank - 1] if len(top_phages) >= rank else None
        score = top_phages.iloc[rank - 1] if len(top_phages) >= rank else None
        phage_names_at_rank.append(phage)
        scores_at_rank.append(round(score, 3) if score is not None else None)

    ranked_table[f"Rank {rank} Phage"] = phage_names_at_rank
    ranked_table[f"Rank {rank} Score"] = scores_at_rank

# Optionally, in Jupyter display the table with horizontal scroll:
# ranked_table.style.set_sticky().set_table_attributes('style="overflow-x:auto; display:block"')

# Save the final top-N ranked table as CSV
ranked_table.to_csv(f"{results_path}/top_{top_n}_ranked_phages{data_suffix}.csv")


# 5 - READ & INTERPRET RESULTS
# --------------------------------------------------
# TODO: Expand this section with summary statistics, top-k precision, or interactive dashboards.
# Minimal examples:
# - Inspect results_df.loc[host].nlargest(10) to see top predicted phages for a given host.
# - Use ranked_results to generate per-host CSV reports or to generate candidate lists for wet-lab testing.
#
# Notes and cautions:
# - If model was trained on species-level interactions, but inference is at strain/accession-level,
#   exercise caution interpreting absolute probabilities (they may be optimistic/pessimistic).
# - Consider calibrating predicted probabilities and evaluating top-k precision using any held-out known interactions.
