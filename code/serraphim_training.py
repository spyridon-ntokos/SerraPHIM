"""
SerraPHIM — Training pipeline
=============================

Created on 17.07.2025
Author: snt
Based on: PhageHostLearn by @dimiboeckaerts

Short description
-----------------
This script orchestrates the SerraPHIM training workflow for Serratia spp. (bacterial)
hosts and their phages. It runs preprocessing (NCBI FASTA processing, Pharokka, Bakta),
extracts candidate receptor and RBP sequences, computes ESM-2 embeddings, constructs
pairwise bacterial receptor↔phage RBP feature matrices, performs hyperparameter tuning (Ray Tune),
trains an XGBoost model, and evaluates it using grouped cross-validation.

Primary outputs
---------------
- `data/interactions/phage_host_interactions{data_suffix}.csv` : binary interaction matrix
  (rows: bacterial accession/species, cols: phage IDs).
- `pharokka_results/<phage>/...` : Pharokka annotation directories per phage.
- `bakta_results/Bact_receptors{data_suffix}.json` and
  `Bact_receptors_verbose{data_suffix}.json` : receptor candidates extracted from Bakta.
- `data/esm2_embeddings_rbp{data_suffix}.csv` : ESM-2 embeddings for predicted RBPs.
- `data/esm2_embeddings_receptors{data_suffix}.csv` : ESM-2 embeddings for receptors.
- `{results_path}/xgb_receptor_model_tuned{data_suffix}.json` : trained XGBoost model.
- `{data_path}/ray_tune_results/` : Ray Tune artifacts for hyperparameter search.
- `{results_path}/serr_receptor_level_kfold_results.pickle` : k-fold CV scores & labels.

Key assumptions & inputs
-----------------------
- Project directory layout and file paths are set near the top of the script (adjust them).
- Phage metadata: a filtered TSV produced or found under `phagescope_metadata_path`.
- Phage genomes: one FASTA per phage (the script will split a multi-FASTA if needed).
- Bacterial genomes: FASTA files in `bact_genomes_path` (the script can process NCBI archives).
- The fine-tuned RBP classifier (PhageRBPdetect_v4) is available in `RBP_detect_model_path`.
- Bakta and Pharokka (and their required external tools) must be installed and available on PATH
  or reachable via the configured wrapper/executable path.

Prerequisites (software & Python packages)
-----------------------------------------
- System binaries: Pharokka, Bakta (and Bakta third-party deps), mmseqs2 (for Pharokka),
  HMMER (if used elsewhere), mash/minced/aragorn/etc. (as needed by pipeline).
- Python packages (example): pandas, numpy, torch, esm, tqdm, biopython, xgboost, sklearn,
  ray, transformers (when using PhageRBPdetect v4 model loader), and others used in helper modules.
  Install these into the project's virtual environment before running.

Resource recommendations
------------------------
- Embedding steps are resource intensive. Recommended:
  - GPU with >=16GB VRAM for ESM-2 embedding, OR
  - Large-memory machine (e.g. 256–512 GB RAM) if running on CPU.
- For hyperparameter tuning and parallel CPU steps, allocate multiple cores (e.g. 16–80 CPUs).
- Adjust `batch_size`, `skip_long_seq` and `n_cpus` to match your resources.

Important caveats & tips
------------------------
- ESM-2 memory usage increases quickly with sequence length. The code contains
  protections (skip / single-thread processing for very long sequences) — inspect
  skipped sequences (CSV is saved) and handle them separately if needed.
- Interaction matrix quality: if host–phage interactions are inferred at species level
  they may be noisy. Consider this when interpreting model performance.
- Grouped cross-validation strategy matters: choose grouping ('receptor' or 'phage')
  to match your expected deployment (predicting new hosts vs new phages).
- Save intermediate artifacts (Pharokka/Bakta results, embedding CSVs) to resume long runs.

How to run
----------
- Edit the path variables near the top of the script to match your environment.
- Ensure prerequisites are installed and available in your environment.
- Run the script (interactive or headless) on an appropriate compute node:
    python serraphim_training.py
  or execute blocks from a notebook (recommended during development).
- For large production runs, use the full `phage_genomes_path` and `bact_genomes_path`
  instead of sample directories.

Contact & license
-----------------
- Author: snt
- If you publish this repository, include an appropriate LICENSE and CONTRIBUTING.md.
"""

# 1 - INITIAL SETUP
# --------------------------------------------------
import ray
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Local modules: preprocessing and feature construction
import serraphim_processing as sp
import serraphim_features as sf

# -------------------------
# Project / data paths
# -------------------------
# Adjust these to your environment. On the HPC cluster you may want different paths.
main_path = '/home/snt/SerraPHIM'  # on yakuza server: '/home/sntokos/SerraPHIM'

# Code / data / results folders inside the project
code_path = main_path + '/code'
data_path = main_path + '/data'
results_path = main_path + '/results'

# Bacterial genome data paths (where processed FASTAs live)
bact_genomes_path = data_path + '/bacteria_genomes'
ncbi_bact_path = data_path + '/ncbi_bacteria/genbank/bacteria'

# Phage genome data and metadata
phage_genomes_path = data_path + '/phage_genomes'
phagescope_multiFASTA_path = data_path + '/PhageScope_data/PhageScope_virulent_complete-HQ_sequences.fasta'
phagescope_metadata_path = data_path + '/PhageScope_metadata'
metadata_file = 'serratia_complete-HQ_virulent_phage_metadata.tsv'

# Suffix appended to outputs so training/inference artifacts don't overlap
data_suffix = '_training'

# Software & DB paths (adjust for your environment)
phl_path = '/home/snt/PhageHostLearn-main'  # on yakuza server: not needed
pharokka_executable_path = phl_path + '/pharokka/bin/pharokka.py'  # on yakuza server: 'apptainer run /stornext/GL_CARPACCIO/home/HPC/opt/singularity/pharokka-1.7.3.sif pharokka.py'
pharokka_db_path = phl_path + '/pharokka/databases'  # on yakuza server: '/stornext/GL_CARPACCIO/home/HPC/opt/db/pharokka-db'

# 2025 fine-tuned RBP detector model
RBP_detect_model_path = main_path + '/RBPdetect_v4_ESMfine'

# Bakta DB and results location
bakta_db_path = phl_path + '/bakta_db/db'
bakta_results_path = data_path + '/bakta_results'

# How many CPUs to use for CPU-bound work. Tune to the machine (HPC has 128 cores).
n_cpus = 16  # on yakuza server: 80

# Convenience sample paths for fast/local testing — replace with full folders for real runs
phage_genomes_sample_path = phage_genomes_path + '/sample'
bact_genomes_sample_path = bact_genomes_path + '/sample'


# 2 - DATA PROCESSING
# --------------------------------------------------
# The pipeline here:
# - ensures bacterial genomes are available (process NCBI raw files if necessary)
# - loads/creates curated phage metadata (PhageScope filter for Serratia hosts)
# - splits multi-FASTA of phage genomes into individual files (one FASTA per phage)
# - derives a binary phage-host interaction matrix from metadata (rows: accession, cols: phage)
# - runs Pharokka to annotate phage genomes and extract nucleic genes
# - runs the PhageRBPdetect_v4 inference to detect RBPs among phage proteins
# - runs Bakta on bacterial genomes to annotate and extract receptor candidate proteins

# If bacterial genomes folder is missing/empty, process the downloaded NCBI archives and
# save cleaned per-genome FASTA files to `bact_genomes_path`.
if not Path(bact_genomes_path).exists() or not any(Path(bact_genomes_path).iterdir()):
    print("Bacterial genomes' directory is empty. Processing NCBI FASTA files...")
    sp.process_ncbi_fasta(
        input_root=ncbi_bact_path,
        output_root=bact_genomes_path,
        combine_plasmids="True"
    )
    # Notes:
    # - set combine_plasmids="True" to combine main chromosome + plasmids into a single file.
    # - change to False to get one FASTA per record (plasmids may be separate).
else:
    print("Bacterial genomes' directory is not empty. Moving on...")


# Create or load phage metadata filtered for Serratia + quality/lifestyle
phage_metadata = sp.load_or_create_curated_phage_metadata(
    phagescope_metadata_path,
    metadata_file
)

# If phage genome files are missing, split the large multi-FASTA into per-phage FASTAs.
if not Path(phage_genomes_path).exists() or not any(Path(phage_genomes_path).iterdir()):
    print("Phage genomes' directory is empty. Processing PhageScope multi-FASTA file...")
    sp.split_fasta_by_phage_id(
        phagescope_multiFASTA_path,
        phage_genomes_path
    )
else:
    print("Phage genomes' directory is not empty. Moving on...")


# Build an interaction matrix (binary 0/1) using species-level hosts found in phage metadata.
# Output written to `data/interactions/phage_host_interactions{data_suffix}.csv`
interaction_filepath = data_path + f'/interactions/phage_host_interactions{data_suffix}.csv'

phage_host_interaction_matrix = sp.derive_phage_host_interaction_matrix(
    bact_genomes_path,
    phage_metadata,
    interaction_filepath
)

# Run Pharokka on the phage genomes (annotation -> produced per-phage `pharokka_results/<phage>/`)
# NOTE: change to `phage_genomes_sample_path` here for speed/testing. For full runs point to `phage_genomes_path`.
sp.pharokka_processing(
    data_path, 
    phage_genomes_path, # 'phage_genomes_path' or 'phage_genomes_sample_path'
    pharokka_executable_path, 
    pharokka_db_path, 
    data_suffix,
    threads=n_cpus
)

# Detect RBPs using the PhageRBPdetect_v4 inference function.
# Device handling: if a CUDA GPU is available it will be used; otherwise the function will
# set PyTorch CPU threads to `threads`. On servers without proper CUDA support, CPU inference is used.
sp.process_and_detect_rbps_v4(
    data_path, 
    RBP_detect_model_path, 
    data_suffix,
    gpu_index=0,
    threads=n_cpus
)

# --------------------------
# Run Bakta to annotate bacterial genomes and extract receptors
# --------------------------
# Define keyword lists (gene names and product keywords) used to filter Bakta annotations.
# - gene_keywords: searched as substrings in Bakta's 'Gene' column
# - product_keywords: searched against 'Product' column with special rules (prefix/suffix/multi-word)
#
# Tweak these lists to include or exclude broader classes of proteins; adding very generic terms
# like "receptor" may increase false positives, so experiment and inspect the `Bact_receptors_verbose` output.
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
    # "receptor", # too generic? any receptors not related to surface receptors? 
    "siderophore receptor", "siderophore",
    "enterobactin receptor",
    "aerobactin receptor",
    "yersiniabactin receptor",
    "salmochelin receptor",
    "phage receptor",
    # "prophage", # how is a possible prophage existence affecting the susceptibility to other phages?
    # "phage", "tail fiber", # + other phage components... (CDS in prophage regions?)
    "porin", "-porin",
    "efflux pump",
    "hemagglutinin", "haemagglutinin",
    "lectin",
    "lipid", "-lipid",
    # "lipoprotein", # too generic? any lipoproteins not related to cell wall / surface receptors?
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
    #"magA", # (K1-specific capsular serotype proteins)
    #"rmpA", # (K2-specific capsular serotype proteins)
    "OmpA", "OmpX", "Omp-", # (outer membrane proteins)
    #"OmpK-", # (Klebsiella-specific outer membrane proteins)
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

# Run bakta on the sample genomes and extract receptors based on keywords.
# The function writes two JSON files per run:
#  - Bact_receptors{data_suffix}.json (simple accession->sequence dict)
#  - Bact_receptors_verbose{data_suffix}.json (detailed metadata per selected locus)
# NOTE: change to `bact_genomes_sample_path` here for speed/testing. For full runs point to `bact_genomes_path`.
sp.run_bakta_and_extract_receptors(
    bact_genomes_path, # 'bact_genomes_path' or 'bact_genomes_sample_path'
    bakta_results_path, 
    bakta_db_path, 
    gene_keywords, product_keywords,
    data_suffix,
    threads=n_cpus,
    training=True
)


# 3 - FEATURE CONSTRUCTION
# --------------------------------------------------
# Compute sequence embeddings (ESM-2) for:
#  - candidate RBPs (phage side)
#  - candidate receptors (bacterial side)
#
# NOTE: these steps are often the most resource-intensive. For large datasets, run on a server
# with GPUs or high-memory CPUs and appropriate batch sizes.

# Compute ESM-2 embeddings for RBPs (reads `RBPbase{data_suffix}.csv` and writes `esm2_embeddings_rbp{...}.csv`)
sf.compute_esm2_embeddings_rbp_improved(
    data_path, 
    data_suffix, 
    batch_size=n_cpus
)

# Compute ESM-2 embeddings for receptors (reads Bakta JSON and writes `esm2_embeddings_receptors{...}.csv`)
# aggregate_by_accession=True averages multiple receptor proteins per accession into a single vector
sf.compute_esm2_embeddings_receptors(
    data_path, 
    bakta_results_path, 
    data_suffix,
    aggregate_by_accession=True,  # aggregate per accession in the internship
    batch_size=n_cpus,
    skip_long_seq=False
)

# Paths for embedding files and interaction matrix
rbp_embeddings_path = data_path + '/esm2_embeddings_rbp' + data_suffix + '.csv'
receptors_embeddings_path = data_path + '/esm2_embeddings_receptors' + data_suffix + '.csv'
interaction_path = data_path + '/interactions'

# Build feature matrix for training. The function pairs each receptor accession with each phage,
# looks up the binary interaction in the matrix and keeps only labelled pairs (0 or 1).
features_esm2, labels, groups_receptors, groups_phage = sf.construct_feature_matrices_receptors(
    interaction_path, 
    data_suffix, 
    receptors_embeddings_path, 
    rbp_embeddings_path
)


# 4 - TRAINING & EVALUATING MODEL
# --------------------------------------------------
# Convert feature lists to arrays and compute class imbalance
X_full = features_esm2
y_full = labels
groups_receptors = np.array(groups_receptors)
groups_phage = np.array(groups_phage)

# Class imbalance handling: compute scale_pos_weight for XGBoost
# scale_pos_weight = (#pos)/(#neg)
imbalance = np.sum(y_full == 1) / np.sum(y_full == 0)
scale_pos_weight_fixed = 1 / imbalance

# ---------------------------
# Hyperparameter tuning (Ray Tune)
# ---------------------------
# Tune using a small search space (learning rate, max_depth, n_estimators, subsample, colsample)
# Ray Tune will run `num_samples` trials using the defined scheduler (ASHAScheduler here).
# Each trial uses the train_xgb_tune function which trains on a stratified split and reports AUC.
search_space = {
    "learning_rate": tune.loguniform(1e-3, 0.3),
    "max_depth": tune.choice([3, 5, 7, 9]),
    "n_estimators": tune.choice([100, 200, 300]),
    "scale_pos_weight": scale_pos_weight_fixed,
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.6, 1.0)
}

def train_xgb_tune(config, X, y):
    """
    Train a single XGBoost model for Ray Tune.
    - Performs a stratified train/validation split.
    - Trains an XGBClassifier with config hyperparameters.
    - Reports AUC on the validation set to Ray Tune.
    """
    # Stratified train/test split preserves positive/negative ratio on both sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        scale_pos_weight=config["scale_pos_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=n_cpus
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    # report metrics to Ray Tune (must be a dict)
    tune.report({"auc": auc})

# Initialize Ray with requested resources
ray.shutdown()
ray.init(
    num_cpus=n_cpus,
    _memory=20 * 1024 * 1024 * 1024,  # reserve 20GB RAM for Ray (adjust for your node)
    ignore_reinit_error=True
)

# Wrap the training function with fixed X, y to pass to tune
trainable = tune.with_parameters(train_xgb_tune, X=X_full, y=y_full)

# Execute Ray Tune hyperparameter search — adjust num_samples for more exhaustive search
analysis = tune.run(
    trainable,
    config=search_space,
    num_samples=100,  # number of sampled configurations
    metric="auc",
    mode="max",
    scheduler=ASHAScheduler(max_t=50, grace_period=5),
    resources_per_trial={"cpu": n_cpus},
    storage_path=f"{data_path}/ray_tune_results",
    name="xgb_receptor_tune"
)

# Best config found by Ray Tune
best_config = analysis.get_best_config(metric="auc", mode="max")
print("Best hyperparameters:", best_config)

# ---------------------------
# Train final model on all available labelled data using best hyperparameters
# ---------------------------
best_model = XGBClassifier(
    **best_config,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=n_cpus
)
best_model.fit(X_full, y_full)
best_model.save_model(f"{results_path}/xgb_receptor_model_tuned{data_suffix}.json")

# ---------------------------
# Group K-Fold Cross-validation
# ---------------------------
# Rationale for grouping:
# - group_by = 'receptor' : when you want to ensure that all samples pertaining to the
#   same receptor accession are held out together (testing ability to generalize to unseen hosts/accessions)
# - group_by = 'phage' : ensure all samples of the same phage are left out together (testing generalization to new phages)
# - group_by = 'both' : use combined keys; if you want to hold out specific pairs (uncommon)
#
# Choose grouping depending on the target use-case:
# - predicting for new hosts -> group by receptor
# - predicting for new phages -> group by phage
# - predicting for new phage-host pairs -> a different CV strategy may be needed
group_by = 'receptor'  # options: 'receptor', 'phage', 'both'

if group_by == 'receptor':
    groups = np.array(groups_receptors)
elif group_by == 'phage':
    groups = np.array(groups_phage)
elif group_by == 'both':
    # each group is a combined identifier; this is rarely used but supported
    groups = np.array([f"{b}_{p}" for b, p in zip(groups_receptors, groups_phage)])
else:
    raise ValueError("Invalid grouping strategy selected.")

print(f"Number of unique groups: {len(set(groups))}")

# Convert group labels (strings) to numeric IDs required by GroupKFold
unique_groups = list(set(groups))
group_to_id = {g: i for i, g in enumerate(unique_groups)}
group_ids = np.array([group_to_id[g] for g in groups])

# Standard k-fold using GroupKFold — avoids leakage across groups.
k = 5
gkf = GroupKFold(n_splits=k)
results = []
true_labels = []

# For each fold train a model using the best hyperparameters from tuning
for train_idx, test_idx in tqdm(gkf.split(X_full, y_full, groups=group_ids), total=k):
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    # Recompute class imbalance on the fold's training set and adjust scale_pos_weight
    imbalance = np.sum(y_train == 1) / np.sum(y_train == 0)
    best_config["scale_pos_weight"] = 1 / imbalance
    
    xgb = XGBClassifier(
        **best_config,
        n_jobs=n_cpus,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    y_score = xgb.predict_proba(X_test)[:, 1]

    results.append(y_score)
    true_labels.append(y_test)

# Persist k-fold results for later analysis
with open(f"{results_path}/serr_receptor_level_kfold_results.pickle", "wb") as f:
    pickle.dump({'scores': results, 'labels': true_labels}, f)


# 5 - READ & INTERPRET RESULTS
# --------------------------------------------------
# A minimal post-processing step printing fold-level arrays and a few metrics.
# NOTE: This section should be extended with per-group analysis, ROC/PR curves,
# calibration plots, top-k ranking per receptor, confusion matrices, etc.

# Pretty-print the scores and labels per fold (concise)
print("\nPredicted Scores (results):")
for i, arr in enumerate(results):
    print(f"Fold {i+1}: {np.array2string(arr, precision=4, separator=', ', floatmode='fixed')}")

print("\nGround Truth (true_labels):")
for i, arr in enumerate(true_labels):
    print(f"Fold {i+1}: {np.array2string(arr, separator=', ')}")

# Flatten arrays from all folds to compute aggregate metrics
all_scores = np.concatenate(results)
all_labels = np.concatenate(true_labels)

# Convert scores to binary predictions using 0.5 threshold. You can change threshold by optimizing
y_pred = (all_scores >= 0.5).astype(int)

# Compute common binary classification metrics
roc_auc = roc_auc_score(all_labels, all_scores)
pr_auc = average_precision_score(all_labels, all_scores)
f1 = f1_score(all_labels, y_pred)
acc = accuracy_score(all_labels, y_pred)

print("Cross-Validation Metrics:")
print(f"ROC AUC:     {roc_auc:.4f}")
print(f"PR AUC:      {pr_auc:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"Accuracy:    {acc:.4f}")

# Notes:
# - For interpretation, inspect per-group performance (i.e. how each held-out receptor
#   or phage performs) and check precision@k if the application is ranking phages to test.
# - If you see very different behavior between folds, investigate dataset splits,
#   small group sizes, or label noise (especially if interactions were inferred at species-level).
