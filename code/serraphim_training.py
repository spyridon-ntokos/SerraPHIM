"""
Created on 17.07.2025

@author: snt 
based on PhageHostLearn by @dimiboeckaerts

SerraPHIM TRAINING
"""

# 1 - INITIAL SETUP
# --------------------------------------------------
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

import serraphim_processing as sp
import serraphim_features as sf

# Project path
main_path = '/home/snt/SerraPHIM'

# General paths
code_path = main_path + '/code'
data_path = main_path + '/data'
results_path = main_path + '/results'

# Bacterial genome data paths
bact_genomes_path = data_path + '/bacteria_genomes'
ncbi_bact_path = data_path + '/ncbi_bacteria/genbank/bacteria'

# Phage genome data and metadata paths
phage_genomes_path = data_path + '/phage_genomes'
phagescope_multiFASTA_path = data_path + '/PhageScope_data/PhageScope_virulent_complete-HQ_sequences.fasta'

phagescope_metadata_path = data_path + '/PhageScope_metadata'
metadata_file = 'serratia_complete-HQ_virulent_phage_metadata.tsv'

data_suffix = '_training'

# Software and database paths
phl_path = '/home/snt/PhageHostLearn-main'
pharokka_executable_path = phl_path + '/pharokka/bin/pharokka.py'
pharokka_db_path = phl_path + '/pharokka/databases'

RBP_detect_model_path = main_path + '/RBPdetect_v4_ESMfine'

bakta_db_path = phl_path + '/bakta_db/db'
bakta_results_path = data_path + '/bakta_results'

# Paths for testing purposes:
phage_genomes_sample_path = phage_genomes_path + '/sample'
bact_genomes_sample_path = bact_genomes_path + '/sample'


# 2 - DATA PROCESSING
# --------------------------------------------------
# 2 - DATA PROCESSING
# --------------------------------------------------

# Check if the bacterial genomes' directory exists and is empty
if not Path(bact_genomes_path).exists() or not any(Path(bact_genomes_path).iterdir()):
    print("Bacterial genomes' directory is empty. Processing NCBI FASTA files...")
    sp.process_ncbi_fasta(
        input_root=ncbi_bact_path,
        output_root=bact_genomes_path,
        combine_plasmids="True"
    )
    # combine_plasmids="True" -> 284 FASTA files saved (plasmid-only FNA files skipped)
    # combine_plasmids="False" -> 578 FASTA files saved
else:
    print("Bacterial genomes' directory is not empty. Moving on...")
    pass

# Create or load phage metadata
phage_metadata = sp.load_or_create_curated_phage_metadata(
    phagescope_metadata_path,
    metadata_file
)

# Split the multi-FASTA file manually created using PhageScope
if not Path(phage_genomes_path).exists() or not any(Path(phage_genomes_path).iterdir()):
    print("Phage genomes' directory is empty. Processing PhageScope multi-FASTA file...")
    sp.split_fasta_by_phage_id(
        phagescope_multiFASTA_path,
        phage_genomes_path
    )
else:
    print("Phage genomes' directory is not empty. Moving on...")
    pass

# Derive interaction matrix
interaction_filepath = data_path + f'/interactions/phage_host_interactions{data_suffix}.csv'

phage_host_interaction_matrix = sp.derive_phage_host_interaction_matrix(
    bact_genomes_path,
    phage_metadata,
    interaction_filepath
)

# Run pharokka
sp.pharokka_processing(
    data_path, 
    phage_genomes_sample_path, #phage_genomes_path, 
    pharokka_executable_path, 
    pharokka_db_path, 
    data_suffix,
    threads=16  # 128 cores in yakuza server
)

# Run RBP detection using PhageRBPdetect_v4
#TODO: No GPUs in yakuza server!
sp.process_and_detect_rbps_v4(
    data_path, 
    RBP_detect_model_path, 
    data_suffix,
    gpu_index=0
)

# Run Bakta
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
sp.run_bakta_and_extract_receptors(
    bact_genomes_sample_path, # bact_genomes_path, 
    bakta_results_path, 
    bakta_db_path, 
    gene_keywords, product_keywords,
    data_suffix,
    threads=16,  # 128 cores in yakuza server
    training=True
)


# 3 - FEATURE CONSTRUCTION
# --------------------------------------------------

# ESM-2 features for RBPs
sf.compute_esm2_embeddings_rbp_improved(
    data_path, 
    data_suffix, 
    batch_size=16
)

# ESM-2 features for bacterial receptors 
sf.compute_esm2_embeddings_receptors(
    data_path, 
    bakta_results_path, 
    data_suffix,
    aggregate_by_accession=True,  # Set to true during internship
    batch_size=16,
    skip_long_seq=False
)

# Construct feature matrices
rbp_embeddings_path = data_path + '/esm2_embeddings_rbp' + data_suffix + '.csv'
receptors_embeddings_path = data_path + '/esm2_embeddings_receptors' + data_suffix + '.csv'
interaction_path = data_path + '/interactions'

features_esm2, labels, groups_receptors, groups_phage = sf.construct_feature_matrices_receptors(
    interaction_path, 
    data_suffix, 
    receptors_embeddings_path, 
    rbp_embeddings_path
)

# 4 - TRAINING & EVALUATING MODELS
# --------------------------------------------------

X_full = features_esm2
y_full = labels
groups_receptors = np.array(groups_receptors)
groups_phage = np.array(groups_phage)

imbalance = np.sum(y_full == 1) / np.sum(y_full == 0)

# Hyperparameter tuning with Ray Tune
search_space = {
    "learning_rate": tune.loguniform(1e-3, 0.3),
    "max_depth": tune.choice([3, 5, 7, 9]),
    "n_estimators": tune.choice([100, 200, 300]),
    "scale_pos_weight": tune.choice(1 / imbalance),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.6, 1.0)
}

# Create objective function for Ray Tune
def train_xgb_tune(config, X, y):
    # Stratified train/test split
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
        n_jobs=8
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    tune.report({"auc": auc})

# Run Ray Tune analysis
analysis = tune.run(
    partial(train_xgb_tune, X=X_full, y=y_full),
    config=search_space,
    num_samples=30,  # increase for more exhaustive search
    metric="auc",
    mode="max",
    scheduler=ASHAScheduler(max_t=50, grace_period=5),
    resources_per_trial={"cpu": 8},
    storage_path=f"{data_path}/ray_tune_results",
    name="xgb_receptor_tune"
)

best_config = analysis.get_best_config(metric="auc", mode="max")
print("Best hyperparameters:", best_config)

# Train final model using best config
best_model = XGBClassifier(
    **best_config,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=8
)
best_model.fit(X_full, y_full)
best_model.save_model(f"{data_path}/xgb_receptor_model_tuned{data_suffix}.json")

# Group k-fold cross validation
# GROUP STRATEGY — choose one of: 'receptor', 'phage', or 'both'
group_by = 'both'

if group_by == 'receptor':
    groups = np.array(groups_receptors)
elif group_by == 'phage':
    groups = np.array(groups_phage)
elif group_by == 'both':
    # Define each group as a (phage, receptor) pair
    groups = np.array([f"{b}_{p}" for b, p in zip(groups_receptors, groups_phage)])
else:
    raise ValueError("Invalid grouping strategy selected.")

print(f"Number of unique groups: {len(set(groups))}")

# Convert group names to numeric codes (required for splitting)
unique_groups = list(set(groups))
group_to_id = {g: i for i, g in enumerate(unique_groups)}
group_ids = np.array([group_to_id[g] for g in groups])

k = 5
gkf = GroupKFold(n_splits=k)
results = []
true_labels = []

for train_idx, test_idx in tqdm(gkf.split(X_full, y_full, groups=group_ids), total=k):
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    imbalance = np.sum(y_train == 1) / np.sum(y_train == 0)
    best_config["scale_pos_weight"] = 1 / imbalance
    
    xgb = XGBClassifier(
        **best_config,
        n_jobs=8,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    y_score = xgb.predict_proba(X_test)[:, 1]

    results.append(y_score)
    true_labels.append(y_test)

# Save results
with open(f"{results_path}/serr_receptor_level_kfold_results.pickle", "wb") as f:
    pickle.dump({'scores': results, 'labels': true_labels}, f)


# 5 - READ & INTERPRET RESULTS
# --------------------------------------------------

#TODO: a deeper analysis can be performed locally after running the training process on the server

# Pretty-print the predictions and true labels
print("\nPredicted Scores (results):")
for i, arr in enumerate(results):
    print(f"Fold {i+1}: {np.array2string(arr, precision=4, separator=', ', floatmode='fixed')}")

print("\nGround Truth (true_labels):")
for i, arr in enumerate(true_labels):
    print(f"Fold {i+1}: {np.array2string(arr, separator=', ')}")

# Flatten predictions and labels
all_scores = np.concatenate(results)
all_labels = np.concatenate(true_labels)

# Threshold scores to get predicted classes (you can tune threshold later)
y_pred = (all_scores >= 0.5).astype(int)

# Compute metrics
roc_auc = roc_auc_score(all_labels, all_scores)
pr_auc = average_precision_score(all_labels, all_scores)
f1 = f1_score(all_labels, y_pred)
acc = accuracy_score(all_labels, y_pred)

print("Cross-Validation Metrics:")
print(f"ROC AUC:     {roc_auc:.4f}")
print(f"PR AUC:      {pr_auc:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"Accuracy:    {acc:.4f}")