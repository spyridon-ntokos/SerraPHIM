"""
Created on 17.07.2025

@author: snt 
based on PhageHostLearn by @dimiboeckaerts

SerraPHIM FEATURE CONSTRUCTION
"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import math
import json
import numpy as np
import pandas as pd
import torch
import esm
from tqdm import tqdm
from itertools import product
import random


# 1 - FUNCTIONS
# --------------------------------------------------
def compute_esm2_embeddings_rbp_improved(
    general_path,
    data_suffix='',
    batch_size=16,
    add=False
):
    """
    Computes ESM-2 embeddings for RBPs using batch processing to improve runtime.
    
    Parameters:
    - general_path: Path to the project data folder.
    - data_suffix: Optional suffix to append to saved file name (default='').
    - batch_size: Number of sequences processed per batch (default=16).
    - add: Whether to append new embeddings to an existing file (default=False).
    
    Output:
    - Saves esm2_embeddings_rbp.csv with embeddings for each protein.
    """

    # Load the ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for deterministic results

    # Load the RBP data
    RBPbase = pd.read_csv(f"{general_path}/RBPbase{data_suffix}.csv")
    if add:
        old_embeddings_df = pd.read_csv(f"{general_path}/esm2_embeddings_rbp{data_suffix}.csv")
        processed_ids = set(old_embeddings_df['protein_ID'])
        RBPbase = RBPbase[~RBPbase['protein_ID'].isin(processed_ids)].reset_index(drop=True)
        print(f"Processing {len(RBPbase)} more sequences (add=True)")

    # Prepare sequences for embedding
    sequences = list(zip(RBPbase['protein_ID'], RBPbase['protein_sequence']))
    phage_ids = RBPbase['phage_ID']
    protein_ids = RBPbase['protein_ID']
    
    # Initialize storage for embeddings
    sequence_representations = []
    batch_phage_ids, batch_protein_ids = [], []

    # Process sequences in batches
    bar = tqdm(total=len(sequences), desc="Embedding sequences")
    for i in range(0, len(sequences), batch_size):
        batch_data = sequences[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_phage_ids.extend(phage_ids[i:i + batch_size])
        batch_protein_ids.extend(protein_ids[i:i + batch_size])
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        # Calculate mean representation for each sequence
        for j, (_, seq) in enumerate(batch_data):
            seq_embedding = token_representations[j, 1:len(seq) + 1].mean(0)
            sequence_representations.append(seq_embedding.cpu().numpy())

        bar.update(len(batch_data))
    bar.close()

    # Convert embeddings to DataFrame
    embeddings_df = pd.DataFrame(sequence_representations)
    embeddings_df.insert(0, 'protein_ID', batch_protein_ids)
    embeddings_df.insert(0, 'phage_ID', batch_phage_ids)

    # Append to existing data if needed
    if add:
        embeddings_df = pd.concat([old_embeddings_df, embeddings_df], ignore_index=True)

    # Save embeddings
    embeddings_df.to_csv(f"{general_path}/esm2_embeddings_rbp{data_suffix}.csv", index=False)
    print(f"Saved embeddings to {general_path}/esm2_embeddings_rbp{data_suffix}.csv")


def compute_esm2_embeddings_receptors(
    general_path,
    bakta_results_path,
    data_suffix='',
    batch_size=16,
    aggregate_by_accession=False,
    skip_long_seq=True,
    add=False
):
    """
    Compute ESM-2 embeddings for bacterial receptors using the receptor protein sequences from
    the Bact_receptors{data_suffix}.json file.

    Parameters:
    - general_path: Path to the project data folder.
    - bakta_results_path: Path where Bakta results are stored.
    - data_suffix: Optional suffix to append to file names (default='').
    - batch_size: Number of sequences processed per batch (default=16).
    - aggregate_by_accession: If True, averages all receptor embeddings per accession.
    - skip_long_seq: If True, skip sequences longer than MAX_SEQ_LEN, otherwise process them solo.
    - add: Whether to append new embeddings to an existing file (default=False).

    Output:
    - Saves esm2_embeddings_receptors.csv with embeddings for each protein.
    """

    # Load the ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for deterministic results

    # ESM-2 models scale quadratically in memory with sequence length,
    # and sequences longer than ~1024–2048 residues can easily crash the kernel

    # Constants
    MAX_SEQ_LEN = 1500
    long_data = []
    normal_data = []
    metadata = []
    skipped_proteins = []

    # Load receptor data
    json_path = os.path.join(bakta_results_path, f"Bact_receptors{data_suffix}.json")
    with open(json_path, 'r') as f:
        receptor_dict = json.load(f)

    if add:
        old_path = os.path.join(bakta_results_path, f"esm2_embeddings_receptors{data_suffix}.csv")
        if os.path.exists(old_path):
            old_df = pd.read_csv(old_path)
            processed = set(old_df['accession'] if aggregate_by_accession else old_df['protein_ID'])
            receptor_dict = {
                acc: seqs for acc, seqs in receptor_dict.items()
                if (acc if aggregate_by_accession else f"{acc}_{i}") not in processed
                for i in range(len(seqs))
            }
            print(f"Processing {len(receptor_dict)} new accessions (add=True)")

    # Prepare data for batching
    data = []
    metadata = []  # to track accession and optionally per-sequence ID
    skipped_proteins = []
    for acc, sequences in receptor_dict.items():
        for i, seq in enumerate(sequences):
            if seq.strip() == "":
                continue
            prot_id = f"{acc}_{i}"
            if len(seq) > MAX_SEQ_LEN:
                if skip_long_seq:
                    print(f"Skipping {prot_id}: too long ({len(seq)} AA)")
                    skipped_proteins.append((acc, i, len(seq)))
                    continue
                else:
                    print(f"Will process {prot_id} solo (length={len(seq)} AA)")
                    long_data.append((prot_id, seq))
            else:
                normal_data.append((prot_id, seq))
            metadata.append((acc, prot_id))

    # Log skipped sequences
    if skipped_proteins:
        pd.DataFrame(skipped_proteins, columns=["accession", "index", "length"]).to_csv(
            os.path.join(general_path, f"skipped_long_sequences{data_suffix}.csv"), index=False
        )

    def embed_batch(data_batch, current_batch_size):
        reps_list = []
        for i in tqdm(range(0, len(data_batch), current_batch_size), desc="Embedding batches"):
            batch = data_batch[i:i+current_batch_size]
            labels, strs, tokens = batch_converter(batch)
            with torch.no_grad():
                results = model(tokens, repr_layers=[33], return_contacts=False)
            reps = results["representations"][33]
            for j, (_, seq) in enumerate(batch):
                emb = reps[j, 1:len(seq)+1].mean(0).cpu().numpy()
                reps_list.append(emb)
        return reps_list

    # Run embedding
    sequence_representations = []
    if normal_data:
        sequence_representations.extend(embed_batch(normal_data, batch_size))
    if long_data:
        sequence_representations.extend(embed_batch(long_data, 1))

    # Prepare output DataFrame
    accessions, prot_ids = zip(*metadata)
    embeddings_df = pd.DataFrame(sequence_representations)

    if aggregate_by_accession:
        df = embeddings_df.copy()
        df['accession'] = accessions
        grouped = df.groupby('accession').mean().reset_index()
        output_df = grouped
    else:
        output_df = pd.concat([
            pd.DataFrame({'accession': accessions, 'protein_ID': prot_ids}),
            embeddings_df
        ], axis=1)

    # If add=True, append to previous
    output_path = os.path.join(general_path, f"esm2_embeddings_receptors{data_suffix}.csv")
    if add and os.path.exists(output_path):
        prev_df = pd.read_csv(output_path)
        output_df = pd.concat([prev_df, output_df], ignore_index=True)

    # Save results
    output_df.to_csv(output_path, index=False)
    print(f"Saved embeddings to: {output_path}")


def construct_feature_matrices_receptors(
    interaction_path, 
    data_suffix, 
    receptor_embeddings_path, 
    rbp_embeddings_path
):
    """
    Construct feature matrices from full interaction matrix (1s and 0s) at species-level.

    Assumes:
    - receptor_embeddings: per receptor protein with 'accession' column.
    - rbp_embeddings: per RBP protein with 'phage_ID' and 'protein_ID' columns.
    - interaction matrix contains binary 1s and 0s only.

    Returns:
    - features, labels, groups_receptors, groups_phage
    """

    # Load data
    receptor_embeddings = pd.read_csv(receptor_embeddings_path)
    RBP_embeddings = pd.read_csv(rbp_embeddings_path)
    interactions = pd.read_csv(f"{interaction_path}/phage_host_interactions{data_suffix}.csv", index_col=0)

    # Average RBP embeddings per phage
    phage_ids = []
    avg_rbps = []
    for phage_id in set(RBP_embeddings['phage_ID']):
        subset = RBP_embeddings[RBP_embeddings['phage_ID'] == phage_id].iloc[:, 2:]
        avg_rbps.append(np.mean(subset.values, axis=0))
        phage_ids.append(phage_id)
    rbp_df = pd.concat([pd.DataFrame({'phage_ID': phage_ids}), pd.DataFrame(avg_rbps)], axis=1)

    # Construct features
    features = []
    labels = []
    groups_receptors = []
    groups_phage = []

    # Pre-extract the accessions and phage IDs
    accessions = receptor_embeddings['accession']
    phage_ids = rbp_df['phage_ID']

    # Total number of iterations
    total = len(accessions) * len(phage_ids)

    # Main loop with progress bar
    for i, j in tqdm(product(range(len(accessions)), range(len(phage_ids))), total=total, desc="Building feature matrix"):
        acc = accessions[i].split("_", 1)[0]
        phage_id = phage_ids[j]
        
        if acc in interactions.index and phage_id in interactions.columns:
            interaction = interactions.at[acc, phage_id]
            combined = pd.concat([
                receptor_embeddings.iloc[i, 1:],  # skip 'accession'
                rbp_df.iloc[j, 1:]  # skip 'phage_ID'
            ])
            features.append(combined.values)
            labels.append(int(interaction))
            groups_receptors.append(acc)
            groups_phage.append(phage_id)

    return np.array(features), np.array(labels), groups_receptors, groups_phage

