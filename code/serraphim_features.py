"""
SerraPHIM — FEATURE CONSTRUCTION
================================

Created on 17.07.2025
Author: snt (based on PhageHostLearn by @dimiboeckaerts)

This module contains functions to construct feature representations used by the
SerraPHIM pipeline. Specifically, it provides utilities to:

- compute ESM-2 protein embeddings for phage RBPs (batched and efficient),
- compute ESM-2 embeddings for bacterial receptor candidate proteins (from Bakta output),
- construct labelled training feature matrices using receptor ↔ phage interaction data,
- construct inference-only feature matrices pairing every receptor with every phage.

Intended use
------------
Embeddings generated here are consumed by downstream modeling code (XGBoost,
cross-validation and inference). Embeddings are saved to CSV files for
reproducibility and to allow large-scale workflows to be resumed.

Important notes / caveats
-------------------------
- The ESM-2 ("t33_650M_UR50D") model used here is large — embedding memory
  usage grows approximately quadratically with sequence length. The functions
  include mechanisms to skip or handle very long protein sequences (parameter
  `skip_long_seq` and `MAX_SEQ_LEN`).
- Batching is applied to improve throughput; adjust `batch_size` to match
  available RAM/GPUs/CPUs.
- The functions expect the JSON produced by the Bakta wrapper:
    `Bact_receptors{data_suffix}.json`
  to exist when computing receptor embeddings.
- These functions assume a simple CSV structure for RBP embeddings and receptor
  embeddings; downstream feature construction concatenates receptor and phage
  vectors in a fixed order.

Dependencies
------------
- Python 3.8+
- pandas
- numpy
- torch
- esm (esm-2 pretrained utilities)
- tqdm

License / contribution
----------------------
Add LICENSE and contribution guidelines in the repo root when publishing.

"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import json
import numpy as np
import pandas as pd
import torch
import esm
from tqdm import tqdm
from itertools import product


# 1 - FUNCTIONS
# --------------------------------------------------
def compute_esm2_embeddings_rbp_improved(
    general_path,
    data_suffix='',
    batch_size=16,
    add=False
):
    """
    Compute ESM-2 embeddings for phage Receptor Binding Proteins (RBPs).

    This function loads the RBPs listed in `RBPbase{data_suffix}.csv` (expected
    to contain at least the columns: ['phage_ID', 'protein_ID', 'protein_sequence'])
    and computes fixed-length sequence embeddings using the ESM-2 pretrained
    model `esm2_t33_650M_UR50D`. Embeddings are computed in batches for
    efficiency and written to:
        {general_path}/esm2_embeddings_rbp{data_suffix}.csv

    The saved CSV contains:
      - 'phage_ID' : phage identifier (string)
      - 'protein_ID' : protein identifier (string)
      - <embedding columns> : numeric embedding vector columns (one column per dimension)

    Parameters
    ----------
    general_path : str
        Root path where `RBPbase{data_suffix}.csv` is read and embedding CSV is written.
    data_suffix : str, optional
        Suffix appended to input/output filenames (e.g. '_inference'). Default: ''.
    batch_size : int, optional
        Number of sequences sent to the model at once (adjust to the available
        memory). Larger batches increase throughput but require more RAM. Default: 16.
    add : bool, optional
        If True and a previous embeddings CSV exists, the function will append
        embeddings only for RBPs not already processed (determined by 'protein_ID').
        Default: False.

    Returns
    -------
    None
        Writes embeddings CSV to disk.

    Notes
    -----
    - The function uses `esm.pretrained.esm2_t33_650M_UR50D()` to load the model
      and the corresponding batch converter.
    - Embeddings are computed as the mean of the residue representations for each
      sequence (`representations[33]`, mean over residues).
    - If GPU(s) are available and PyTorch is configured to use them, ESM will
      use the GPU automatically; if running on CPU, lower the `batch_size`.
    - The embedding CSV may be large; store and handle with care.

    Example
    -------
    compute_esm2_embeddings_rbp_improved('/project/data', data_suffix='_v1', batch_size=32)
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
    Compute ESM-2 embeddings for bacterial receptor candidate proteins.

    This function loads receptor protein sequences from the Bakta-derived JSON:
        {bakta_results_path}/Bact_receptors{data_suffix}.json
    The JSON is expected to map accession -> [protein sequences]. The function
    computes per-protein ESM-2 embeddings and either saves one row per protein
    or averages embeddings per accession (control via `aggregate_by_accession`).

    Output is saved to:
        {general_path}/esm2_embeddings_receptors{data_suffix}.csv

    Parameters
    ----------
    general_path : str
        Root path where the output CSV will be written (used for naming).
    bakta_results_path : str
        Directory where `Bact_receptors{data_suffix}.json` is located.
    data_suffix : str, optional
        Suffix appended to input/output filenames (default '').
    batch_size : int, optional
        Number of sequences processed per batch for normal-length sequences.
        Default: 16.
    aggregate_by_accession : bool, optional
        If True, average all receptor embeddings for a given accession and produce
        one row per accession. If False, keep one row per protein with columns:
            ['accession', 'protein_ID', <embedding columns>]
        Default: False.
    skip_long_seq : bool, optional
        If True, skip sequences longer than MAX_SEQ_LEN (1500 aa) and record them
        in `{general_path}/skipped_long_sequences{data_suffix}.csv`. If False, long
        sequences are processed solo (batch size = 1). Default: True.
    add : bool, optional
        If True and a previous embeddings CSV exists, new embeddings are appended
        keeping previously computed rows. Default: False.

    Returns
    -------
    None
        Writes embeddings CSV to disk. Also writes a CSV of skipped long sequences
        if any were skipped.

    Notes
    -----
    - The function uses `esm.pretrained.esm2_t33_650M_UR50D()` and the residue
      mean pooling strategy (`representations[33]`).
    - ESM-2 memory scales strongly with sequence length; use `skip_long_seq=True`
      to avoid OOM on typical machines. On large-memory servers set
      `skip_long_seq=False` and ensure appropriate batching.
    - When `aggregate_by_accession=True` the resulting CSV contains a column
      'accession' and averaged embedding columns; otherwise it contains both
      'accession' and 'protein_ID' as identifiers.

    Example
    -------
    compute_esm2_embeddings_receptors('/proj', '/proj/bakta_results', data_suffix='_v1',
                                     batch_size=32, aggregate_by_accession=True)
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
    Build training feature matrices by pairing receptor embeddings with phage embeddings.

    This function expects:
      - receptor_embeddings_path -> CSV with per-protein or per-accession receptor embeddings
        (column 'accession' + embedding columns if per-protein; if per-protein, the index
        order should match the accession/protein pairs),
      - rbp_embeddings_path -> CSV with per-RBP embeddings containing 'phage_ID' and 'protein_ID'
        columns.

    The function averages RBP embeddings per phage (to produce a single phage vector) and
    then iterates over every receptor accession × phage pair. If an interaction matrix
    `{interaction_path}/phage_host_interactions{data_suffix}.csv` is present, pairs for which
    an interaction is recorded (0/1) are added to the feature list. Returned outputs:
        features : numpy.ndarray shape (N_pairs, dim_receptor+dim_rbp)
        labels : numpy.ndarray shape (N_pairs,) with values {0,1}
        groups_receptors : list of accession identifiers (one per row in `features`)
        groups_phage : list of phage identifiers (one per row in `features`)

    Parameters
    ----------
    interaction_path : str
        Directory or path prefix where `phage_host_interactions{data_suffix}.csv` is located.
    data_suffix : str
        Suffix used to find the interaction CSV (e.g. '_inference' or '').
    receptor_embeddings_path : str
        CSV path to receptor embeddings (as produced by compute_esm2_embeddings_receptors).
    rbp_embeddings_path : str
        CSV path to RBP embeddings (as produced by compute_esm2_embeddings_rbp_improved).

    Returns
    -------
    tuple
        (features, labels, groups_receptors, groups_phage)

    Notes
    -----
    - The interaction CSV must contain rows indexed by accession and columns by phage ID,
      with values 0 or 1.
    - The function uses tqdm to report progress; building the full pairwise matrix can
      be costly for large receptor × phage counts.
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


def construct_feature_matrices_inference(
    receptor_embeddings_path, 
    rbp_embeddings_path
):
    """
    Build inference-only feature matrix pairing every receptor accession with every phage.

    This helper constructs the complete pairwise (host receptor accession × phage)
    feature set using the receptor embeddings and the averaged-per-phage RBP embeddings.
    No labels are required or returned.

    Returns
    -------
    features : numpy.ndarray
        Array of concatenated receptor + phage embedding vectors (one row per pair).
    groups_hosts : list[str]
        Host accession identifiers per row (repeated for each phage).
    groups_phages : list[str]
        Phage identifiers per row (repeated per host).

    Example
    -------
    features, groups_hosts, groups_phages = construct_feature_matrices_inference(
        'data/esm2_embeddings_receptors.csv',
        'data/esm2_embeddings_rbp.csv'
    )
    """
    receptor_embeddings = pd.read_csv(receptor_embeddings_path)
    RBP_embeddings = pd.read_csv(rbp_embeddings_path)

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
    groups_hosts = []
    groups_phages = []

    accessions = receptor_embeddings['accession']
    phage_ids = rbp_df['phage_ID']

    for i, acc in enumerate(accessions):
        receptor_emb = receptor_embeddings.iloc[i, 1:].values  # skip 'accession'
        for j, phage_id in enumerate(phage_ids):
            rbp_emb = rbp_df.iloc[j, 1:].values  # skip 'phage_ID'
            combined = np.concatenate([receptor_emb, rbp_emb])
            features.append(combined)
            groups_hosts.append(acc)
            groups_phages.append(phage_id)

    return np.array(features), groups_hosts, groups_phages

