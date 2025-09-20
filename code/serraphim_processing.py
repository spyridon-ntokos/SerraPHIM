"""
SerraPHIM — DATA PROCESSING
=================================

Created on 17.07.2025
Author: snt (based on PhageHostLearn by @dimiboeckaerts)

This module contains utility functions used in the SerraPHIM project to:
- process downloaded NCBI bacterial fasta archives into per-genome FASTA files,
- curate and filter phage metadata TSVs from PhageScope,
- split multi-FASTA phage collections into single-file FASTAs,
- derive a species-aware phage-host interaction matrix from metadata,
- run Pharokka on phage genomes and extract gene sequences,
- run the PhageRBPdetect_v4 inference workflow against Pharokka outputs,
- run Bakta on bacterial genomes and extract receptor-related proteins.

The functions here are intended for batch processing in reproducible pipelines.
They make minimal changes to input files and write outputs into the `data/` tree.

Dependencies
------------
- Python 3.8+
- biopython (Bio)
- pandas
- tqdm
- transformers (for model loading in RBP detection)
- esm (optional depending on other flows)
- Pharokka and Bakta installed / available on PATH for the respective functions

License / contribution
----------------------
This file is intended to be included in the SerraPHIM public repository.
Please include appropriate LICENSE and citation information in the repo root.

"""

# 0 - LIBRARIES
# --------------------------------------------------
import os
import sys
import re
import json
import gzip
import subprocess
import pandas as pd
import torch
from pathlib import Path
from io import StringIO
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 1 - FUNCTIONS
# --------------------------------------------------
def process_ncbi_fasta(
    input_root="./data/ncbi_bacteria/genbank/bacteria", 
    output_root="./data/bacteria_genomes",
    combine_plasmids=False
):
    """
    Convert NCBI .fna.gz GenBank/RefSeq dumps into per-genome FASTA files.

    This function walks `input_root` recursively, opens any `*.fna.gz` file found,
    and writes one or more FASTA files into `output_root`. The output filenames are
    made filesystem-safe and attempt to preserve the first record header as a
    human-readable identifier.

    Behavior:
    - If `combine_plasmids` is False (default) each sequence in the archive will
      become a separate FASTA file (one-per-record).
    - If `combine_plasmids` is True the function attempts to detect a main
      chromosome record and any plasmid records, and writes either a single
      `<name>_+_plasmids.fasta` (main + plasmids) or `<name>.fasta` when there are
      no plasmids. If the header format is unexpected the function saves the
      records but prints a warning.

    Parameters
    ----------
    input_root : str or Path
        Directory containing downloaded NCBI `.fna.gz` files (recursively searched).
    output_root : str or Path
        Directory to save individual genome FASTA files.
    combine_plasmids : bool, optional
        If True, attempt to group chromosomes + plasmids into one file per genome.
        Default: False.

    Returns
    -------
    None
        Files are written directly to `output_root`. Progress is printed to stdout.

    Notes
    -----
    - The function is conservative; when a header is ambiguous it issues a warning
      and falls back to saving the provided records.
    - The cleaning of file names replaces characters not in [A-Za-z0-9_.-] with '_'.

    Example
    -------
    process_ncbi_fasta("~/downloads/ncbi", "./data/bacteria_genomes", combine_plasmids=True)
    """

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Traverse and process all .fna.gz files
    for gz_path in input_root.rglob("*.fna.gz"):
        print(f"Processing: {gz_path}")
        with gzip.open(gz_path, "rt") as handle:
            records = list(SeqIO.parse(handle, "fasta"))

            if combine_plasmids:
                # # Use first sequence header for naming
                # main_header = records[0].description.split(",")[0]
                # clean_name = re.sub(r'[^A-Za-z0-9_.-]', '_', main_header)
                # filename = f"{clean_name}_+_plasmids.fasta"
                # output_path = output_root / filename

                # with open(output_path, "w") as out_handle:
                #     SeqIO.write(records, out_handle, "fasta")
                # print(f"Saved combined: {output_path}")
                first_record = records[0]
                first_desc = first_record.description
                first_header = first_desc.split(",")[0]
                clean_name = re.sub(r'[^A-Za-z0-9_.-]', '_', first_header)

                # --- CASE 1: complete genome ---
                matches = [
                    "complete genome", 
                    "complete sequence", 
                    "whole genome shotgun sequence"
                ]
                if any(x in first_desc for x in matches):
                    if len(records) > 1:
                        # Combine main + plasmids into one file
                        filename = f"{clean_name}_+_plasmids.fasta"
                        records_to_write = records
                    else:
                        # Only one sequence (main genome)
                        filename = f"{clean_name}.fasta"
                        records_to_write = [first_record]

                # --- CASE 2: chromosome ---
                elif "chromosome: " in first_desc:
                    plasmid_records = []

                    for rec in records[1:]:
                        record_parts = rec.description.split()
                        rec_desc = " ".join(record_parts[-2:]).strip().lower()

                        if rec_desc.startswith("plasmid"):
                            plasmid_records.append(rec)
                        else:
                            print(f"WARNING: Check {gz_path}")

                    if plasmid_records:
                        filename = f"{clean_name}_+_plasmids.fasta"
                        records_to_write = [first_record] + plasmid_records
                    else:
                        filename = f"{clean_name}.fasta"
                        records_to_write = [first_record]

                # --- CASE 3: starts with plasmid only ---
                elif "plasmid: " in first_desc:
                    if len(records) > 1:
                        print(f"WARNING: Check {gz_path}")
                    # Ignore plasmid-only files
                    print(f"Skipped (plasmid-only record): {gz_path}")
                    continue

                else:
                    print(f"Unrecognized header description in: {gz_path}. Saving as is...")
                    filename = f"{clean_name}.fasta"
                    records_to_write = records

                # Write combined/single FASTA with a single header (first record only)
                output_path = output_root / filename
                with open(output_path, "w") as out_handle:
                    SeqIO.write(records_to_write, out_handle, "fasta")
                print(f"Saved: {output_path}")
            else:
                for record in records:
                    # Construct a clean and informative filename from the FASTA header
                    record_parts = record.description.split()
                    raw_name = " ".join(record_parts[:-2]).strip().rstrip(',') 
                    clean_name = re.sub(r'[^A-Za-z0-9_.-]', '_', raw_name)
                    filename = f"{clean_name}.fasta"

                    # Save file
                    output_path = output_root / filename
                    with open(output_path, "w") as out_handle:
                        SeqIO.write(record, out_handle, "fasta")
                    print(f"Saved: {output_path}")

    print("NCBI data processing complete.")


def load_or_create_curated_phage_metadata(
    input_dir="./data/PhageScope_metadata",
    output_file="serratia_complete-HQ_virulent_phage_metadata.tsv"
):
    """
    Load an existing curated phage metadata dataframe or create one by filtering TSVs.

    This helper reads all TSV files under `input_dir` and filters rows relevant to
    Serratia hosts, complete/high-quality genomes, and virulent lifestyle. If a
    consolidated TSV (`output_file`) already exists it is returned directly.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing raw metadata TSV files (one or more).
    output_file : str
        Relative filename under `input_dir` to save the consolidated table.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered metadata. If no matching phages are
        found an empty DataFrame is returned.

    Side effects
    ------------
    - Writes a TSV file to `input_dir/output_file` when it is created.

    Example
    -------
    df = load_or_create_curated_phage_metadata("./data/PhageScope_metadata")
    """
    output_path = Path(input_dir + '/' + output_file)
    input_dir = Path(input_dir)

    if output_path.exists():
        print(f"Metadata file already exists: {output_file}")
        return pd.read_csv(output_path, sep="\t")

    print("Metadata file not found. Creating it from TSVs...")

    filtered_frames = []

    for tsv_file in input_dir.glob("*.tsv"):
        print(f"Reading: {tsv_file.name}")
        df = pd.read_csv(tsv_file, sep="\t")

        # Add missing 'Phage_source' column
        if "Phage_source" not in df.columns:
            source = tsv_file.stem.split("_")[0].upper()
            df["Phage_source"] = source

        # Apply filters
        filtered_df = df[
            df["Host"].astype(str).str.startswith("Serratia", na=False) &
            df["Completeness"].isin(["Complete", "High-quality"]) &
            (df["Lifestyle"] == "virulent")
        ]

        if not filtered_df.empty:
            filtered_frames.append(filtered_df)

    if filtered_frames:
        result_df = pd.concat(filtered_frames, ignore_index=True)
        
        # Sanitize Phage_IDs for safe filenames
        result_df["Phage_ID"] = result_df["Phage_ID"].astype(str).apply(
            lambda pid: re.sub(r'[^A-Za-z0-9_.-]', '_', pid.strip())
        )
        result_df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved filtered metadata to: {output_path}")
    else:
        print("No matching phages found with specified filters.")
        result_df = pd.DataFrame()  # empty DataFrame

    return result_df


def split_fasta_by_phage_id(
    big_fasta_path, 
    output_dir
):
    """
    Split a multi-record FASTA into one FASTA file per sequence.

    Parameters
    ----------
    big_fasta_path : str or Path
        Path to the multi-FASTA file to split.
    output_dir : str or Path
        Directory where individual FASTA files will be written.

    Returns
    -------
    None
        Files are written to `output_dir`. Prints a counter of written sequences.

    Example
    -------
    split_fasta_by_phage_id("phages_all.fasta", "./data/phage_genomes")
    """
    big_fasta_path = Path(big_fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading multi-FASTA from: {big_fasta_path}")

    count = 0
    for record in SeqIO.parse(big_fasta_path, "fasta"):
        raw_id = record.id.strip()
        safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', raw_id)
        output_path = output_dir / f"{safe_id}.fasta"

        with open(output_path, "w") as out_handle:
            SeqIO.write(record, out_handle, "fasta")
        count += 1

    print(f"Split complete: {count} sequences saved to {output_dir}")


def derive_phage_host_interaction_matrix(
    bacterial_genome_dir="./data/bacteria_genomes",
    phage_metadata_df=None,
    output_csv_path="./data/phage_host_interactions.csv"
):
    """
    Derive a species-aware phage-host interaction matrix from phage metadata.

    The function builds a binary matrix (0/1) whose rows are bacterial accession IDs
    parsed from filenames in `bacterial_genome_dir` and whose columns are phage IDs
    from `phage_metadata_df`. Cells are set to 1 when the phage metadata indicates
    that the phage infects the species associated with the accession.

    Important
    ---------
    - This function expects the bacterial genome filenames to follow the convention
      `'ACCESSION_<species>_<...>.fasta'`. The code extracts the first two tokens
      after the accession to form a `Species_Tag` such as `Serratia_marcescens`.
    - `phage_metadata_df` must contain at least the columns `Phage_ID` and `Host`.

    Parameters
    ----------
    bacterial_genome_dir : str or Path
        Directory containing per-accession bacterial FASTA files.
    phage_metadata_df : pandas.DataFrame
        DataFrame with phage metadata including 'Phage_ID' and 'Host'.
    output_csv_path : str or Path
        Path to save the resulting interaction CSV.

    Returns
    -------
    pandas.DataFrame
        The constructed interaction matrix (rows: accession IDs, columns: phage IDs).

    Raises
    ------
    ValueError
        If `phage_metadata_df` is None or empty.

    Example
    -------
    df = derive_phage_host_interaction_matrix("./data/bacteria_genomes", phage_metadata_df)
    """
    output_path = Path(output_csv_path)
    bacterial_genome_dir = Path(bacterial_genome_dir)

    if output_path.exists():
        print(f"Interaction matrix already exists at: {output_csv_path}")
        return pd.read_csv(output_path, index_col=0)

    if phage_metadata_df is None or phage_metadata_df.empty:
        raise ValueError("Phage metadata dataframe is missing or empty.")

    # Step 1: Parse bacterial genome filenames
    bacteria_info = []
    for file in bacterial_genome_dir.glob("*.fasta"):
        parts = file.stem.split("_", 1)
        if len(parts) < 2:
            continue
        accession_id = parts[0]
        
        # Extract only species part: look for the first two tokens in the rest
        rest = parts[1]
        rest_parts = rest.split("_")
        if len(rest_parts) < 2:
            continue
        species = f"{rest_parts[0]}_{rest_parts[1]}"  # e.g. Serratia_marcescens
        bacteria_info.append((accession_id, species))

    # Create DataFrame with rows = bacteria accession IDs
    bacterial_df = pd.DataFrame(bacteria_info, columns=["Accession_ID", "Species_Tag"])
    bacterial_df.set_index("Accession_ID", inplace=True)

    # Step 2: Create empty interaction matrix
    phage_ids = phage_metadata_df["Phage_ID"].tolist()
    interaction_df = pd.DataFrame(
        data=0,
        index=bacterial_df.index,
        columns=phage_ids
    )

    # Step 3: Fill interaction values
    for _, row in phage_metadata_df.iterrows():
        phage_id = row["Phage_ID"]
        host = str(row["Host"]).strip()

        if host in ["Serratia", "Serratia sp."]:
            target_species = ["Serratia_marcescens", "Serratia_plymuthica"]
        elif host == "Serratia marcescens":
            target_species = ["Serratia_marcescens"]
        elif host == "Serratia plymuthica":
            target_species = ["Serratia_plymuthica"]
        else:
            target_species = []

        for accession_id, species_tag in bacteria_info:
            if species_tag in target_species:
                interaction_df.at[accession_id, phage_id] = 1

    # Step 4: Save and return
    interaction_df.to_csv(output_path)
    print(f"Saved interaction matrix to: {output_path}")
    return interaction_df


def pharokka_processing(
    general_path, 
    phage_genomes_path, 
    pharokka_executable, 
    databases_path, 
    data_suffix='',
    threads=None,
    add=False
):
    """
    Run Pharokka on phage genomes and extract nucleotide gene sequences.

    This function iterates over FASTA files in `phage_genomes_path`, calls Pharokka
    (the `phanotate`/`pharokka` pipeline) to annotate genes and then parses the
    generated `pharokka.gff` file to extract gene coordinates and sequences.
    The output is a CSV saved as `phage_genes{data_suffix}.csv` under `general_path`
    with columns: ['phage_ID', 'gene_ID', 'gene_sequence'].

    Parameters
    ----------
    general_path : str
        Root path of the project (used to write `pharokka_results` and output CSV).
    phage_genomes_path : str
        Directory containing phage genome FASTA files (one FASTA per phage).
    pharokka_executable : str
        Command or full path to the Pharokka executable (e.g. `pharokka`).
    databases_path : str
        Path to Pharokka's database folder (passed with `-d`).
    data_suffix : str, optional
        Suffix appended to output CSV filename (default: '').
    threads : int, optional
        Number of threads to pass to Pharokka. Defaults to all available CPUs.
    add : bool, optional
        If True, append newly processed genes to an existing `phage_genes{suffix}.csv`
        instead of overwriting.

    Returns
    -------
    None
        Writes `phage_genes{data_suffix}.csv` to `general_path`.

    Notes
    -----
    - Pharokka must be installed and working on the PATH, or provide the full path.
    - The function expects Pharokka to produce `pharokka.gff` in the per-phage
      results folder; it will read that file and extract features not starting
      with `tRNA`.
    - On failure of Pharokka (non-zero exit), a CalledProcessError is raised.

    Example
    -------
    pharokka_processing('/proj', '/proj/phage_genomes', '/usr/bin/pharokka', '/proj/pharokka_db')
    """

    if threads == None:
        max_threads = os.cpu_count() or 1  # Use all available CPU cores
    else:
        max_threads = threads
    phage_files = os.listdir(phage_genomes_path)
    if add:
        RBPbase = pd.read_csv(os.path.join(general_path, f'RBPbase{data_suffix}.csv'))
        phage_ids = set(RBPbase['phage_ID'])
        phage_files = [x for x in phage_files if os.path.splitext(x)[0] not in phage_ids]
        print(f'Processing {len(phage_files)} more phages (add=True)')

    name_list, gene_ids, gene_list = [], [], []
    bar = tqdm(total=len(phage_files), position=0, leave=True)

    for file in phage_files:
        count = 1
        file_path = os.path.join(phage_genomes_path, file)
        output_dir = os.path.join(general_path, 'pharokka_results', os.path.splitext(file)[0])
        os.makedirs(output_dir, exist_ok=True)

        cmd = (f"{pharokka_executable} -i {file_path} -o {output_dir} "
               f"-d {databases_path} -f -t {max_threads}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, bufsize=1)

        # Print Pharokka output in real time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
        process.stdout.close()
        retcode = process.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, cmd)

        # Parse GFF and extract gene sequences
        gff_file = os.path.join(output_dir, "pharokka.gff")
        phage_name = os.path.splitext(file)[0]
        with open(gff_file, 'r') as gff:
            lines = [line.strip() for line in gff if not line.startswith('#')]
            internal_phage_name = str(lines[0]).split('\t')[0]
            # print("internal phage name:", internal_phage_name)
            lines = [line.strip() for line in lines if line.startswith(internal_phage_name)]

        sequence = str(SeqIO.read(file_path, 'fasta').seq)
        for line in lines:
            cols = line.split('\t')
            if str(cols[1]).startswith('tRNA'):
                continue
            start, stop, strand = int(cols[3]), int(cols[4]), cols[6]
            if strand == '+':
                gene_seq = sequence[start-1:stop]
            else:
                gene_seq = str(Seq(sequence[start-1:stop]).reverse_complement())
            metadata = cols[8]
            matches = re.search('ID=(.*);transl_table.*', metadata)
            gene_id = matches.group(1)
            name_list.append(phage_name)
            gene_ids.append(gene_id)
            gene_list.append(gene_seq)
            count += 1
        
        bar.update(1)

    bar.close()

    genebase = pd.DataFrame({'phage_ID': name_list, 'gene_ID': gene_ids, 'gene_sequence': gene_list})

    if add:
        old_genebase = pd.read_csv(os.path.join(general_path, f'phage_genes{data_suffix}.csv'))
        genebase = pd.concat([old_genebase, genebase], axis=0)

    genebase.to_csv(os.path.join(general_path, f'phage_genes{data_suffix}.csv'), index=False)

    print("Pharokka processing complete.")


def process_and_detect_rbps_v4(
    general_path, 
    model_path, 
    data_suffix='', 
    gpu_index=0,
    threads=16,
    add=False
):
    """
    Run PhageRBPdetect_v4 inference on Pharokka protein predictions and save RBPs.

    This function iterates per-phage over the Pharokka result folders, loads the
    fine-tuned ESM-2 sequence-classification model (PhageRBPdetect_v4), performs
    per-protein inference and collects the proteins predicted as RBPs. The
    resulting table `RBPbase{data_suffix}.csv` contains columns:
        ['phage_ID', 'protein_ID', 'protein_sequence', 'dna_sequence', 'score']

    Device handling
    ---------------
    - If a CUDA-compatible GPU is available, the function will attempt to use it.
    - If no CUDA GPU is present, it sets PyTorch thread limits to `threads` to
      control CPU parallelism.

    Parameters
    ----------
    general_path : str
        Project root containing `pharokka_results` and where outputs are written.
    model_path : str
        Path to the fine-tuned sequence-classifier model directory (HF transformers
        format) used by PhageRBPdetect_v4.
    data_suffix : str, optional
        Suffix appended to the output filename (default '').
    gpu_index : int, optional
        CUDA device index to expose via CUDA_VISIBLE_DEVICES (default: 0).
    threads : int, optional
        Number of CPU threads to use when running on CPU (default: 16).
    add : bool, optional
        If True, append new predictions to existing `RBPbase{data_suffix}.csv`.

    Returns
    -------
    None
        Writes `RBPbase{data_suffix}.csv` into `general_path`.

    Raises
    ------
    FileNotFoundError
        If no `phage_genes{data_suffix}.csv` exists in `general_path`.

    Notes
    -----
    - The function expects Pharokka results to contain `phanotate.faa` files with
      protein FASTA records for each phage subdirectory.
    - The `model_path` should point to a transformers-compatible model directory
      (the function uses `AutoTokenizer` and `AutoModelForSequenceClassification`).
    - If your model requires `trust_remote_code=True` you should adjust the loader,
      or pre-import the custom model class into the environment.

    Example
    -------
    process_and_detect_rbps_v4('/proj', 'RBPdetect_v4_ESMfine', data_suffix='_inference')
    """
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cpu":
        torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)

    print(f"Using device: {device} (with {threads} threads if CPU)")

    # Load model & tokenizer (allow custom EsmTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path #, trust_remote_code=True
    ).to(device).eval()

    # Define directories
    pharokka_results_dir = os.path.join(general_path, 'pharokka_results')

    # Load the gene database
    gene_db_path = os.path.join(general_path, f'phage_genes{data_suffix}.csv')
    if not os.path.exists(gene_db_path):
        raise FileNotFoundError(f"Gene database file not found: {gene_db_path}")
    gene_db = pd.read_csv(gene_db_path)

    # Load previously processed phages if adding more
    processed_phages = set()
    if add and os.path.exists(os.path.join(general_path, f'RBPbase{data_suffix}.csv')):
        rbpbase = pd.read_csv(os.path.join(general_path, f'RBPbase{data_suffix}.csv'))
        processed_phages = set(rbpbase['phage_ID'])

    # Collect all pharokka subdirectories
    pharokka_dirs = [d for d in os.listdir(pharokka_results_dir)
                     if os.path.isdir(os.path.join(pharokka_results_dir, d))]

    # Process and detect RBPs
    all_predictions = []
    for phage_dir in tqdm(pharokka_dirs, desc="Processing phages"):
        phage_name = phage_dir
        if add and phage_name in processed_phages:
            continue

        pharokka_subdir = os.path.join(pharokka_results_dir, phage_dir)
        faa_file = os.path.join(pharokka_subdir, 'phanotate.faa')
        if not os.path.exists(faa_file):
            print(f"Skipping {phage_name}: No 'phanotate.faa' file found.")
            continue

        # Parse protein and DNA sequences
        proteins = list(SeqIO.parse(faa_file, 'fasta'))
        protein_seqs = [str(record.seq) for record in proteins]
        protein_ids = [record.id for record in proteins]

        # Perform inference
        predictions = []
        scores = []
        for sequence in tqdm(protein_seqs, desc=f"Inference for {phage_name}"):
            encoding = tokenizer(sequence, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                output = model(**encoding)
                predictions.append(int(output.logits.argmax(-1)))
                scores.append(float(output.logits.softmax(-1)[:, 1]))

        # Collect only RBP-predicted results
        rbp_data = []
        for protein_id, protein_seq, pred, score in zip(protein_ids, protein_seqs, predictions, scores):
            if pred == 1:  # Filter for RBPs
                # Fetch DNA sequence from gene database
                dna_sequence = gene_db.loc[
                    (gene_db['phage_ID'] == phage_name) & 
                    (gene_db['gene_ID'] == protein_id), 'gene_sequence'
                ].values
                dna_sequence = dna_sequence[0] if len(dna_sequence) > 0 else 'NA'

                rbp_data.append({
                    'phage_ID': phage_name,
                    'protein_ID': protein_id,
                    'protein_sequence': protein_seq,
                    'dna_sequence': dna_sequence,
                    'score': score  # Using new model's score
                })

        all_predictions.extend(rbp_data)

    # Save the RBPbase file
    if all_predictions:
        rbpbase = pd.DataFrame(all_predictions)
        if add and os.path.exists(os.path.join(general_path, f'RBPbase{data_suffix}.csv')):
            old_rbpbase = pd.read_csv(os.path.join(general_path, f'RBPbase{data_suffix}.csv'))
            rbpbase = pd.concat([old_rbpbase, rbpbase], ignore_index=True)

        rbpbase.to_csv(os.path.join(general_path, f'RBPbase{data_suffix}.csv'), index=False)
        print(f"RBP detection complete. Results saved to 'RBPbase{data_suffix}.csv'")
    else:
        print("No RBP predictions were generated.")


def run_bakta_and_extract_receptors(
    bacteria_path, 
    bakta_results_path, 
    db_path,
    gene_keywords, 
    product_keywords, 
    data_suffix='', 
    threads=None,
    training=True
):
    """
    Run Bakta on bacterial genome files and extract putative receptor proteins.

    This wrapper runs Bakta on every `*.fasta` in `bacteria_path` (unless a per-genome
    `bakta_results/<genome_name>` folder already exists), then parses the Bakta TSV
    output to filter genes likely involved in surface structures using `gene_keywords`
    and `product_keywords`. Two JSON files are produced:
      - Bact_receptors{data_suffix}.json : minimal map {accession: [protein sequences]}
      - Bact_receptors_verbose{data_suffix}.json : detailed list with locus tag, product,
        DbXrefs and protein sequence for each selected gene.

    Keyword rules
    -------------
    - Gene keywords are matched against the `Gene` column using substring matching.
    - Product keywords support three modes:
        * multi-word exact-match where all words must be present (order and adjacency
          not required),
        * prefix/suffix indicators using a trailing `-` or leading `-` in the keyword
          (e.g. `-porin` matches any token ending with 'porin'),
        * simple whole-word presence otherwise.

    Parameters
    ----------
    bacteria_path : str
        Directory with bacterial FASTA files to annotate with Bakta.
    bakta_results_path : str
        Directory where Bakta will write per-genome results (and from which results
        will be read).
    db_path : str
        Path to the Bakta database (passed with `--db`).
    gene_keywords : list[str]
        List of keywords to search in the Bakta `Gene` column (substring matching).
    product_keywords : list[str]
        List of keywords for the Bakta `Product` column. Supports the advanced rules
        described above (prefix/suffix/multi-word).
    data_suffix : str, optional
        Optional suffix for output files (default '').
    threads : int or None, optional
        Number of threads to use when invoking Bakta. If None the system CPU count
        is used.
    training : bool, optional
        If True, Bakta is run with many annotation steps skipped (faster) suitable
        for training pipelines. If False, Bakta is run with defaults.

    Returns
    -------
    None
        Writes two JSON files into `bakta_results_path`: the minimal receptor dict
        and the verbose receptor metadata dict.

    Notes
    -----
    - Bakta must be installed and available on PATH or provided as a wrapper command.
    - If a `bakta_results/<genome_name>` directory already exists the function will
      skip running Bakta for that genome and proceed to parse the existing outputs.
    - The function expects Bakta to produce a `<genome_name>.tsv` and
      `<genome_name>.faa` in the per-genome results folder.

    Example
    -------
    run_bakta_and_extract_receptors(
        "./data/bacteria_genomes", "./data/bakta_results", "/opt/bakta_db",
        gene_keywords=["waaL"], product_keywords=["-porin", "outer membrane"])
    """
    os.makedirs(bakta_results_path, exist_ok=True)

    if threads == None:
        max_threads = os.cpu_count() or 1  # Use all available CPU cores
    else:
        max_threads = threads

    # Dictionaries to store the results
    receptors_dict = {}
    verbose_receptors_dict = {}

    # Helper function for product / gene keyword matching
    def match_field_keywords(field, keywords):
        #print("field:", field)
        for keyword in keywords:
            field_words = field.lower().split()
            if keyword.startswith("-"):  # Suffix keyword
                suffix_keyword = keyword[1:].lower()
                if field and any(field_word.endswith(suffix_keyword) for field_word in field_words):
                    #print(f"keyword: '{keyword}' | suffix_keyword: '{suffix_keyword}' | field: '{field}'")
                    return True
            elif keyword.endswith("-"):  # Prefix keyword
                prefix_keyword = keyword[:-1].lower()
                if field and any(field_word.startswith(prefix_keyword) for field_word in field_words):
                    #print(f"keyword: '{keyword}' | prefix_keyword: '{prefix_keyword}' | field: '{field}'")
                    return True
            else:  # Multi-word keyword
                words = keyword.lower().split()
                if all(word in field.lower() for word in words):
                    #print(f"keyword: '{keyword}' | field: '{field}'")
                    return True
        return False

    # Iterate over all .fasta files in the bacteria_path
    fasta_files = [f for f in os.listdir(bacteria_path) if f.endswith('.fasta')]
    for fasta_file in tqdm(fasta_files, desc="Processing genomes"):
        genome_path = os.path.join(bacteria_path, fasta_file)
        genome_name = os.path.splitext(fasta_file)[0]
        output_path = os.path.join(bakta_results_path, genome_name)
        tsv_file = os.path.join(output_path, f"{genome_name}.tsv")
        faa_file = os.path.join(output_path, f"{genome_name}.faa")
        
        if not os.path.exists(output_path):
            # Run Bakta
            bakta_command = [
                #"apptainer run /stornext/GL_CARPACCIO/home/HPC/opt/singularity/bakta-1.11.0.sif",
                "bakta",
                "--db", db_path,
                "--output", output_path,
                "--force",
                "--threads", str(max_threads),
            ]
            if training:
                bakta_command.extend([
                    "--skip-trna", "--skip-tmrna", "--skip-rrna",
                    "--skip-ncrna", "--skip-ncrna-region", "--skip-crispr", "--skip-pseudo",
                    "--skip-sorf", "--skip-gap", "--skip-plot"
                ])
            bakta_command.append(genome_path)
            cmd = ' '.join(bakta_command)

            print("Command:", cmd)

            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                    text=True, bufsize=1)

            # Print Bakta output in real time
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                sys.stdout.flush()
            process.stdout.close()
            retcode = process.wait()
            if retcode != 0:
                raise subprocess.CalledProcessError(retcode, cmd)
        else:
            print(f"Skipping Bakta for {genome_name}: output already exists.")

        # Parse the .tsv file to filter receptor-related genes
        if not os.path.exists(tsv_file):
            print(f"No TSV file found for {genome_name}. Skipping...")
            continue
        
        # Read the TSV file and set the header manually
        with open(tsv_file, 'r') as f:
            lines = f.readlines()

        # Extract the header (after "#") and the data rows
        header_line = [line for line in lines if line.startswith("#")][-1]
        header = header_line.lstrip("#").strip().split("\t")
        data_lines = [line for line in lines if not line.startswith("#")]

        # Create the DataFrame
        data_str = "\n".join(data_lines)
        tsv_data = pd.read_csv(StringIO(data_str), sep='\t', header=None, names=header)

        # Filter genes by keywords
        gene_filter = tsv_data['Gene'].apply(lambda g: match_field_keywords(str(g), gene_keywords))
        product_filter = tsv_data['Product'].apply(lambda p: match_field_keywords(p, product_keywords))
        # print("Gene filter:", gene_filter)
        # print("Gene filtered TSV:", tsv_data[gene_filter])
        # print("Product filter:", product_filter)
        # print("Product filtered TSV:", tsv_data[product_filter])
        receptor_genes = tsv_data[gene_filter | product_filter]
        print("Receptor genes:", receptor_genes)

        # Extract protein sequences for receptor-related genes
        if not os.path.exists(faa_file):
            print(f"No FAA file found for {genome_name}. Skipping...")
            continue
        
        receptor_proteins = []
        verbose_receptors = []
        for record in SeqIO.parse(faa_file, 'fasta'):
            if record.id in receptor_genes['Locus Tag'].values:
                receptor_proteins.append(str(record.seq))
                gene_row = receptor_genes[receptor_genes['Locus Tag'] == record.id].iloc[0]
                verbose_receptors.append({
                    "Locus Tag": record.id,
                    "Gene": str(gene_row["Gene"]),
                    "Product": gene_row["Product"],
                    "Protein Sequence": str(record.seq),
                    "DbXrefs": gene_row["DbXrefs"]
                })
        
        # Save to the dictionaries
        receptors_dict[genome_name] = receptor_proteins
        verbose_receptors_dict[genome_name] = verbose_receptors

    # Dump the results to a JSON file
    json_output_path = os.path.join(bakta_results_path, f"Bact_receptors{data_suffix}.json")
    verbose_json_output_path = os.path.join(bakta_results_path, f"Bact_receptors_verbose{data_suffix}.json")
    with open(json_output_path, 'w') as json_file:
        json.dump(receptors_dict, json_file, indent=4)
    with open(verbose_json_output_path, 'w') as verbose_json_file:
        json.dump(verbose_receptors_dict, verbose_json_file, indent=4)
    
    print(f"Receptor extraction complete. Results saved to {json_output_path} and {verbose_json_output_path}")


