# SerraPHIM

**SerraPHIM (Serratia Phage–Host Interaction Model)** is a machine learning framework for predicting interactions between *Serratia* phages and bacterial hosts, developed based on [PhageHostLearn](https://github.com/dimiboeckaerts/PhageHostLearn).  

This repository provides code and Jupyter notebooks for:
- Training SerraPHIM with phage/bacteria genome data  
- Running inference on new genomes  
- Ranking predicted phage–host interactions  

---

## 📂 Project Setup

After cloning the repository, please create the following directories in the **root project folder**:

```
results/  
data/ncbi_bacteria/  
data/bacteria_genomes/  
data/phage_genomes/  
```

---

## 🧬 Download Serratia Genomes

To populate the `ncbi_bacteria/` folder with *Serratia* genomes from NCBI, run:

```
pip install ncbi-genome-download  
ncbi-genome-download \  
    --section genbank \  
    --formats fasta \  
    --assembly-levels complete \  
    --species-taxid 615,82996 \  
    --output-folder ./data/ncbi_bacteria \  
    bacteria  
```

---

## ⚙️ Dependencies

SerraPHIM relies on several external tools for phage/bacteria genome annotation.  
You can install them conveniently via **conda** (recommended), or via **pip** + manual dependency setup.

- **Pharokka** (for phage genome annotation)  
  - Dependencies: MMseqs2, MinCED, ARAGORN, Mash, Dnaapler  

- **Bakta** (for bacterial genome annotation)  
  - Dependencies: AMRFinderPlus, PILER-CR, Diamond, NCBI-BLAST+  

If you install via `pip`, make sure all dependencies are on your `$PATH` (e.g. under `/usr/local/bin/`).  

---

## 📓 Usage

The easiest way to run the code is through the provided **Jupyter notebook**, i.e. `serraphim_interactive_yakuza.ipynb`.  
Adjust the following paths to match your environment:  

- `main_path`  
- `pharokka_executable_path`  
- `pharokka_db_path`  
- `bakta_db_path`  

Also update:  
- `n_cpus` → number of CPUs to use for training or inference   

---

## 🚀 Notes

- Your environment preparation (virtualenv, conda, WSL, etc.) may differ; choose the method that best fits your setup.  
- For training, a larger dataset improves robustness.  
- Inference generates ranked predictions of phage–host interactions, which can be compared with experimental spot assay results.  

---

## 📖 Citation

If you use this project in your research, please cite:  
> Ntokos, S. *Repurposing PhageHostLearn to Serratia marcescens and Serratia plymuthica bacteria & phages* Internship Report, 2025.  
