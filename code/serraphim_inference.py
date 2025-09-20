"""
Created on 17.07.2025

@author: snt 
based on PhageHostLearn by @dimiboeckaerts

SerraPHIM INFERENCE
"""

# 1 - INITIAL SETUP
# --------------------------------------------------
# data paths
main_path = '/home/snt/PhageHostLearn-main'
data_path = main_path + '/data'
code_path = main_path + '/code'

phages_path = data_path + '/phage_genomes'
bacteria_path = data_path + '/bacteria_genomes'

pfam_path = code_path + '/RBPdetect_phageRBPs.hmm'
xgb_path = code_path + '/RBPdetect_xgb_hmm.json'

kaptive_db_path = main_path + '/Kaptive/reference_database/Klebsiella_k_locus_primary_reference.gbk'
pharokka_db_path = main_path + '/pharokka/databases'
bakta_db_path = main_path + '/bakta_db/db'

bakta_results_path = data_path + '/bakta_results'

suffix = '_inference'

# software paths
hmmer_path = main_path + '/hmmer-3.4'
phanotate_path = main_path + '/phl-venv/bin/phanotate.py'
pharokka_path = main_path + '/pharokka/bin/pharokka.py'
#pharokka_path = main_path + '/phl-venv/bin/pharokka.py'
RBP_detect_model_path = main_path + '/RBPdetect_v4_ESMfine'

# 2 - DATA PROCESSING
# --------------------------------------------------
import serraphim_processing as sp

# run pharokka
sp.pharokka_processing(data_path, phages_path, pharokka_path, pharokka_db_path, data_suffix=suffix)

# run RBP detection using PhageRBPdetect_v4
sp.process_and_detect_rbps_v4(data_path, RBP_detect_model_path, data_suffix=suffix)

# run Bakta
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
    bacteria_path, 
    bakta_results_path, 
    bakta_db_path, 
    gene_keywords, product_keywords,
    data_suffix=suffix, 
    training=True
)


# 3 - FEATURE CONSTRUCTION
# --------------------------------------------------
import serraphim_features as sf

# ESM-2 features for RBPs
sf.compute_esm2_embeddings_rbp_improved(data_path, data_suffix=suffix, batch_size=16)

# ESM-2 features for bacterial receptors (~4:30 hours for 4301 receptor-related proteins)
sf.compute_esm2_embeddings_receptors(data_path, bakta_results_path, data_suffix=suffix, batch_size=16)

# Construct feature matrices
rbp_embeddings_path = data_path + '/esm2_embeddings_rbp' + suffix + '.csv'
receptors_embeddings_path = data_path + '/esm2_embeddings_receptors' + suffix + '.csv'
features_esm2, groups_bact = sf.construct_feature_matrices(
    data_path, 
    suffix, 
    receptors_embeddings_path, 
    rbp_embeddings_path, 
    mode='test'
)


# 4 - PREDICT & RANK NEW INTERACTIONS
# --------------------------------------------------


# 5 - READ & INTERPRET RESULTS
# --------------------------------------------------
