# Sc-RNASeq analysis of neutrophils and progenitors from healthy and tumor-bearing mice 

## Data availability: 
The data can be found on GEO under accession number: GSE285117

## Scripts: 

**1.Packages_and_functions.R**: Packages and functions that need to be loaded for the following analysis 

**2.Data_preprocessing.R**: Integration of cell-count matrices into the Seurat workflow and initial quality cutoffs 

**3.1.Annotation_and_extraction_of_neutrophils.R**: Annotation to extract neutrophils from different tissues

**3.2.Annotation_and_extraction_progenitors.R**: Annotation and extraction of progenitors from bone marrow samples

**4.2.Neutrophils_Progenitor_integration.R**: Integration and clustering of neutrophils and progenitors from different tissues and phenotypes 

**4.Neutrophils_integration.R**: Integration and clustering of neutrophils from different tissues and phenotypes

**5.Neutrophils_cross_species_comparison.R**: Analysis of orthologs from genes that were found specific for human TAN and BN within mouse neutrophil clusters 

**6.2.Neutrophils_slingshot_analysis_between_phenotype.R**: Pseudotime trajectory analysis of neutrophils from bone marrow to the colon/tumor between phenotypes 

**6.Neutrophils_slingshot_trajectory_analysis.R**: Pseudotime trajectory analysis of neutrophils from bone marrow to the colon/tumor 

**7.1.Neutrophils_prog_DEG_analysis.R**: DEG analysis of progenitors comparing healthy and tumor 

**7.Neutrophils_DEG_analysis.R**: DEG analysis of neutrophils from blood and bone marrow comparing healthy and tumor 

**8.Neutrophils_cluster10.R**: Analysis of cluster 10 in terms of gene expression and cell cycle scoring
