########## This code applies marker genes found in human neutrophil data to mouse clusters  ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
obj <- readRDS(file ="/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_integrated.rds")

##### plot human gene markers in mouse data 
markers_human <- c("FCN1","MMP9","PGLYRP1","RFLNB","S100A12","MTARC1","LYZ","RBP7","PLP2","C1orf162", #BN1
                   "SIGLEC10", "JADE1","PIGB","SET","RPL27A","FAS","TRIM8", "KLF3","PHOSPHO1","ZFP36L2",#BN2
                   "PPARG", "ANXA2","LMNA","PARD3B","MYO1E", "MYO1D","SNX9","PCCA","RHPN2","DACH1",#BN3
                   "CENPU","EREG","AREG","ERI2","PTX3","IL1B","NR4A3", "HOMER1","OLR1","NLRP3",#TAN1
                   "TGM2","CCL3","HLA-DRA","CCL4","CXCL2","CD22","CCL4L2","CCRL2","PLIN2","CDKN1A",#TAN2
                   "HILPDA","CCL20","HSPA1B", "SLAMF7","IL1A","C15orf48", "CCL3L3",#TAN3 
                   "MT-ND4", "RHOH","N4BP2","CD69" #TAN4 
                   )

### convert human markers to mouse symbols
markers_human_df <- as.data.frame(markers_human)
rownames(markers_human_df) <- markers_human_df$markers_human

markers_mouse_ortholog_df <- convert_orthologs(gene_df = markers_human_df,
                                        gene_input = "rownames", 
                                        gene_output = "rownames", 
                                        input_species = "human",
                                        output_species = "mouse",
                                        non121_strategy = "drop_both_species",
                                        method = "gprofiler") 
markers_mouse_orthologs <- rownames(markers_mouse_ortholog_df)
print(markers_mouse_ortholog_df)

# print the missing orhthologues 
markers_human[!markers_human %in%markers_mouse_ortholog_df$markers_human ]
#however there are orthologues for CCL3, CCL4, CXCL2 and HSPA1B --> will include them 

### plot BN and TAN markers in mouse data 
print(markers_mouse_ortholog_df)
markers_mouse <- c("Fcnb","Mmp9","Pglyrp1","Rflnb","Mtarc1","Lyz1","Rbp7","Plp2","I830077J02Rik",#BN1 9
                   "Siglecg", "Jade1","Pigb","Rpl27a","Fas","Trim8", "Klf3","Phospho1","Zfp36l2",#BN2 9
              "Pparg", "Anxa2","Lmna","Pard3b","Myo1e", "Myo1d","Snx9","Pcca","Rhpn2","Dach1",#BN3 10
              "Cenpu","Ereg","Areg","Eri2","Ptx3","Il1b","Nr4a3", "Homer1","Olr1","Nlrp3",#TAN1 10
              "Tgm2","Ccl3", "H2-Ea","Ccl4","Cxcl2", "Cd22","Ccrl2","Plin2","Cdkn1a",#TAN2 9
              "Hilpda","Ccl20","Hspa1b", "Slamf7","Il1a","AA467197",#TAN3 6 
              "mt-Nd4", "Rhoh","N4bp2","Cd69" #TAN4 4
)

# heatmap per cluster and condition 
heatmap_goi_coi(obj, "mnn.clusters",markers_mouse,c("BN1","BN2","BN3","TAN1","TAN2","TAN3","TAN4"), 
                c(9,9,10,10,9,6,4),
                c("#1051E2",  "#F49211","#0A9B32",  "#BF1029",
                  "#9410BF",  "#774937","#F28FE4"),
                c(BN1="#1051E2", BN2= "#F49211", BN3= "#0A9B32", TAN1= "#BF1029",
                  TAN2= "#9410BF", TAN3= "#774937", TAN4= "#F28FE4"),F,T)
