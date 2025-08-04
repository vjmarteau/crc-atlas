########## This code investigates the identify of cluster 10 ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
obj <- readRDS("/data/khandl/CRC_atlas/neutrophils_integrated.rds")
DimPlot(obj,reduction = "umap",raster=FALSE, label = TRUE) 

##### DEG cluster 10 vs. cluster 6
DefaultAssay(obj) <- "RNA"
Idents(obj) <- "mnn.clusters"
degs <- FindMarkers(object = obj, ident.1 = "10", ident.2 = "6", only.pos = FALSE, min.pct = 0.25, 
                    logfc.threshold = 0.2,slot = "data")
write.csv(degs, "/scratch/khandl/CRC_atlas/cluster10/Cluster10_vs_cluster6_DEGs.csv")

# plot in volcano 
EnhancedVolcano(degs,lab = paste0("italic('", rownames(degs), "')"),
                x = 'avg_log2FC', y = 'p_val_adj',title = 'Cluster 10 vs. cluster 6',
                pCutoff = 0.05, FCcutoff = 0.25, pointSize = 3.0,
                labSize = 4, col=c('black', 'black', 'black', 'red3'),colAlpha = 1,
                parseLabels = TRUE)

##### cell cycle score 
# Load Seurat’s default cell cycle gene sets
cc.genes <- Seurat::cc.genes.updated.2019

s.features_mouse <- c("Mcm5","Pcna","Tyms", "Fen1","Mcm2","Mcm4","Rrm1","Ung","Gins2","Mcm6","Cdca7","Dtl","Prim1",
                      "Uhrf1","Hells","Rfc2","Rpa2","Nasp","Rad51ap1","Gmnn","Wdr76","Slbp","Ccne2","Ubr7","Pold3",
                      "Msh2","Atad2","Rad51","Rrm2","Cdc45","Cdc6","Exo1","Tipin","Dscc1","Blm","Casp8ap2","Usp1","Clspn",
                      "Pola1","Chaf1b","Brip1","E2f8")

g2m.features_mouse <- c("Hmgb2","Cdk1","Nusap1","Ube2c","Birc5","Tpx2","Top2a","Ndc80","Cks2",
                        "Nuf2","Cks1b","Mki67","Tmpo","Cenpf","Tacc3","Smc4","Ccnb2","Ckap2l","Ckap2","Aurkb","Bub1",
                        "Kif11","Anp32e","Tubb4b","Gtse1","Kif20b","Hjurp","Cdca3","Cdc20","Ttk","Cdc25c","Kif2c","Rangap1",
                        "Ncapd2","Dlgap5","Cdca2","Cdca8","Ect2","Kif23","Hmmr","Aurka","Psrc1","Anln","Lbr","Ckap5","Cenpe",
                        "Ctcf","Nek2","G2e3","Gas2l3","Cbx5","Cenpa")

# Score cell cycle
obj <- CellCycleScoring(
  obj,
  s.features = s.features_mouse,
  g2m.features = g2m.features_mouse,
  set.ident = TRUE  # optionally set the identity class to the cell cycle phase
)

p <- DimPlot(obj, group.by = "Phase", reduction = "umap") + 
  ggtitle("UMAP Colored by Cell Cycle Phase")
ggsave("/scratch/khandl/CRC_atlas/cluster10/cell_cycle_plot.svg", width = 8, height = 6, plot = p)


