########## This code applies DEG analysis of neutrophils from bone marrow and blood - healthy vs. tumor ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
obj <- readRDS(file ="/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_integrated.rds")

##### bone marrow 
# extract only BM 
Idents(obj) <- "tissue"
sub <- subset(obj, idents = "BM")

sub <- NormalizeData(sub, normalization.method = "LogNormalize",
                               scale.factor = 10000,
                               margin = 1, assay = "RNA")

DefaultAssay(sub) <- "RNA"
Idents(sub) <- "condition"
degs <- FindMarkers(object = sub, ident.1 = "BM_no_tumor", ident.2 = "BM_tumor", only.pos = FALSE, min.pct = 0.25, 
                       logfc.threshold = 0.2,slot = "data")
write.csv(degs, "/scratch/khandl/CRC_atlas/DEG_analysis/neutrophils_BM_healthy_vs_BM_tumor.csv")

# plot in volcano 
EnhancedVolcano(degs,
                lab = paste0("italic('", rownames(degs), "')"),
                x = 'avg_log2FC',
                y = 'p_val_adj',
                title = 'BM healthy vs. BM  tumor',
                pCutoff = 0.05,
                FCcutoff = 0.25,
                pointSize = 3.0,
                labSize = 4,
                col=c('black', 'black', 'black', 'red3'),
                colAlpha = 1,
                parseLabels = TRUE)

##### blood 
# extract only blood 
Idents(obj) <- "tissue"
sub <- subset(obj, idents = "blood")

sub <- NormalizeData(sub, normalization.method = "LogNormalize",
                     scale.factor = 10000,
                     margin = 1, assay = "RNA")

DefaultAssay(sub) <- "RNA"
Idents(sub) <- "condition"
degs <- FindMarkers(object = sub, ident.1 = "blood_no_tumor", ident.2 = "blood_tumor", only.pos = FALSE, min.pct = 0.25, 
                    logfc.threshold = 0.2,slot = "data")
write.csv(degs, "/scratch/khandl/CRC_atlas/DEG_analysis/neutrophils_blood_healthy_vs_blood_tumor.csv")

# plot in volcano 
EnhancedVolcano(degs,
                lab = paste0("italic('", rownames(degs), "')"),
                x = 'avg_log2FC',
                y = 'p_val_adj',
                title = 'blood healthy vs. blood tumor',
                pCutoff = 0.05,
                FCcutoff = 0.25,
                pointSize = 3.0,
                labSize = 4,
                col=c('black', 'black', 'black', 'red3'),
                colAlpha = 1,
                parseLabels = TRUE)
