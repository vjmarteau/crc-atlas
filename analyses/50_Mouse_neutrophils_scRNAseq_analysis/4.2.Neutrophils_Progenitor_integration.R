########## This code integrates neutrophils from all tissues ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
neutrophils <- readRDS("/data/khandl/CRC_atlas/neutrophils_integrated.rds")
progenitor <- readRDS( "/scratch/khandl/CRC_atlas/seurat_objects/blood_bm_progenitors.rds")

current.cluster.ids <- c("blood_tumor", "blood_no_tumor","BM_tumor", "BM_no_tumor")
new.cluster.ids <- c("blood", "blood","BM","BM")
progenitor$tissue <- plyr::mapvalues(x = progenitor$condition, from = current.cluster.ids, to = new.cluster.ids)

current.cluster.ids <- c("blood_tumor", "blood_no_tumor","BM_tumor", "BM_no_tumor")
new.cluster.ids <- c("tumor","healthy","tumor", "healthy")
progenitor$phenotype <- plyr::mapvalues(x = progenitor$condition, from = current.cluster.ids, to = new.cluster.ids)

current.cluster.ids <- c("blood_tumor", "blood_no_tumor","BM_tumor", "BM_no_tumor")
new.cluster.ids <- c("blood_tumor", "blood_wt","BM_tumor","BM_wt")
progenitor$condition <- plyr::mapvalues(x = progenitor$condition, from = current.cluster.ids, to = new.cluster.ids)

##### merging
obj <- merge(neutrophils,progenitor,
             add.cell.ids = c("neut","progenitor"))
obj <- JoinLayers(obj)

##### add experiment identity 
current.cluster.ids <- c("adjacent_colon_wt","adult_colon_wt","mets_wt","tumor_wt", "blood_tumor", "blood_wt","BM_tumor", "BM_wt")
new.cluster.ids <- c("exp1","exp1","exp1","exp1", "exp2", "exp2","exp2", "exp2")
obj$experiment <- plyr::mapvalues(x = obj$condition, from = current.cluster.ids, to = new.cluster.ids)

##### pre-processing 
obj <- NormalizeData(obj,normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj,vars.to.regress = c("nFeature_RNA","nCount_RNA","percent.mt"))
obj <- RunPCA(object = obj, features = VariableFeatures(object =obj), npcs = 20, verbose = FALSE)

##### fastMNN integration 
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$condition)
obj <- IntegrateLayers(object = obj, method = FastMNNIntegration,new.reduction = "integrated.mnn",
                       verbose = FALSE)
ElbowPlot(obj)
obj <- FindNeighbors(obj, reduction = "integrated.mnn", dims = 1:10)
obj <- FindClusters(obj, resolution = 1, cluster.name = "mnn.clusters", algorithm = 2)
obj <- RunUMAP(obj, reduction = "integrated.mnn", dims = 1:10, reduction.name = "umap",seed.use = 5)
DimPlot(obj,reduction = "umap",raster=FALSE, label = TRUE, label.size = 8) 

FeaturePlot(obj, features = c("S100a8","Msi2","Meis1","Cd34","Epx","Elane","Cebpe","Csf1r","Irf8"))

DotPlot(obj, features = c("S100a8","Msi2","Meis1","Cd34","Epx","Elane","Cebpe","Csf1r","Irf8"),dot.scale = 10, scale = TRUE, assay = "RNA") + 
  theme(legend.title = element_text(size = 20), legend.text = element_text(size = 20)) + 
  theme(title = element_text(size = 20))+ theme(axis.text = element_text(size = 20)) + theme(axis.text.x = element_text(angle = 90)) 

##### Extract neutrophils, GMPs and ProNeutro and re-cluster 
Idents(obj) <- "mnn.clusters"
obj <- subset(obj, idents = c(0:9, 12,14,15,16,18))
obj <- JoinLayers(obj)
obj <- NormalizeData(obj,normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj,vars.to.regress = c("nFeature_RNA","nCount_RNA","percent.mt"))
obj <- RunPCA(object = obj, features = VariableFeatures(object =obj), npcs = 20, verbose = FALSE)

obj[["RNA"]] <- split(obj[["RNA"]], f = obj$experiment)
obj <- IntegrateLayers(object = obj, method = FastMNNIntegration,new.reduction = "integrated.mnn",
                       verbose = FALSE)
ElbowPlot(obj)
obj <- FindNeighbors(obj, reduction = "integrated.mnn", dims = 1:10)
obj <- FindClusters(obj, resolution = 0.5, cluster.name = "mnn.clusters", algorithm = 2)
obj <- RunUMAP(obj, reduction = "integrated.mnn", dims = 1:10, reduction.name = "umap",seed.use = 5)
DimPlot(obj,reduction = "umap",raster=FALSE, label = TRUE, label.size = 8) 
DimPlot(obj,reduction = "umap",raster=FALSE, label = TRUE, label.size = 8,split.by = "phenotype") 

FeaturePlot(obj, features = c("S100a8","Msi2","Meis1","Cd34","Elane","Cebpe"))

markers <- c("S100a8","Msi2","Meis1","Cd34","Elane","Cebpe")
for(i in markers) {
  p <- FeaturePlot(obj, features = i, reduction = "umap", pt.size = 0.1) + scale_color_gradientn( colours = c('grey', 'darkred'),  limits = c(0,5))
  ggsave(paste0("/scratch/khandl/CRC_atlas/figures/",i,".svg"), width = 8, height = 5, plot = p)
}

p <- DimPlot(obj,reduction = "umap",group.by = "mnn.clusters",raster=FALSE, label= TRUE, label.size = 5) 
ggsave("/scratch/khandl/CRC_atlas/figures/umap_clustering.svg", width = 8, height = 5, plot = p)

p <- DimPlot(obj,reduction = "umap",group.by = "tissue",raster=FALSE, label= TRUE, label.size = 5, cols = c("#EA8F0C","#4A5FDD", "#359B35")) 
ggsave("/scratch/khandl/CRC_atlas/figures/umap_spit_tissue.svg", width = 12, height = 5, plot = p)

p <- DimPlot(obj,reduction = "umap",group.by = "phenotype",raster=FALSE, label= TRUE, label.size = 5, cols = c("#4A5FDD","#EA8F0C")) 
ggsave("/scratch/khandl/CRC_atlas/figures/umap_spit_phenotype.svg", width = 12, height = 5, plot = p)

## rename clusters 
current.cluster.ids <- c(0:11)
new.cluster.ids <- c("Neutrophils", "Neutrophils","Neutrophils","Neutrophils", "Neutrophils","Neutrophils", "Neutrophils","Neutrophils","Neutrophils","ProNeutro","GMP","Neutrophils")
obj$annotation <- plyr::mapvalues(x = obj$mnn.clusters, from = current.cluster.ids, to = new.cluster.ids)

p <- DimPlot(obj,reduction = "umap",group.by = "annotation",raster=FALSE, label= TRUE, label.size = 5) 
ggsave("/scratch/khandl/CRC_atlas/figures/umap_anno.svg", width = 8, height = 5, plot = p)

## marker genes used for annotation
Idents(obj) <- "annotation"
p <- DotPlot(obj, features = c("Msi2","Meis1","Cd34","Elane","Cebpe","S100a9", "S100a8"),dot.scale = 10, scale = FALSE, assay = "RNA",cols = c("white","darkred")) + 
  theme(legend.title = element_text(size = 20), legend.text = element_text(size = 20)) + 
  theme(title = element_text(size = 20))+ theme(axis.text = element_text(size = 10)) + theme(axis.text.x = element_text(angle = 45)) 
ggsave("/scratch/khandl/CRC_atlas/figures/DotPlot_markers.svg", width = 10, height = 6, plot = p)

##### DEGs analysis 
obj <- JoinLayers(obj)
obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
Idents(obj) <- "annotation"
markers <- FindAllMarkers(object = obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, assay = "RNA", slot = "data")

## plot top DEGs
marker_genes <- c(markers %>% group_by(cluster) %>% top_n(n =10, wt = avg_log2FC))$gene

# heatmap per cluster and condition 
heatmap_goi_coi(obj, "mnn.clusters",marker_genes,c("markers"), 
                length(marker_genes),
                c("#6383EA"),
                c(markers="#6383EA"),T,T)


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
ggsave("/scratch/khandl/CRC_atlas/figures/cell_cycle_plot.svg", width = 8, height = 6, plot = p)

##### save integrated R object
saveRDS(obj, "/data/khandl/CRC_atlas/neutrophils_progenitors_annotated.rds")
obj <- readRDS("/data/khandl/CRC_atlas/neutrophils_progenitors_annotated.rds")
