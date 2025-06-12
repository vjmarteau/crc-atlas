########## This code annotates and extracts precursors from the blood and bm ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

### load R object 
obj <- readRDS(file = "/scratch/khandl/CRC_atlas/seurat_objects/pre_processing_blood_bm.rds")

### pre-processing and clustering 
obj <- NormalizeData(obj,normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj,vars.to.regress = c("nFeature_RNA","nCount_RNA","percent.mt"))
obj <- RunPCA(object = obj, features = VariableFeatures(object =obj), npcs = 20, verbose = FALSE)
obj <- FindNeighbors(object = obj, dims = 1:15)
obj <- FindClusters(obj, resolution = 0.5, random.seed = 5, algorithm = 2)
obj <- RunUMAP(obj, dims = 1:15, seed.use = 5)
DimPlot(obj, label = TRUE, label.size = 8)

obj <- JoinLayers(obj)

### SingleR automatic annotation 
mouse.se <- celldex::ImmGenData()
results <- SingleR(test = as.SingleCellExperiment(obj), ref = mouse.se, labels = mouse.se$label.main)
plotScoreHeatmap(results)
cell.types <- unique(results$pruned.labels)
Idents(obj) <- "seurat_clusters"
lapply(cell.types, function(x) project_annotation_to_umap(x, results, obj))

##### markers from Garner et al 2025, Cancer Cell: https://www.sciencedirect.com/science/article/pii/S1535610825001667
markers <- c("Msi2","Meis1","Cd34","Cebpa","Elane","Cebpe","S100a8","Ly6c2","Csf1r","Irf8","Klf4","H2-Aa","Flt3","Dntt",
             "Il7r","Gata1","Pf4","Klf1","Ms4a2")

DotPlot(obj, features = markers,dot.scale = 10, scale = TRUE, assay = "RNA") + 
  theme(legend.title = element_text(size = 20), legend.text = element_text(size = 20)) + 
  theme(title = element_text(size = 20))+ theme(axis.text = element_text(size = 20)) + theme(axis.text.x = element_text(angle = 90)) 

FeaturePlot(obj, features = c("Procr","Klf1","Pf4","Csf1r","Siglech","Dntt","Vpreb3","Elane","H2-Aa","Prg2","Mcpt8","Cma1"),raster=FALSE,reduction = "umap")
FeaturePlot(obj,features = c("Msi2","Meis1","Cd34","Siglecf","Epx"))
FeaturePlot(obj,features = c("Csf1r"))

### DEGs 
obj <- NormalizeData(obj, normalization.method = "LogNormalize",scale.factor = 10000,margin = 1, assay = "RNA")
Idents(obj) <- "seurat_clusters"
markers <- FindAllMarkers(object = obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, assay = "RNA", layer = "data")
View(markers %>% group_by(cluster) %>% top_n(n =10, wt = avg_log2FC))

### rename clusters
current.cluster.ids <- c(0:18)
new.cluster.ids <- c("other","other","other","other","other","Monocytes","other","other","progenitor","progenitor","EoP","Monocytes","other",
                     "other","other","other","other","other","other")
obj$annotation <- plyr::mapvalues(x = obj$seurat_clusters, from = current.cluster.ids, to = new.cluster.ids)

### extract projenitors
Idents(obj) <- "annotation"
sub <- subset(obj, idents = c("progenitor","EoP"))

### save object 
saveRDS(sub, "/scratch/khandl/CRC_atlas/seurat_objects/blood_bm_progenitors.rds")





