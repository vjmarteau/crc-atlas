########## This code annotates and extracts neutrophils from colon, tumor, blood and bone marrow datasets ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### colon and tumor 
### load R object 
obj <- readRDS(file = "/scratch/khandl/CRC_atlas/seurat_objects/pre_processing_tumor_colon.rds")

### pre-processing
obj <- NormalizeData(obj,normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj,vars.to.regress = c("nFeature_RNA","nCount_RNA","percent.mt"))
obj <- RunPCA(object = obj, features = VariableFeatures(object =obj), npcs = 20, verbose = FALSE)

### fastMNN integration 
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$condition)
obj <- IntegrateLayers(object = obj, method = FastMNNIntegration,new.reduction = "integrated.mnn",
                       verbose = FALSE)
obj <- FindNeighbors(obj, reduction = "integrated.mnn", dims = 1:15)
obj <- FindClusters(obj, resolution = 0.5, cluster.name = "mnn.clusters", algorithm = 2)
obj <- RunUMAP(obj, reduction = "integrated.mnn", dims = 1:15, reduction.name = "umap.mnn",return.model = TRUE)
DimPlot(obj,reduction = "umap.mnn", label = TRUE)

obj <- JoinLayers(obj)

### SingleR automatic annotation 
mouse.se <- celldex::ImmGenData()
results <- SingleR(test = as.SingleCellExperiment(obj), ref = mouse.se, labels = mouse.se$label.main)
plotScoreHeatmap(results)
cell.types <- unique(results$pruned.labels)
Idents(obj) <- "mnn.clusters"
lapply(cell.types, function(x) project_annotation_to_umap_fastMNN(x, results, obj))

### Neutrophil marker genes 
FeaturePlot(obj, features = c("S100a8","S100a9"))

### DEGs per cluster 
obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
Idents(obj) <- "seurat_clusters"
markers <- FindAllMarkers(object = obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, assay = "RNA", slot = "data")
View(markers %>% group_by(cluster) %>% top_n(n =10, wt = avg_log2FC))
#cluster 12 shows T cell marker and cluster 15 B cell markers --> only 0 and 4 are neutrophils 
VlnPlot(obj, features = "nFeature_RNA")

### extract neutrophils 
neutrophils <- subset(obj, idents = c(0,4))
neutrophils$anntation <- "Neutrophils"

### save object 
saveRDS(neutrophils, "/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_colon_tumor.rds")

##### blood and BM 
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
DimPlot(obj, label = TRUE)

obj <- JoinLayers(obj)

### SingleR automatic annotation 
mouse.se <- celldex::ImmGenData()
results <- SingleR(test = as.SingleCellExperiment(obj), ref = mouse.se, labels = mouse.se$label.main)
plotScoreHeatmap(results)
cell.types <- unique(results$pruned.labels)
Idents(obj) <- "seurat_clusters"
lapply(cell.types, function(x) project_annotation_to_umap(x, results, obj))

### Neutrophil marker genes 
FeaturePlot(obj, features = c("S100a8","S100a9"))

### extract neutrophils 
neutrophils <- subset(obj, idents = c(0,1,2,4,5,6,7,8))

### save object 
saveRDS(neutrophils, "/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_blood_bm.rds")




