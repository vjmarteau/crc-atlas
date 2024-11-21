##########  Pre-processing of scRNAseq data ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### tumor and colon samples 
### Seurat object generation after removal of cell free RNA (decontX)
tumor <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data", "M25", "tumor_Expression_Data.st"), 
  project = "tumor", condition = "tumor",3,200)

disseminated <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data","M25", "Disseminated_Expression_Data.st"), 
  project = "disseminated", condition = "disseminated",3,200)

adjacent_colon <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data","M25", "Adjacent_Colon_Expression_Data.st"), 
  project = "adjacent_colon", condition = "adjacent_colon",3,200)

colon <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data","M25", "Colon_Expression_Data.st"), 
  project = "colon", condition = "colon",3,200)

### Merge samples
obj <- merge(tumor, y = c(disseminated,adjacent_colon, colon),
               add.cell.ids = c("tumor","disseminated","adjacent_colon", "colon"))
obj <- JoinLayers(obj)

### Add mitochondrial percentage per cell 
obj$percent.mt <- PercentageFeatureSet(obj, pattern = "^mt.")

### apply mitochondrial and nFeature cutoffs
obj <- subset(obj, subset = nFeature_RNA < 5000 & percent.mt < 25)

### save object
saveRDS(obj, file = "/scratch/khandl/CRC_atlas/seurat_objects/pre_processing_tumor_colon.rds")

##### blood and bone marrow samples 
### Seurat object generation after removal of cell free RNA (decontX )
BM_tumor <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data","M25", "BM_tumor_Expression_Data.st"), 
  project = "BM_tumor", condition = "BM_tumor",3,200)

BM_no_tumor <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data","M25", "BM_no_tumor_Expression_Data.st"), 
  project = "BM_no_tumor", condition = "BM_no_tumor",3,200)

blood_tumor <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data","M25", "Blood_tumor_Expression_Data.st"), 
  project = "blood_tumor", condition = "blood_tumor",3,200)

blood_no_tumor <- create_seurat_plus_DecontX(
  path_to_st_file = file.path("/data/khandl/raw_data","M25", "Blood_no_tumor_Expression_Data.st"), 
  project = "blood_no_tumor", condition = "blood_no_tumor",3,200)

### Merge samples
obj <- merge(BM_tumor, y = c(BM_no_tumor,blood_tumor, blood_no_tumor),
             add.cell.ids = c("BM_tumor","BM_no_tumor","blood_tumor", "blood_no_tumor"))
obj <- JoinLayers(obj)

### Add mitochondrial percentage per cell 
obj$percent.mt <- PercentageFeatureSet(obj, pattern = "^mt.")

### apply mitochondrial and nFeature cutoffs
obj <- subset(obj, subset = nFeature_RNA < 5000 & percent.mt < 25)

### save object
saveRDS(obj, file = "/scratch/khandl/CRC_atlas/seurat_objects/pre_processing_blood_bm.rds")
