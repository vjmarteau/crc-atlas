# Packages 
library(Seurat)
library(dplyr)
library(tidyr)
library(decontX)
library(Matrix)
library(SeuratWrappers)
library(SingleR)
library(celldex)
library(ggplot2)
library(pheatmap)
library(orthogene)
library(stringr)
library(orthogene)
library(slingshot)
library(RColorBrewer)

##### Functions for pre-processing 
### Function to read in counts matrix data generated from BD Seven Bridges platform (by Simona Baghai)
data_to_sparse_matrix <- function(data.st_file_path) {
  # read in file with cell index - gene name - values
  # import for one cartridge, one sample
  input <-read.table(data.st_file_path, header = T)
  # transform to matrix (data.frame actually)
  # we take as default values from the column "RSEC_Adjusted_Molecules" (= error corrected UMIs)
  mat <- input %>% pivot_wider(id_cols = Bioproduct, 
                               values_from = RSEC_Adjusted_Molecules, 
                               names_from = Cell_Index, values_fill = 0)  %>% 
    tibble::column_to_rownames("Bioproduct")
  # convert to sparse matrix (~ dgMatrix)
  sparse_mat = Matrix(as.matrix(mat),sparse=TRUE)
  return(sparse_mat)
}

### Function to generate Seurat objects for each sample + application of DecontX (accounting for cell free RNA contamination) 
create_seurat_plus_DecontX <- function(
    path_to_st_file,
    project,
    condition, 
    min.cells,
    min.features
) {
  input_matrix <- data_to_sparse_matrix(path_to_st_file)
  condition_sample <-CreateSeuratObject(input_matrix, 
                                        project = project,
                                        min.cells = min.cells,
                                        min.features = min.features)
  sce <- as.SingleCellExperiment(condition_sample)
  # run decontX with default settings 
  sce.delta <- decontX(sce)
  # convert back to a Seurat object 
  seuratObject <- CreateSeuratObject(round(decontXcounts(sce.delta)))
  seuratObject$condition <- condition
  
  return(seuratObject)
}

##### Functions for cell type annotation 
### Function to project the cell type from SingleR result to the umap space to identify which cluster represents which cell type 
# after FastMNN data integration 
project_annotation_to_umap_fastMNN <- function(cell.type, singleResult, seurat_object) {
  temp <- as.data.frame(singleResult[4])
  temp$cell <- rownames(temp)
  temp <- temp%>%filter(temp$pruned.labels %in% cell.type)
  temp <- temp$cell
  print(DimPlot(seurat_object, reduction = "umap.mnn", label = TRUE, label.size = 10, pt.size = 2, cells.highlight = temp, sizes.highlight = 2,raster=FALSE) + NoLegend() + ggtitle(cell.type))
}

# without FastMNN data integration 
project_annotation_to_umap <- function(cell.type, singleResult, seurat_object) {
  temp <- as.data.frame(singleResult[4])
  temp$cell <- rownames(temp)
  temp <- temp%>%filter(temp$pruned.labels %in% cell.type)
  temp <- temp$cell
  print(DimPlot(seurat_object, reduction = "umap", label = TRUE, label.size = 10, pt.size = 0.5, cells.highlight = temp, sizes.highlight = 2) + NoLegend() + ggtitle(cell.type))
}

##### Functions for data visualization 
### Function to plot genes of interest in a scaled heatmap per condition of interest with colors of interest 
heatmap_goi_coi <- function(
    seurat_object,
    condition_oi,
    markers_oi,
    groups_of_markers,
    number_of_markers_per_group,
    colors_per_group,
    groups_and_colors,
    cluster_rows_cond,
    cluster_cols_cond
){
  #average expression per cluster and condition 
  average_expression <- AverageExpression(seurat_object, return.seurat = FALSE, features = markers_oi, normalization.method = "LogNormalize",assays = "RNA", group.by = condition_oi)
  average_expression_df <- as.data.frame(average_expression)
  average_expression_df <- average_expression_df[match(markers_oi, rownames(average_expression_df)),]
  
  #prepare palette for pheatmap
  paletteLength   <- 50
  myColor         <- colorRampPalette(c("blue", "white", "darkorange"))(paletteLength)
  breaksList      = seq(-2, 2, by = 0.04)
  
  #prepare annotation for pheatmap
  annotation_rows             <- data.frame(markers = rep(groups_of_markers, number_of_markers_per_group))
  rownames(annotation_rows)   <- rownames(average_expression_df)
  annotation_rows$markers     <- factor(annotation_rows$markers, levels = groups_of_markers)
  
  mycolors <- colors_per_group
  names(mycolors) <- unique(annotation_rows$markers)
  mycolors <- list(category = mycolors)
  annot_colors=list(markers=groups_and_colors)
  
  p <- pheatmap(average_expression_df,scale = "row",
                color = colorRampPalette(c("blue", "white", "darkorange"))(length(breaksList)), # Defines the vector of colors for the legend (it has to be of the same lenght of breaksList)
                breaks = breaksList,
                cluster_rows = cluster_rows_cond, cluster_cols = cluster_cols_cond, 
                border_color = "black", 
                legend_breaks = -2:2, 
                cellwidth = 10, cellheight = 5,
                angle_col = "45", 
                annotation_colors = annot_colors,
                annotation_row = annotation_rows,
                fontsize = 5)
  print(p)
}
