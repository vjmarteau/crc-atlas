########## This code applies slingshot trajectory analysis to neutrophils ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
obj <- readRDS(file ="/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_integrated.rds")

### convert Seurat object to SCE 
Idents(obj) <- "mnn.clusters"
sce <- sceasy::convertFormat(obj, from="seurat", to="sce")

### trajectory inference on mnn.clusters 
sce <- slingshot(sce, clusterLabels = "mnn.clusters", reducedDim = "UMAP")

# visualization and embedd in UMAP 
umap_curve_embedding <- embedCurves(sce, newDimRed=reducedDims(sce)$UMAP)
SlingshotDataSet(umap_curve_embedding)@curves[[1]]$ord
colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)

plotcol <- colors[cut(sce$slingPseudotime_1, breaks=100)]
plot(reducedDims(sce)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(umap_curve_embedding), lwd = 2, col = 'black')

# legend 
lgd <- matrix(colors, nrow=1)
rasterImage(lgd, -5,4,-2,4.3)
text(-3.5, 4.1, pos = 3, cex = .7, labels = 'Pseudotime')

### Density plot 
a <- as.data.frame(colData(sce)$slingPseudotime_1)
b <- as.data.frame(colData(sce)$tissue)
c <- as.data.frame(rownames(colData(sce)))
df <- data.frame(a,b,c)
colnames(df) <- c("Pseudotime","phenotype","cell_id")

#color by tissue 
p <- ggplot(df, aes(x = Pseudotime, fill = phenotype)) +
  geom_density(alpha = .5) +
  theme_minimal() +
scale_fill_manual(values = c("#EE7B22",  "#426FB6","#33B44A"))
ggsave("/scratch/khandl/CRC_atlas/figures/sling_shot_density.svg", width = 8, height = 5, plot = p)
