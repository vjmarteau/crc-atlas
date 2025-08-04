########## This code applies slingshot trajectory analysis to neutrophils ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
obj <- readRDS(file ="/data/khandl/CRC_atlas/neutrophils_progenitors_annotated.rds")

Idents(obj) <- "annotation"
obj <- subset(obj, idents = c(0:7))
DimPlot(obj)

##### Infer slingshot trajectory 
# convert Seurat object to a SCE object 
sce <- sceasy::convertFormat(obj, from="seurat", to="sce")

### trajectory inference 
sling <- slingshot(sce, clusterLabels = colData(sce)$tissue, reducedDim = "UMAP"
                   ,omega = TRUE, approx_points = 100, start.clus = "BM")

# visualization and embedd in UMAP 
umap_curve_embedding <- embedCurves(sling, newDimRed=reducedDims(sling)$UMAP)
SlingshotDataSet(umap_curve_embedding)@curves[[1]]$ord
colors <- colorRampPalette(brewer.pal(11,'Spectral')[-6])(100)

plotcol <- colors[cut(sling$slingPseudotime_1, breaks=100)]
plot(reducedDims(sling)$UMAP, col = plotcol, pch=16, asp = 1)
lines(SlingshotDataSet(umap_curve_embedding), lwd = 2, col = 'black')

# legend 
lgd <- matrix(colors, nrow=1)
rasterImage(lgd, -5,4,-2,4.3)
text(-3.5, 4.1, pos = 3, cex = .7, labels = 'Pseudotime')

### Differential progression between tumor and healthy 
### Density plot 
a <- as.data.frame(colData(sling)$slingPseudotime_1)
b <- as.data.frame(colData(sling)$phenotype)
c <- as.data.frame(rownames(colData(sling)))
df <- data.frame(a,b,c)
colnames(df) <- c("Pseudotime","phenotype","cell_id")

#color by tissue 
p <- ggplot(df, aes(x = Pseudotime, fill = phenotype)) +
  geom_density(alpha = .5) +
  theme_minimal() +
  scale_fill_manual(values = c( "#426FB6","#EE7B22"))
ggsave("/scratch/khandl/CRC_atlas/figures/density_plot_phenotype_comp.svg", width = 8, height = 5, plot = p)

### Differential expression using TradeSeq 
## fit GAM
set.seed(3)
genes = VariableFeatures(obj)
conditions = factor(sling$phenotype)
BPPARAM <- BiocParallel::bpparam()
BPPARAM$workers <- 12

sling <- fitGAM(counts = sling, nknots = 5, 
                conditions =conditions, parallel = T, BPPARAM = BPPARAM,genes = genes)

saveRDS(sling, "/data/khandl/CRC_atlas/neutrophils_integrated_fitGAM_between_cond.rds")
sling <- readRDS("/data/khandl/CRC_atlas/neutrophils_integrated_fitGAM_between_cond.rds")

## differential expresssion between conditions
condRes <- conditionTest(sling, l2fc = log2(2))
condRes$padj <- p.adjust(condRes$pvalue, "fdr")
mean(condRes$padj <= 0.05, na.rm = TRUE)
sum(condRes$padj <= 0.05, na.rm = TRUE)

# extract genes with significant p values 
condRes_only_sig <- condRes[!is.na(condRes$waldStat),]
condRes_only_sig <- condRes_only_sig[condRes_only_sig$padj <= 0.05,]
condRes_only_sig$gene <- rownames(condRes_only_sig)
condRes_only_sig <- condRes_only_sig[order(condRes_only_sig$waldStat, decreasing = TRUE),]

# based on mean smoother
yhatSmooth <- predictSmooth(sling, gene = condRes_only_sig$gene, nPoints = 50, tidy = FALSE) %>%log1p()
yhatSmoothScaled <- t(apply(yhatSmooth,1, scales::rescale))
# sort rows the same order as the data frame that is ordered by waldStat
yhatSmoothScaled_df <- as.data.frame(yhatSmoothScaled)
yhatSmoothScaled <- within(yhatSmoothScaled_df, rownames(yhatSmoothScaled_df) <- factor(rownames(yhatSmoothScaled_df), 
                                                                                        levels = condRes_only_sig$gene))
yhatSmoothScaled_df2 <- yhatSmoothScaled_df[order(match(rownames(yhatSmoothScaled_df),condRes_only_sig$gene)),]
yhatSmoothScaled_df2$waldStat <- condRes_only_sig$waldStat
yhatSmoothScaled_df2$padj <- condRes_only_sig$padj

write.csv(yhatSmoothScaled_df2,"/scratch/khandl/CRC_atlas/DEGs_along_pseudotime_healthy_vs_tumor.csv" )

### Visiualise genes of interest along pseudotime 
## plot all genes in hetmap 
genes_of_interest <- c("Rpl27","Syne1","Ly6g", "Ffar2","Eif5a","Cd63","Rps17", "Rpl22","Rps19","Rpl32","Tnf","Il23a","H2-Q7","CCrl2","Il1a","Ptgs2","Ccl4","Ccl3","Nr4a1","Tgm2",
                       "Cxcl2","Il1rn","Nlrp3","Thbs1","G0s2","Serpine1","Serpine2","Osm","Tnfaip6","Cdk6","Wfdc17","Elane",
                       "Top2a","Hspa5","Lcn2","anxa1","Plac8","Slfn5","Hmgb1","Ncam1","Syngr2","Nedd4","Zbtb16","Stfa2","Cstdc5","Retnla","Tnfaip6","Nos2",
                       "Ifit3","Anln","Retnlg","Ifi27l2a","Olfm4")

yhatSmoothScaled_df3 <- yhatSmoothScaled_df2[rownames(yhatSmoothScaled_df2) %in% genes_of_interest,]
yhatSmoothScaled_df3$waldStat <- NULL
yhatSmoothScaled_df3$padj <- NULL

  heatSmooth <- pheatmap(yhatSmoothScaled_df3,
                       cluster_cols = FALSE,cluster_rows = TRUE,
                       show_rownames = TRUE, show_colnames = FALSE, main = "healthy/tumor", legend = TRUE,
                       silent = TRUE
)
heatSmooth
