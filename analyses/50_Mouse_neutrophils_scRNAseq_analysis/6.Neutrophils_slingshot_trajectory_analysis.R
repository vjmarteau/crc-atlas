########## This code applies slingshot trajectory analysis to neutrophils ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
obj <- readRDS(file ="/data/khandl/CRC_atlas/neutrophils_integrated.rds")

##### integrate and cluster again with only these conditions 
obj <- NormalizeData(obj,normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
obj <- FindVariableFeatures(obj)
obj <- ScaleData(obj,vars.to.regress = c("nFeature_RNA","nCount_RNA","percent.mt"))
obj <- RunPCA(object = obj, features = VariableFeatures(object =obj), npcs = 20, verbose = FALSE)

obj[["RNA"]] <- split(obj[["RNA"]], f = obj$condition)
obj <- IntegrateLayers(object = obj, method = FastMNNIntegration,new.reduction = "integrated.mnn",
                       verbose = FALSE)
obj <- FindNeighbors(obj, reduction = "integrated.mnn", dims = 1:5)
obj <- FindClusters(obj, resolution = 0.5, cluster.name = "mnn.clusters", algorithm = 2)
obj <- RunUMAP(obj, reduction = "integrated.mnn", dims = 1:5, reduction.name = "umap")
DimPlot(obj,reduction = "umap",group.by = "tissue",raster=FALSE)
DimPlot(obj,reduction = "umap",group.by = "phenotype",raster=FALSE)

obj <- JoinLayers(obj)

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

### Density plot 
a <- as.data.frame(colData(sling)$slingPseudotime_1)
b <- as.data.frame(colData(sling)$tissue)
c <- as.data.frame(rownames(colData(sling)))
df <- data.frame(a,b,c)
colnames(df) <- c("Pseudotime","phenotype","cell_id")

#color by tissue 
p <- ggplot(df, aes(x = Pseudotime, fill = phenotype)) +
  geom_density(alpha = .5) +
  theme_minimal() +
  scale_fill_manual(values = c("#EE7B22",  "#426FB6","#33B44A"))
ggsave("/scratch/khandl/CRC_atlas/figures/density_plot.svg", width = 8, height = 5, plot = p)

##### Differential expression using TradeSeq 
#https://www.bioconductor.org/packages/release/bioc/vignettes/tradeSeq/inst/doc/tradeSeq.html
## fit GAM
set.seed(3)
genes = VariableFeatures(obj)
#conditions = factor(sling$age)
BPPARAM <- BiocParallel::bpparam()
BPPARAM$workers <- 2

sling <- fitGAM(counts = sling, nknots = 5, parallel = T, BPPARAM = BPPARAM,genes = genes)

saveRDS(sling, "/data/khandl/CRC_atlas/sling_after_fitGam.rds")
sling <- readRDS("/data/khandl/CRC_atlas/sling_after_fitGam.rds")

### within-lineage comparison 
# this checks if gene expression is associated with a particular lineage 
# associationTest is testing a null hypothesis that all smoother coefficients are equal to each other 
# in other words: if the average gene expression is significantly changing along pseudotime 
assoRes <- associationTest(sling)
head(assoRes)
assoRes2 <- na.omit(assoRes)
assoRes2$padj <- p.adjust(assoRes2$pvalue, "fdr")

# extract significant genes 
assoRes2 <- assoRes2[assoRes2$padj <= 0.05,]
assoRes2$gene <- rownames(assoRes2)

# sort decreasing by wald stat value 
assoRes2 <- assoRes2[order(assoRes2$meanLogFC, decreasing = TRUE),]

# based on mean smoother
yhatSmooth <- predictSmooth(sling, gene = assoRes2$gene, nPoints = 50, tidy = FALSE) %>%log1p()
yhatSmoothScaled <- t(apply(yhatSmooth,1, scales::rescale))
# sort rows the same order as the data frame that is ordered by waldStat
yhatSmoothScaled_df <- as.data.frame(yhatSmoothScaled)
yhatSmoothScaled <- within(yhatSmoothScaled_df, rownames(yhatSmoothScaled_df) <- factor(rownames(yhatSmoothScaled_df), 
                                                                                        levels = assoRes2$gene))
yhatSmoothScaled_df2 <- yhatSmoothScaled_df[order(match(rownames(yhatSmoothScaled_df),assoRes2$gene)),]
yhatSmoothScaled_df2$waldStat <- assoRes2$waldStat
yhatSmoothScaled_df2$padj <- assoRes2$padj
yhatSmoothScaled_df2$meanLogFC <- assoRes2$meanLogFC

write.csv(yhatSmoothScaled_df2,"/scratch/khandl/CRC_atlas/tradeSeq/DEGs_along_pseudotime.csv" )

### compare specific pseudotime values within a lineage 
# This appplies a Wald test to assess the null hypothesis that the average expression at one point of the smoother is equal to the average
# expression at the other point of the smoother 
# pseudotime 0 with 10 
#pseutotime_0_10  <- startVsEndTest(sling, pseudotimeValues = c(0,10))
#pseutotime_0_10 <- na.omit(pseutotime_0_10)
#pseutotime_0_10$padj <- p.adjust(pseutotime_0_10$pvalue, "fdr")
#pseutotime_0_10 <- pseutotime_0_10[pseutotime_0_10$padj <= 0.05,]

##### extract genes with high Waldstat coeffienct 
## DEGs 
marker_genes <- c("Cxcl3","Cd14","Acod1","Thbs1","G0s2" ,"Ccl4","Cstdc4","Dusp16","Gm6977","Fth1", #cl0
                  "Ptgs2","Ccrl2","Tgm2","Nr4a1","Tgif1","Egr1","Ifrd1","Cd274","Nr4a3","Gadd45b", #cl1
                  "Fgl2","Tagln2","Myadm","Steap4","Gpcpd1","Plxdc2","Prr5l","Slc40a1","Cd244a","Scnn1a", #cl2
                  "Gm2a","Trim30d","Txnip","Sirpb1c","Itga4","Lfng","Tsc22d3", #cl3
                  "Wfdc17","Jaml","St8sia4","Chsy1","Dck","Mov10","Tmem43", #cl4
                  "Mmp8","Timp2","Eid1","Retnlg","Asb7","S100a6","Prok2","Mfsd14a","Rfc2", #cl5
                  "Camp","Tmem216","Ltf","Zmpste24","Ngp","Mlst8","Orm1","Manea","Ear2","Smim5", #cl6
                  "Il1a","Ccl3","Tnf","Icam1","Il23a","Dusp2","Csf1","Bcl2a1d","Bcl2a1a","Ikbke", #cl7
                  "Ly6c2","Tinagl1","Ceacam10","Gpr27","Hopx","Cd55","Eif4e3","Tmsb10", #cl8
                  "Itgb2l","Ly6a2","Acvrl1","1700047M11Rik","Anxa5","F730016J06Rik", #cl9
                  "Pclaf","Cdca8","Ube2c","Spc24","Pbk","Hmgb3","Paics","Ranbp1","Cln6","Pimreg" #cl10     
)

assoRes3 <- assoRes2[assoRes2$gene %in% marker_genes,]

# based on mean smoother
yhatSmooth <- 
  predictSmooth(sling, gene = assoRes3$gene, nPoints = 50, tidy = FALSE) %>%
  log1p()
yhatSmoothScaled <- t(apply(yhatSmooth,1, scales::rescale))

# sort rows the same order as the data frame that is ordered by waldStat
yhatSmoothScaled_df <- as.data.frame(yhatSmoothScaled)
yhatSmoothScaled <- within(yhatSmoothScaled_df, rownames(yhatSmoothScaled_df) <- factor(rownames(yhatSmoothScaled_df), 
                                                                                        levels = assoRes3$gene))
yhatSmoothScaled_df2 <- yhatSmoothScaled_df[order(match(rownames(yhatSmoothScaled_df),assoRes3$gene)),]

heatSmooth <- pheatmap(yhatSmoothScaled_df2[,],
                       cluster_cols = FALSE,cluster_rows = TRUE,
                       show_rownames = TRUE, show_colnames = FALSE, legend = TRUE,
                       silent = TRUE
)
heatSmooth

### Visiualise genes of interest 

# plot GOI 
goi <- c("Ngp","Ear2","Retnlg","S100a6","Cd14","Fth1","Tgm2","Ccl4","Cdca8","Cln6")
for (i in goi) {
  p <- plotSmoothers(sling, assays(sling)$counts,
                     gene = i,
                     alpha = 1, border = TRUE) +
    ggtitle(i)
  ggsave(paste0("/scratch/khandl/CRC_atlas/tradeSeq/",i,".svg"), width = 10, height = 6, plot = p)
}


