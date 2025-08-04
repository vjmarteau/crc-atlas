########## This code integrates neutrophils from all tissues ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
neutrophils1 <- readRDS(file ="/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_colon_tumor.rds")
neutrophils2 <- readRDS(file ="/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_blood_bm.rds")

##### merging
obj <- merge(neutrophils1,neutrophils2,
              add.cell.ids = c("colon_tumor","blood_bm"))
obj <- JoinLayers(obj)

# add tissue and phenotype information 
current.cluster.ids <- c("adjacent_colon","colon","blood_tumor", "blood_no_tumor","BM_tumor", 
                         "BM_no_tumor","disseminated","tumor")
new.cluster.ids <- c("colon","colon","blood", "blood","BM", 
                     "BM","colon","colon")
neut$tissue <- plyr::mapvalues(x = neut$condition, from = current.cluster.ids, to = new.cluster.ids)

current.cluster.ids <- c("adjacent_colon","colon","blood_tumor", "blood_no_tumor","BM_tumor", 
                         "BM_no_tumor","disseminated","tumor")
new.cluster.ids <- c("tumor","healthy","tumor", "healthy","tumor", 
                     "healthy","tumor","tumor")
obj$phenotype <- plyr::mapvalues(x = obj$condition, from = current.cluster.ids, to = new.cluster.ids)

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
obj <- FindNeighbors(obj, reduction = "integrated.mnn", dims = 1:5)
obj <- FindClusters(obj, resolution = 0.5, cluster.name = "mnn.clusters", algorithm = 2)
obj <- RunUMAP(obj, reduction = "integrated.mnn", dims = 1:5, reduction.name = "umap",seed.use = 5)
DimPlot(obj,reduction = "umap",raster=FALSE, label = TRUE) 

p <- DimPlot(obj,reduction = "umap",group.by = "mnn.clusters",raster=FALSE, label= TRUE, label.size = 5, split.by = "phenotype") 
ggsave("/scratch/khandl/CRC_atlas/figures/umap_clustering.svg", width = 8, height = 5, plot = p)

##### DEGs analysis 
obj <- JoinLayers(obj)
obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 10000,margin = 1, assay = "RNA")
Idents(obj) <- "mnn.clusters"
markers <- FindAllMarkers(object = obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, assay = "RNA", slot = "data")
write.csv(markers, "/scratch/khandl/CRC_atlas/data_files/DEGs_neutrophil_clusters.csv")
View(markers %>% group_by(cluster) %>% top_n(n =10, wt = avg_log2FC))

## plot top DEGs
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

# heatmap per cluster and condition 
heatmap_goi_coi(obj, "mnn.clusters",marker_genes,c("c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"), 
                c(10,10,10,7,7,9,10,10,8,6,10),
                c("#6383EA",  "#ED9718","#0BB720",  "#F40B27",
                  "#CC0DC3",  "#84590D","#D974ED","#8AEF29",  "#1AE5E5","#C7DFEA",  "#EFEF0A"),
                c(c0="#6383EA", c1= "#ED9718", c2= "#0BB720", c3= "#F40B27",
                  c4= "#CC0DC3", c5= "#84590D", c6= "#D974ED",c7= "#8AEF29",
                  c8= "#1AE5E5", c9= "#C7DFEA",c10= "#EFEF0A"),F,T)

##### save integrated R object
saveRDS(obj, "/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_integrated.rds")

