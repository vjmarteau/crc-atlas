########## This code applies DEG analysis of neutrophils from bone marrow and blood - healthy vs. tumor ##########

##### link to libraries and functions
source("~/Projects/CRC_atlas_mouse_neutrophils/1.Packages_and_functions.R")

##### load R objects 
obj <- readRDS(file ="/scratch/khandl/CRC_atlas/seurat_objects/neutrophils_progenitors_annotated.rds")

##### GMP
Idents(obj) <- "annotation"
sub <- subset(obj, idents = "GMP")
table(sub$phenotype) #113 healthy, 120 tumor 

sub <- NormalizeData(sub, normalization.method = "LogNormalize",
                     scale.factor = 10000,
                     margin = 1, assay = "RNA")

DefaultAssay(sub) <- "RNA"
Idents(sub) <- "phenotype"
degs <- FindMarkers(object = sub, ident.1 = "tumor", ident.2 = "healthy", only.pos = FALSE, min.pct = 0.25, 
                    logfc.threshold = 0.2,slot = "data")
write.csv(degs, "/scratch/khandl/CRC_atlas/figures/GMP_tumor_vs_healthy.csv")

# plot in volcano 
EnhancedVolcano(degs,
                lab = paste0("italic('", rownames(degs), "')"),
                x = 'avg_log2FC',
                y = 'p_val_adj',
                title = 'GMP tumor vs. healthy',
                pCutoff = 0.05,
                FCcutoff = 0.25,
                pointSize = 3.0,
                labSize = 6,
                col=c('black', 'black', 'black', 'red3'),
                colAlpha = 1,
                parseLabels = TRUE)

##### ProNeutro
Idents(obj) <- "annotation"
sub <- subset(obj, idents = "ProNeutro")
table(sub$phenotype) #112 healthy, 392 tumor 

sub <- NormalizeData(sub, normalization.method = "LogNormalize",
                     scale.factor = 10000,
                     margin = 1, assay = "RNA")

DefaultAssay(sub) <- "RNA"
Idents(sub) <- "phenotype"
degs <- FindMarkers(object = sub, ident.1 = "tumor", ident.2 = "healthy", only.pos = FALSE, min.pct = 0.25, 
                    logfc.threshold = 0.2,slot = "data")
write.csv(degs, "/scratch/khandl/CRC_atlas/figures/ProNeutro_tumor_vs_healthy.csv")

# plot in volcano 
EnhancedVolcano(degs,
                lab = paste0("italic('", rownames(degs), "')"),
                x = 'avg_log2FC',
                y = 'p_val_adj',
                title = 'ProNeutro tumor vs. healthy',
                pCutoff = 0.05,
                FCcutoff = 0.25,
                pointSize = 3.0,
                labSize = 6,
                col=c('black', 'black', 'black', 'red3'),
                colAlpha = 1,
                parseLabels = TRUE)

##### GSEA analysis 
### load GO terms 
#Biological process 
m_df      <- msigdbr(species = "Mus musculus", category = "C5", subcategory = "BP")
BP        <- m_df %>% split(x = .$gene_symbol, f = .$gs_name)
#Hallmarks
m_df      <- msigdbr(species = "Mus musculus", category = "H")
Hallmarks       <- m_df %>% split(x = .$gene_symbol, f = .$gs_name)
#Reactome
m_df      <- msigdbr(species = "Mus musculus", category = "C2", subcategory = "REACTOME")
Reactome       <- m_df %>% split(x = .$gene_symbol, f = .$gs_name)
#KEGG
m_df      <- msigdbr(species = "Mus musculus", category = "C2", subcategory = "KEGG")
KEGG       <- m_df %>% split(x = .$gene_symbol, f = .$gs_name)

### run GSEA 
## ProNeutro
degs <- read.csv(file = "/scratch/khandl/CRC_atlas/figures/ProNeutro_tumor_vs_healthy.csv")
degs <- na.omit(degs)
degs <- degs[degs$p_val_adj <= 0.05,]
degs$p_val_adj[degs$p_val_adj == 0] <- .Machine$double.xmin #replace 0 with lowest number if p-adjusted is 0
print(sort(degs$p_val_adj))
ranks <- degs %>% na.omit()%>% mutate(ranking=-log10(p_val_adj)/sign(avg_log2FC))
print(sort(ranks$ranking)) 
ranks <- ranks$ranking
names(ranks) <- degs$X
head(ranks, 10)
#remove infinit numbers 
ranks <- ranks[!is.infinite(ranks)]

fGSEA_to_csv(ranks, "/scratch/khandl/CRC_atlas/figures/ProNeutro_GSEA.csv")
df <- read.csv(  "/scratch/khandl/CRC_atlas/figures/ProNeutro_GSEA.csv")

pathways_oi <- c("HALLMARK_GLYCOLYSIS","HALLMARK_FATTY_ACID_METABOLISM","GOBP_TISSUE_REMODELING","HALLMARK_HYPOXIA",
                 "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION","HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
                 "GOBP_VASCULATURE_DEVELOPMENT","HALLMARK_INTERFERON_GAMMA_RESPONSE","GOBP_ANTIGEN_RECEPTOR_MEDIATED_SIGNALING_PATHWAY",
                 "GOBP_IMMUNE_RESPONSE","HALLMARK_INTERFERON_ALPHA_RESPONSE")

df <- df[df$pathway %in% pathways_oi,]
rownames(df) <- df$pathway

df <- df[pathways_oi,]

p <- ggplot(df, aes(x = NES, y = pathway,color = -log10(padj))) + 
  geom_point(size = 10)  + theme_light()+ scale_color_gradientn(colors = c("blue", "yellow", "red")) 
ggsave("/scratch/khandl/eos_tumor/mono/Spp1_TAMs_against_the_rest.svg", width = 12, height = 6, plot = p)

### save object 
saveRDS(obj, "/data/khandl/Eosinophils_in_CRC/seurat_objects/tumor_phil_wt_macs_annotated.rds")
obj <- readRDS("/data/khandl/Eosinophils_in_CRC/seurat_objects/tumor_phil_wt_macs_annotated.rds")

