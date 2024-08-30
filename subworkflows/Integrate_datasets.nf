#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { REMOVE_AMBIENT_RNA } from "${baseDir}/modules/local/scAR.nf"
include { CELL_CYCLE_PHASE } from "${baseDir}/modules/local/Score_cell_cycle.nf"
include { JUPYTERNOTEBOOK as Concat_datasets;
          JUPYTERNOTEBOOK as Filter_mito;
          JUPYTERNOTEBOOK as Get_Seeds;
          JUPYTERNOTEBOOK as HVG_TUMOR;
          JUPYTERNOTEBOOK as HVG_NORMAL;
          JUPYTERNOTEBOOK as HVG_BLOOD;
          JUPYTERNOTEBOOK as HVG_METASTASIS;
          JUPYTERNOTEBOOK as HVG_LYMPH_NODE;
          JUPYTERNOTEBOOK as HVG_BD } from "${baseDir}/modules/local/jupyternotebook/main"
include { SCVI as SCVI_SEED } from "${baseDir}/modules/local/scVI.nf"
include { NEIGHBORS_LEIDEN_UMAP as NEIGHBORS_LEIDEN_UMAP_SEED } from "${baseDir}/modules/local/neighbors_leiden_paga_umap.nf"
include { JUPYTERNOTEBOOK as SEED_TUMOR;
          JUPYTERNOTEBOOK as SEED_NORMAL;
          JUPYTERNOTEBOOK as SEED_BLOOD;
          JUPYTERNOTEBOOK as SEED_METASTASIS;
          JUPYTERNOTEBOOK as SEED_LYMPH_NODE;
          JUPYTERNOTEBOOK as SEED_BD_RHAPSODY;
          JUPYTERNOTEBOOK as Merge_Seeds;
          JUPYTERNOTEBOOK as Merge_SOLO } from "${baseDir}/modules/local/jupyternotebook/main"
include { SCVI } from "${baseDir}/modules/local/scVI.nf"
include { SOLO_DOUBLETS } from "${baseDir}/modules/local/solo.nf"
include { NEIGHBORS_LEIDEN_UMAP } from "${baseDir}/modules/local/neighbors_leiden_paga_umap.nf"
include { SCANVI } from "${baseDir}/modules/local/scVI.nf"
include { NEIGHBORS_LEIDEN_UMAP as NEIGHBORS_LEIDEN_UMAP_SCANVI } from "${baseDir}/modules/local/neighbors_leiden_paga_umap.nf"

/*
 * Integrate available datasets into core-atlas
 *   - Denoise counts if possible (scAR)
 *   - Compute highly variable genes (hvg) per sample type
 *   - Seed annotaions for SCANVI
 *   - Compute doublet scores (solo)
 */

out_dir = file(params.outdir)
mode = params.publish_dir_mode

workflow Integrate_datasets {

    take:
        datasets
        adata_gtf

    main:

    REMOVE_AMBIENT_RNA(datasets)
                
    ch_concat_datasets = adata_gtf

    Concat_datasets(
        Channel.value([
            [id: "Concat_datasets"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/10-Concat_datasets.py", checkIfExists: true)
            ]),
        ch_concat_datasets.map{ adata_var_gtf ->
            [
                "adata_var_gtf": adata_var_gtf.name,
                "datasets_path": "."
            ]},
        ch_concat_datasets.mix(REMOVE_AMBIENT_RNA.out.scAR
            .collect { _id, path -> path }
            .flatten())
            .collect()
        )

    CELL_CYCLE_PHASE(
        Concat_datasets.out.artifacts
            .flatten()
            .filter { it -> it.name.contains(".h5ad") }
            .map { it ->
            [it.baseName.split("\\-")[0], it]
        }
    )
   
    ch_merged_adata = Concat_datasets.out.artifacts
        .flatten()
        .filter { it -> it.name.contains(".h5ad") }

    Filter_mito(
        Channel.value([
            [id: "Filter_mito"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/20-Filter_mito.py", checkIfExists: true)
        ]),
        ch_merged_adata.map { adata_path ->
            [
                "adata_path": adata_path.name,
                "phase_dir": "."
            ]
        },
        ch_merged_adata.mix(
            CELL_CYCLE_PHASE.out.cell_cyle_phase
        ).collect()
    )

    ch_filtered_adata = Filter_mito.out.artifacts
        .flatten()
        .filter { it -> it.name.contains("mito_filtered") }
   
    Get_Seeds(
        Channel.value([
            [id: "Get_seeds"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/30-Get_seeds.py", checkIfExists: true)
        ]),
        ch_filtered_adata.map { adata_path -> ["adata_path": adata_path.name] },
        ch_filtered_adata
    )

    ch_get_seeds = Get_Seeds.out.artifacts
        .flatten()
        .filter{ it -> it.name.contains(".h5ad") }

    
    ch_tumor_hvg = ch_get_seeds.filter { it.name.contains("tumor") }

    HVG_TUMOR(
        Channel.value([
            [id: "Tumor_hvg"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/31-Compute_tumor_hvg.py", checkIfExists: true)
        ]),
        ch_tumor_hvg.map { adata_path -> ["adata_path": adata_path.name] },
        ch_tumor_hvg
    )

    ch_normal_hvg = ch_get_seeds.filter { it.name.contains("normal") }

    HVG_NORMAL(
        Channel.value([
            [id: "Normal_hvg"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/32-Compute_normal_hvg.py", checkIfExists: true)
        ]),
        ch_normal_hvg.map { adata_path -> ["adata_path": adata_path.name] },
        ch_normal_hvg
    )

    ch_blood_hvg = ch_get_seeds.filter { it.name.contains("blood") }
 
    HVG_BLOOD(
        Channel.value([
            [id: "Blood_hvg"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/33-Compute_blood_hvg.py", checkIfExists: true)
        ]),
        ch_blood_hvg.map { adata_path -> ["adata_path": adata_path.name] },
        ch_blood_hvg
    )

    ch_metastasis_hvg = ch_get_seeds.filter { it.name.contains("metastasis") }

    HVG_METASTASIS(
        Channel.value([
            [id: "Metastasis_hvg"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/34-Compute_metastasis_hvg.py", checkIfExists: true)
        ]),
        ch_metastasis_hvg.map { adata_path -> ["adata_path": adata_path.name] },
        ch_metastasis_hvg
    )

    ch_lymph_node_hvg = ch_get_seeds.filter { it.name.contains("lymph_node") }

    HVG_LYMPH_NODE(
        Channel.value([
            [id: "Lymph_node_hvg"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/35-Compute_lymph_node_hvg.py", checkIfExists: true)
        ]),
        ch_lymph_node_hvg.map { adata_path -> ["adata_path": adata_path.name] },
        ch_lymph_node_hvg
    )

    ch_bd_hvg = Filter_mito.out.artifacts
        .flatten()
        .filter { it -> it.name.contains("bd") }

    HVG_BD(
        Channel.value([
            [id: "BD_Rhapsody_hvg"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/36-Compute_bd_rhapsody_hvg.py", checkIfExists: true)
        ]),
        ch_bd_hvg.map { adata_path -> ["adata_path": adata_path.name] },
        ch_bd_hvg
    )

    ch_scvi_seeds = HVG_TUMOR.out.artifacts.concat(
        HVG_NORMAL.out.artifacts,
        HVG_BLOOD.out.artifacts,
        HVG_METASTASIS.out.artifacts,
        HVG_LYMPH_NODE.out.artifacts,
        HVG_BD.out.artifacts
        )
        .flatten()
        .filter{ it -> it.name.contains(".h5ad") }
        .map { it ->
            [it.baseName.split("\\-")[0], it]
        }

    SCVI_SEED(
        ch_scvi_seeds,
        ["batch", "gene_dispersion_label", "gene-label"]
     // ["batch", null, "gene-batch"]
        )

    NEIGHBORS_LEIDEN_UMAP_SEED(
        SCVI_SEED.out.adata
            .collect { _id, path -> path }
            .flatten(),
        "X_scVI",
        2.0
        )

    ch_bd_seed = NEIGHBORS_LEIDEN_UMAP_SEED.out.adata
        .filter { id, _ -> id.contains("bd") }
        .collect { _id, path -> path }
        .flatten()
        .concat(
            Channel.fromPath("${baseDir}/tables/bd-is_droplet.csv")
        )
        .collect()

    SEED_BD_RHAPSODY(
        Channel.value([
            [id: "Seed_bd"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/45-Annotate_bd_rhapsody_seeds.py", checkIfExists: true)
        ]),
        ch_bd_seed.map { adata_path, bd_is_droplet ->
            [
                "adata_path": adata_path.name,
                "bd_is_droplet_path": bd_is_droplet.name
            ]
        },
        ch_bd_seed
    )

    ch_tumor_seed = NEIGHBORS_LEIDEN_UMAP_SEED.out.adata
        .filter { id, _ -> id.contains("tumor") }
        .collect { _id, path -> path }
        .flatten()
        .concat(
            SEED_BD_RHAPSODY.out.artifacts
                .flatten()
                .filter { it -> it.name.contains("is_droplet") },
            Channel.fromPath("${baseDir}/tables/colon_atas_cell_type_marker_genes.csv")
        )
        .collect()

    SEED_TUMOR(
        Channel.value([
            [id: "Seed_tumor"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/40-Annotate_tumor_seeds.py", checkIfExists: true)
        ]),
        ch_tumor_seed.map { adata_path, bd_is_droplet, markers ->
            [
                "adata_path": adata_path.name,
                "bd_is_droplet": bd_is_droplet.name,
                "marker_genes_path": markers.name
            ]
        },
        ch_tumor_seed
    )

    ch_normal_seed = NEIGHBORS_LEIDEN_UMAP_SEED.out.adata
        .filter { id, _ -> id.contains("normal") }
        .collect { _id, path -> path }
        .flatten()
        .concat(
            SEED_BD_RHAPSODY.out.artifacts
                .flatten()
                .filter { it -> it.name.contains("is_droplet") },
            Channel.fromPath("${baseDir}/tables/colon_atas_cell_type_marker_genes.csv")
        )
        .collect()

    SEED_NORMAL(
        Channel.value([
            [id: "Seed_normal"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/41-Annotate_normal_seeds.py", checkIfExists: true)
        ]),
        ch_normal_seed.map { adata_path, bd_is_droplet, markers ->
            [
                "adata_path": adata_path.name,
                "bd_is_droplet": bd_is_droplet.name,
                "marker_genes_path": markers.name
            ]
        },
        ch_normal_seed
    )

    ch_blood_seed = NEIGHBORS_LEIDEN_UMAP_SEED.out.adata
        .filter { id, _ -> id.contains("blood") }
        .collect { _id, path -> path }
        .flatten()
        .concat(
            SEED_BD_RHAPSODY.out.artifacts
                .flatten()
                .filter { it -> it.name.contains("is_droplet") },
            Channel.fromPath("${baseDir}/tables/colon_atas_cell_type_marker_genes.csv")
        )
        .collect()

    SEED_BLOOD(
        Channel.value([
            [id: "Seed_blood"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/42-Annotate_blood_seeds.py", checkIfExists: true)
        ]),
        ch_blood_seed.map { adata_path, bd_is_droplet, markers ->
            [
                "adata_path": adata_path.name,
                "bd_is_droplet": bd_is_droplet.name,
                "marker_genes_path": markers.name
            ]
        },
        ch_blood_seed
    )
 
    ch_metastasis_seed = NEIGHBORS_LEIDEN_UMAP_SEED.out.adata
        .filter { id, _ -> id.contains("metastasis") }
        .collect { _id, path -> path }
        .flatten()
        .concat(
            Channel.fromPath("${baseDir}/tables/colon_atas_cell_type_marker_genes.csv")
        )
        .collect()

    SEED_METASTASIS(
        Channel.value([
            [id: "Seed_metastasis"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/43-Annotate_metastasis_seeds.py", checkIfExists: true)
        ]),
        ch_metastasis_seed.map { adata_path, markers ->
            [
                "adata_path": adata_path.name,
                "marker_genes_path": markers.name
            ]
        },
        ch_metastasis_seed
    )

    ch_lymph_node_seed = NEIGHBORS_LEIDEN_UMAP_SEED.out.adata
        .filter { id, _ -> id.contains("lymph_node") }
        .collect { _id, path -> path }
        .flatten()
        .concat(
            Channel.fromPath("${baseDir}/tables/colon_atas_cell_type_marker_genes.csv")
        )
        .collect()

    SEED_LYMPH_NODE(
        Channel.value([
            [id: "Seed_lymph_node"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/44-Annotate_lymph_node_seeds.py", checkIfExists: true)
        ]),
        ch_lymph_node_seed.map { adata_path, markers ->
            [
                "adata_path": adata_path.name,
                "marker_genes_path": markers.name
            ]
        },
        ch_lymph_node_seed
    )

    ch_csv = HVG_TUMOR.out.artifacts.concat(
                HVG_NORMAL.out.artifacts,
                HVG_BLOOD.out.artifacts,
                HVG_METASTASIS.out.artifacts,
                HVG_LYMPH_NODE.out.artifacts).flatten()
            .mix(
                SEED_TUMOR.out.artifacts.concat(
                SEED_NORMAL.out.artifacts,
                SEED_BLOOD.out.artifacts,
                SEED_METASTASIS.out.artifacts,
                SEED_LYMPH_NODE.out.artifacts).flatten()
                ).filter{ it -> it.name.contains(".csv") }
    
    ch_merge_seeds = ch_filtered_adata.concat(
            SEED_BD_RHAPSODY.out.artifacts
                .flatten()
                .filter { it -> it.name.contains("is_droplet") }
            )
            .collect()
        
    Merge_Seeds(
        Channel.value([
            [id: "Merge_seeds"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/50-Merge_seeds.py", checkIfExists: true)
        ]),
        ch_merge_seeds.map { adata_path, bd_is_droplet -> 
            [
                "adata_path": adata_path.name,
                "bd_is_droplet": bd_is_droplet.name,
                "hvg_dir": "."
            ] },
        ch_merge_seeds.mix(ch_csv).collect()
    )
    
    ch_scvi_merged = Merge_Seeds.out.artifacts
        .flatten()
        .filter{ it -> it.name.contains(".h5ad") }
        .map { it ->
            [it.baseName.split("\\-")[0], it]
        }
  
    SCVI(
        ch_scvi_merged,
        ["batch", null, "gene-batch"]
        )

    NEIGHBORS_LEIDEN_UMAP(
        SCVI.out.adata
            .collect { _id, path -> path }
            .flatten(),
        "X_scVI",
        2.0
        )
        
    SOLO_DOUBLETS(
        SCVI.out.adata
            .filter { id, _ -> id in [ 'merged' ] }
            .collect { id, path -> path }
            .flatten(),
        SCVI.out.scvi_model
            .filter { id, _ -> id in [ 'merged' ] }
            .collect { id, path -> path }
            .flatten()
    )

    ch_merge_solo = NEIGHBORS_LEIDEN_UMAP.out.adata
        .collect { id, path -> path }
        .flatten()
    
    Merge_SOLO(
        Channel.value([
            [id: "Merge_SOLO"],
            file("${baseDir}/analyses/03_qc_and_seed_annotations/60-Merge_solo.py", checkIfExists: true)
        ]),
        ch_merge_solo.map { adata_path -> 
            [
                "adata_path": adata_path.name,
                "solo_dir": "."
            ] },
        ch_merge_solo.mix(
            SOLO_DOUBLETS.out.doublets.filter{ it -> it.name.contains(".csv") }
        ).collect()
    )

    ch_scanvi = Merge_SOLO.out.artifacts
        .flatten()
        .filter{ it -> it.name.contains("merged") }
        .map { it ->
            [it.baseName.split("\\-")[0], it]
        }

    SCANVI(
        ch_scanvi,
        SCVI.out.scvi_model
            .filter { id, _ -> id in [ 'merged' ] }
            .collect { id, path -> path }
            .flatten(),
        ["batch", "cell_type_fine"]
    )

    NEIGHBORS_LEIDEN_UMAP_SCANVI(
        SCANVI.out.adata
            .collect { _id, path -> path }
            .flatten(),
        "X_scANVI",
        2.0
        )
       
    emit:
        adata = NEIGHBORS_LEIDEN_UMAP_SCANVI.out.adata
                    .collect { _id, path -> path }
                    .flatten()

}
