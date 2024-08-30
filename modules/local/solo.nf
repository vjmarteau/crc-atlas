#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SPLIT_ADATA_BY_BATCH } from "${baseDir}/modules/local/split_adata.nf"

process SOLO {
    publishDir "${out_dir}", mode: "$mode"
    label "gpu"
    errorStrategy = 'finish'
    memory ='128 GB'

    input:
        tuple val(id), path(adata)
        each path(scvi_model)
        tuple val(batch_key), val(labels_key)

    output:
        path("*solo_doublet_status.csv"), emit: doublets

    script:
    def labels_key_arg = labels_key ? "--labels_key=${labels_key}" : ""
    """
    mkdir -p /tmp/$USER/torch_kernel_cache
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus}  \\
        MKL_NUM_THREADS=${task.cpus} OMP_NUM_cpus=${task.cpus}  \\
        MKL_NUM_cpus=${task.cpus} OPENBLAS_NUM_cpus=${task.cpus}
    solo.py \\
    --input_adata=${adata} \\
    --input_model=${scvi_model} \\
    --batch_key=${batch_key} \\
    ${labels_key_arg} \\
    --doublet_status=${id}-solo_doublet_status.csv \\
    --cpus=${task.cpus}
    """
}

workflow SOLO_DOUBLETS {
    take:
        adata
        input_model

    main:

/*
    SOLO( 
        adata.map { it -> [it.baseName.split("\\-")[0], it] },
        input_model,
        ["gene_dispersion_label", "batch"]
    ) 
*/
    SPLIT_ADATA_BY_BATCH(
            adata.map { it -> [it.baseName.split("\\-")[0], it] },
            Channel.value( "dataset" )
        )
    
    // Remove non droplet based datasets
    ch_droplet_based = SPLIT_ADATA_BY_BATCH.out.adatas_by_batch
        .flatten()
        .map { it -> [it.baseName.split("\\-")[0], it] }
        .map { tuple -> [tuple[0], tuple[1]] }.filter { 
                id, _ -> 
                id !in [
                    'Bian_2018_Dong_protocol',
                    'Bian_2018_Tang_protocol',
                    'Han_2020',
                    'Li_2017',
                    'MUI_Innsbruck',
                    'MUI_Innsbruck_AbSeq',
                    'Mazzurana_2021_CD127Pos',
                    'UZH_Zurich_CD45Pos',
                    'UZH_Zurich_healthy_blood',
                    'Wang_2021',
                    'Wang_2023_CD45Neg',
                    'Wang_2023_CD45Pos',
                    'Zhang_2018_CD3Pos',
                    'Zhang_2020_CD45Pos_CD45Neg'
                ] 
            }

   SOLO(
        ch_droplet_based,
        input_model,
        ["batch", null]
    ) 


    emit:
        doublets = SOLO.out.doublets

}