#!/usr/bin/env nextflow
nextflow.enable.dsl = 2


process SCVI {
    publishDir "${out_dir}", mode: "$mode"
    label "gpu"
    errorStrategy = 'finish'
    //memory ='300 GB'
    
    input:
        tuple val(id), path(input_adata)
        tuple val(batch_key), val(labels_key), val(dispersion)

    output:
        tuple val(id), path("*-integrated_scvi.h5ad"), emit: adata
        tuple val(id), path("*-scvi_model"), emit: scvi_model

    script:
    def labels_key_arg = labels_key ? "--labels_key=${labels_key}" : ""
    def dispersion_arg = dispersion ? "--model_gene_dispersion=${dispersion}" : ""
    def args = task.ext.args ?: ''
    """
    mkdir -p /tmp/$USER/torch_kernel_cache
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus}  \\
        MKL_NUM_THREADS=${task.cpus} OMP_NUM_cpus=${task.cpus}  \\
        MKL_NUM_cpus=${task.cpus} OPENBLAS_NUM_cpus=${task.cpus}
    scVI.py \\
    --input_adata=${input_adata} \\
    --batch_key=${batch_key} \\
    ${labels_key_arg} \\
    ${dispersion_arg} \\
    --model_out=${id}-scvi_model \\
    --adata_out=${id}-integrated_scvi.h5ad \\
    --cpus=${task.cpus} \\
    $args
    """
}

process SCANVI {
    publishDir "${out_dir}", mode: "$mode"
    label "gpu"
    errorStrategy = 'finish'
    //memory ='990 GB'
    
    input:
        tuple val(id), path(input_adata)
        each path(scvi_model)
        tuple val(batch_key), val(labels_key)

    output:
        tuple val(id), path("*-integrated_scanvi.h5ad"), emit: adata
        tuple val(id), path("*-scanvi_model"), emit: scanvi_model

    script:
    """
    mkdir -p /tmp/$USER/torch_kernel_cache
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus}  \\
        MKL_NUM_THREADS=${task.cpus} OMP_NUM_cpus=${task.cpus}  \\
        MKL_NUM_cpus=${task.cpus} OPENBLAS_NUM_cpus=${task.cpus}
    scANVI.py \\
    --input_adata=${input_adata} \\
    --input_model=${scvi_model} \\
    --batch_key=${batch_key} \\
    --labels_key=${labels_key} \\
    --model_out=${id}-scanvi_model \\
    --adata_out=${id}-integrated_scanvi.h5ad \\
    --cpus=${task.cpus}
    """
}
