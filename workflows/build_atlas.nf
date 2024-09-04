#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { Load_datasets } from "${baseDir}/subworkflows/Load_datasets.nf"
include { Integrate_datasets } from "${baseDir}/subworkflows/Integrate_datasets.nf"
include { Tidy_atlas } from "${baseDir}/subworkflows/Tidy_atlas.nf"

workflow build_atlas {
    
    Load_datasets()
    
    Integrate_datasets(
        Load_datasets.out.datasets,
        Load_datasets.out.adata_gtf
    )

    Tidy_atlas(Integrate_datasets.out.adata)
    
}