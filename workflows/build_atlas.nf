#!/usr/bin/env nextflow

nextflow.enable.dsl = 2


include { Load_datasets } from "${baseDir}/subworkflows/Load_datasets.nf"

workflow build_atlas {

    Load_datasets()
    
}