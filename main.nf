#!/usr/bin/env nextflow

include { build_atlas } from "./workflows/build_atlas.nf"

workflow {

     build_atlas()
}