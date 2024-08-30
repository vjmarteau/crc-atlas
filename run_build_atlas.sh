#!/bin/bash

NXF_VER=23.04.0 nextflow run ./main.nf --workflow build_atlas -profile cluster \
-w /data/scratch/marteau/nf-work-dir/crc-atlas/work \
-params-file params_build_atlas.yaml \
-resume
