#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { JUPYTERNOTEBOOK as Tidy_annotation;
          JUPYTERNOTEBOOK as Tidy_metadata;
          JUPYTERNOTEBOOK as Export_atlas } from "${baseDir}/modules/local/jupyternotebook/main"

out_dir = file(params.outdir)
mode = params.publish_dir_mode

workflow Tidy_atlas {

    take:
        adata

    main:

    ch_tidy_annot = adata
        .concat(
            Channel.fromPath("${baseDir}/results/v1/final/h5ads/crc_atlas-epithelial-adata.h5ad",
            checkIfExists: true
            )
        )
        .collect()


    Tidy_annotation(
        Channel.value([
            [id: "Tidy_annotation"],
            file("${baseDir}/analyses/04_tidy_atlas/Tidy_annotation.py", checkIfExists: true)
        ]),
        ch_tidy_annot.map { adata_path, adata_epi_path ->
            [
                "adata_path": adata_path.name,
                "adata_epi_path": adata_epi_path.name
            ] },
        ch_tidy_annot
    )

     ch_tidy_atlas = Tidy_annotation.out.artifacts
        .flatten()
        .filter { it -> it.name.contains("adata.h5ad") }
        .concat(
            Channel.fromPath("${baseDir}/tables/colon_atas_cell_type_marker_genes.csv")
        )
        .collect()

    Tidy_metadata(
        Channel.value([
            [id: "Tidy_metadata"],
            file("${baseDir}/analyses/04_tidy_atlas/Tidy_metadata.py", checkIfExists: true)
        ]),
        ch_tidy_atlas.map { adata_path, markers ->
            [
                "adata_path": adata_path.name,
                "marker_genes_path": markers.name
            ] },
        ch_tidy_atlas
    )

    core_atlas = Tidy_metadata.out.artifacts
        .flatten()
        .filter { it -> it.name.contains("atlas-adata.h5ad") }


    Export_atlas(
        Channel.value([
            [id: "Export_atlas"],
            file("${baseDir}/analyses/04_tidy_atlas/Export_atlas.py", checkIfExists: true)
        ]),
        core_atlas.map{ it -> ["adata_path": it.name]},
        core_atlas
    )

}