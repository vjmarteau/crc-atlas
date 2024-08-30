#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SPLIT_ADATA_BY_BATCH } from "${baseDir}/modules/local/split_adata.nf"

process SCAR {
    publishDir "${out_dir}", mode: "$mode"
    label "gpu"
    errorStrategy = 'finish'
    memory ='128 GB'

    input:
        tuple val(id), path(raw_adata), path(filtered_adata)

    output:
        path("${id}-denoised_adata.h5ad"), emit: denoised_adata

	script:
	"""
    mkdir -p /tmp/$USER/torch_kernel_cache
    export OPENBLAS_NUM_THREADS=${task.cpus} OMP_NUM_THREADS=${task.cpus}  \\
        MKL_NUM_THREADS=${task.cpus} OMP_NUM_cpus=${task.cpus}  \\
        MKL_NUM_cpus=${task.cpus} OPENBLAS_NUM_cpus=${task.cpus}
    scAR.py \\
        --raw_adata=${raw_adata} \\
        --filtered_adata=${filtered_adata} \\
        --cpus=${task.cpus} \\
        --adata_out="${id}-denoised_adata.h5ad"
	"""
}


process MERGE_SCAR {
    publishDir "${out_dir}", mode: "$mode"
    
    cpus = 12

    memory ='50 GB'

    input:
        path(adatas_path)

    output:
        path("*-denoised_adata.h5ad"), emit: scAR_denoised

	script:
	"""
    #!/usr/bin/env python3
    import sys
    from pathlib import Path
    import anndata
    import scanpy as sc
    from threadpoolctl import threadpool_limits
    from tqdm.contrib.concurrent import process_map

    threadpool_limits(${task.cpus})
    sc.settings.n_jobs = ${task.cpus}

    def concat_adatas_by_study(cpus):

        # Get file paths and reorder by study and batch
        files = [str(x) for x in Path(".").glob("*.h5ad")]
        files = [x for x in files if not isinstance(x, float)]
        files.sort()

        # Concat adatas from specified directory
        adatas = process_map(sc.read_h5ad, files, max_workers=cpus)
        adata = anndata.concat(adatas, index_unique=".", join="outer", fill_value=0)
        adata.obs_names = adata.obs_names.str.rsplit(".").str[0]
        assert adata.obs_names.is_unique

        # map back gene symbols to .var
        gtf = adatas[0].var.reset_index(names=["var_names"])

        for column in [col for col in gtf.columns if col != "var_names"]:
            adata.var[column] = adata.var_names.map(gtf.set_index("var_names")[column])
        
        assert adata.var_names.is_unique

        dataset = adata.obs["dataset"].unique()[0]
        adata.write_h5ad(f"{dataset}-denoised_adata.h5ad", compression="lzf")
        

    def main():
        concat_adatas_by_study(cpus=${task.cpus})


    if __name__ == "__main__":
        sys.exit(main())
	"""
}

workflow REMOVE_AMBIENT_RNA {
    take:
        datasets

    main:
    // Create a subset channel with raw counts and filtered counts as input for scAR

        ch_datasets = datasets.map { it ->
            [it.baseName.split("\\-")[0], it]
        }

        ch_scAR = ch_datasets
            .filter { id, path -> path.baseName.contains("-raw") }
            .join(ch_datasets.filter { id, path -> path.baseName.contains("-adata") })
            .map { id, raw, adata -> tuple(raw, adata) }
            .collect()


        SPLIT_ADATA_BY_BATCH(
            ch_scAR.flatten().map { it -> [it.baseName, it] },
            Channel.value( "batch" )
        )


        ch_scAR_input = SPLIT_ADATA_BY_BATCH.out.adatas_by_batch
            .flatten()
            .flatMap { it ->
                parts = it.baseName.split("\\-")
                batch = parts[0] + "-" + parts[1]
                adata_type = parts[2]
                return [[batch, adata_type, it]]
            }
            .groupTuple(sort: true)
            .map { batch, types, paths ->
                [batch] + paths.findAll { it.baseName.contains("-raw") } + paths.findAll { it.baseName.contains("-adata") }
            }


        SCAR(ch_scAR_input)


        ch_denoised = SCAR.out.denoised_adata
            .map { file -> [file.name.split("\\-")[1], file] }
            .groupTuple(sort: true)
            .flatMap { id, filesList -> 
                filesList.groupBy { id }
                         .collect { dataset_id, path -> path }
            }


        MERGE_SCAR(ch_denoised)


        // Replace denoised dataset in input datasets, keep adata if denoising not possible
        ch_scAR = datasets.mix(MERGE_SCAR.out.scAR_denoised)
            .map { it -> [it.baseName.split("\\-")[0], it] }
            .groupTuple(sort: true)
            .map { id, paths ->
                [id, *paths.findAll { it.baseName.contains("-denoised") } ?: paths.findAll { it.baseName.contains("-adata") }]
            }

    emit:
    scAR = ch_scAR
}
