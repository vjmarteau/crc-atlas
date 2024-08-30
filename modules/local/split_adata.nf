#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

process SPLIT_ADATA_BY_BATCH {
    publishDir "${out_dir}", mode: "$mode"

    memory = '500 GB'
    cpus 4

    input:
        tuple val(id), path(adata)
        val(batch)

    output:
        path("*.h5ad"), emit: adatas_by_batch

	script:
	"""
    #!/usr/bin/env python3
    import sys
    import scanpy as sc
    from threadpoolctl import threadpool_limits

    threadpool_limits(${task.cpus})
    sc.settings.n_jobs = ${task.cpus}


    def split_adata_by_batch(adata, batch, _id):
        [
            adata[adata.obs[batch] == batch_key, :].write_h5ad(f"{batch_key}-{_id}.h5ad", compression="lzf")
            for batch_key in adata.obs[batch].unique()
            if adata[adata.obs[batch] == batch_key, :].shape[0] != 0
        ]
    

    def main():
        split_adata_by_batch(adata=sc.read("${adata}"), batch="${batch}", _id="${id}")


    if __name__ == "__main__":
        sys.exit(main())
	"""
}