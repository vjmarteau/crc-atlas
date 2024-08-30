#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

process NEIGHBORS {
    cpus 8
    memory ='990 GB'

    input:
        tuple val(id), path(adata)
        val use_rep

    output:
        tuple val(id), path("*.h5ad"), emit: adata

    script:
    """
    #!/usr/bin/env python

    import scanpy as sc
    from threadpoolctl import threadpool_limits

    threadpool_limits(${task.cpus})
    sc.settings.n_jobs = ${task.cpus}

    adata = sc.read_h5ad("${adata}")
    sc.pp.neighbors(adata, use_rep="${use_rep}")
    adata.write_h5ad("${id}-neighbors.h5ad")
    """
}


process LEIDEN_PAGA_UMAP {
    cpus 8
    memory ='990 GB'

    input:
        tuple val(id), path(adata)
        each resolution

    output:
        tuple val(id), path("*.h5ad"), emit: adata

    script:
    """
    #!/usr/bin/env python

    import scanpy as sc
    from threadpoolctl import threadpool_limits

    threadpool_limits(${task.cpus})
    sc.settings.n_jobs = ${task.cpus}

    adata = sc.read_h5ad("${adata}")
    sc.tl.leiden(adata, resolution=${resolution}, flavor="igraph", n_iterations=-1)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos='paga')
    adata.write_h5ad("${id}-umap.h5ad")
    """
}

workflow NEIGHBORS_LEIDEN_UMAP {
    take:
        adata
        neihbors_rep
        leiden_res

    main:

        ch_adatas = adata.map { it ->
            [it.baseName.split("\\-")[0], it]
        }

        NEIGHBORS(ch_adatas, neihbors_rep)
        LEIDEN_PAGA_UMAP(NEIGHBORS.out.adata, leiden_res)

    emit:
        adata = LEIDEN_PAGA_UMAP.out.adata
}
