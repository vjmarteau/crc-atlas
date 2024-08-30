#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { SPLIT_ADATA_BY_BATCH } from "${baseDir}/modules/local/split_adata.nf"

process SCORE_CELL_CYCLE {
    publishDir "${out_dir}", mode: "$mode"

    input:
        tuple val(id), path(input_adata)

    output:
        path("${id}-cell_cycle_score.csv"), emit: phase

		script:
	"""
    #!/usr/bin/env python3
    import sys
    import scanpy as sc
    import pandas as pd
    from threadpoolctl import threadpool_limits

    threadpool_limits(${task.cpus})
    sc.settings.n_jobs = ${task.cpus}

    def Score_cell_cycle(adata):

        # Annotate cell cycle phase
        cell_cycle_genes = [
            "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
            "PRIM1", "UHRF1", "CENPU", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP",
            "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1",
            "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8", "HMGB2", "CDK1", "NUSAP1",
            "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3",
            "PIMREG", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
            "KIF20B", "HJURP", "CDCA3", "JPT1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5",
            "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
            "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]

        # Split into 2 lists
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]
        cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]

        sc.pp.normalize_total(adata, target_sum=None)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

        sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

        # Save cell cycle phase
        pd.DataFrame(adata.obs[["S_score", "G2M_score", "phase"]]).reset_index(names=["obs_names"]).to_csv(
            f"${id}-cell_cycle_score.csv", index=False
        )

        return adata
        

    def main():
        adata = Score_cell_cycle(adata=sc.read_h5ad("${input_adata}"))


    if __name__ == "__main__":
        sys.exit(main())
	"""
}

workflow CELL_CYCLE_PHASE {
    take:
        adata

    main:


    SPLIT_ADATA_BY_BATCH(
            adata,
            Channel.value( "batch" )
        )

    SCORE_CELL_CYCLE(
        SPLIT_ADATA_BY_BATCH.out.adatas_by_batch
            .flatten()
            .map { it -> [it.baseName.split("\\-")[0], it] }
    )
    

    emit:
        cell_cyle_phase = SCORE_CELL_CYCLE.out.phase

}