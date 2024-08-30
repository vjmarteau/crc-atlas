#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { JUPYTERNOTEBOOK as Bian_2018_Science;
          JUPYTERNOTEBOOK as Borras_2023_Cell_Discov;
          JUPYTERNOTEBOOK as Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol;
          JUPYTERNOTEBOOK as Che_2021_Cell_Discov;
          JUPYTERNOTEBOOK as Chen_2021_Cell;
          JUPYTERNOTEBOOK as Chen_2024_Cancer_Cell;
          JUPYTERNOTEBOOK as Conde_2022_Science;
          JUPYTERNOTEBOOK as DeVries_2023_Nature;
          JUPYTERNOTEBOOK as Elmentaite_2021_Nature;
          JUPYTERNOTEBOOK as GarridoTrigo_2023_Nat_Commun;
          JUPYTERNOTEBOOK as Giguelay_2022_Theranostics;
          JUPYTERNOTEBOOK as Guo_2022_JCI_Insight;
          JUPYTERNOTEBOOK as Han_2020_Nature;
          JUPYTERNOTEBOOK as Harmon_2023_Nat_Cancer;
          JUPYTERNOTEBOOK as He_2020_Genome_Biol;
          JUPYTERNOTEBOOK as HTAPP_HTAN;
          JUPYTERNOTEBOOK as Huang_2024_Nat_Cancer;
          JUPYTERNOTEBOOK as James_2020_Nat_Immunol;
          JUPYTERNOTEBOOK as Ji_2024_Cancer_Lett;
          JUPYTERNOTEBOOK as Ji_2024_PLoS_Genet;
          JUPYTERNOTEBOOK as Joanito_2022_Nat_Genet;
          JUPYTERNOTEBOOK as Khaliq_2022_Genome_Biol;
          JUPYTERNOTEBOOK as Kong_2023_Immunity;
          JUPYTERNOTEBOOK as Lee_2020_Nat_Genet;
          JUPYTERNOTEBOOK as Li_2017_Nat_Genet;
          JUPYTERNOTEBOOK as Li_2023_Cancer_Cell;
          JUPYTERNOTEBOOK as Liu_2022_Cancer_Cell;
          JUPYTERNOTEBOOK as Liu_2024_Cancer_Res;
          JUPYTERNOTEBOOK as Masuda_2022_JCI_Insight;
          JUPYTERNOTEBOOK as Mazzurana_2021_Cell_Res;
          JUPYTERNOTEBOOK as MUI_Innsbruck_AbSeq;
          JUPYTERNOTEBOOK as MUI_Innsbruck;
          JUPYTERNOTEBOOK as Parikh_2019_Nature;
          JUPYTERNOTEBOOK as Pelka_2021_Cell;
          JUPYTERNOTEBOOK as Qi_2022_Nat_Commun;
          JUPYTERNOTEBOOK as Qian_2020_Cell_Res;
          JUPYTERNOTEBOOK as Qin_2023_Cell_Rep_Med;
          JUPYTERNOTEBOOK as Sathe_2023_Clin_Cancer_Res;
          JUPYTERNOTEBOOK as Scheid_2023_J_EXP_Med;
          JUPYTERNOTEBOOK as Terekhanova_2023_Nature;
          JUPYTERNOTEBOOK as Thomas_2024_Nat_Med;
          JUPYTERNOTEBOOK as Tian_2023_Nat_Med;
          JUPYTERNOTEBOOK as Uhlitz_2021_EMBO_Mol_Med;
          JUPYTERNOTEBOOK as UZH_Zurich;
          JUPYTERNOTEBOOK as Wang_2020_J_Exp_Med;
          JUPYTERNOTEBOOK as Wang_2021_Adv_Sci;
          JUPYTERNOTEBOOK as Wang_2023_Sci_Adv;
          JUPYTERNOTEBOOK as Wu_2022_Cancer_Discov;
          JUPYTERNOTEBOOK as Wu_2024_Cell;
          JUPYTERNOTEBOOK as Yang_2023_Front_Oncol;
          JUPYTERNOTEBOOK as Zhang_2018_Nature;
          JUPYTERNOTEBOOK as Zhang_2020_Cell;
          JUPYTERNOTEBOOK as Zheng_2022_Signal_Transduct_Target_Ther } from "${baseDir}/modules/local/jupyternotebook/main"
include { JUPYTERNOTEBOOK as Harmonize_datasets } from "${baseDir}/modules/local/jupyternotebook/main"

/*
 * Concat available datasets
 *   - Load available datasets and map metadata
 *   - Harmonize gene annotations and metadata
 */

out_dir = file(params.outdir)
mode = params.publish_dir_mode


workflow Load_datasets {

    main:

        // start workflow

        ch_dataloader = Channel.fromPath(params.external_datasets, type: 'dir')
            .concat(
                Channel.fromPath(params.annotation, type: 'dir'),
                Channel.fromPath(params.reference_meta)
            )
            .collect()


        dataloader_nxfvars_params = ch_dataloader.map{ databases_path, annotation, reference_meta ->
            [
                "databases_path": databases_path.name,
                "annotation": annotation.name,
                "reference_meta": reference_meta.name
            ]
        }


        Bian_2018_Science(
            Channel.value([
                [id: "Bian_2018_Science"],
                file("${baseDir}/analyses/01_dataloader/Bian_2018_Science.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Bian_2018_Science = Bian_2018_Science.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Borras_2023_Cell_Discov(
            Channel.value([
                [id: "Borras_2023_Cell_Discov"],
                file("${baseDir}/analyses/01_dataloader/Borras_2023_Cell_Discov.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Borras_2023_Cell_Discov = Borras_2023_Cell_Discov.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol(
            Channel.value([
                [id: "Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol"],
                file("${baseDir}/analyses/01_dataloader/Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol = Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Che_2021_Cell_Discov(
            Channel.value([
                [id: "Che_2021_Cell_Discov"],
                file("${baseDir}/analyses/01_dataloader/Che_2021_Cell_Discov.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Che_2021_Cell_Discov = Che_2021_Cell_Discov.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Chen_2021_Cell(
            Channel.value([
                [id: "Chen_2021_Cell"],
                file("${baseDir}/analyses/01_dataloader/Chen_2021_Cell.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Chen_2021_Cell = Chen_2021_Cell.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Chen_2024_Cancer_Cell(
            Channel.value([
                [id: "Chen_2024_Cancer_Cell"],
                file("${baseDir}/analyses/01_dataloader/Chen_2024_Cancer_Cell.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Chen_2024_Cancer_Cell = Chen_2024_Cancer_Cell.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Conde_2022_Science(
            Channel.value([
                [id: "Conde_2022_Science"],
                file("${baseDir}/analyses/01_dataloader/Conde_2022_Science.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Conde_2022_Science = Conde_2022_Science.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        DeVries_2023_Nature(
            Channel.value([
                [id: "DeVries_2023_Nature"],
                file("${baseDir}/analyses/01_dataloader/deVries_2023_Nature.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_DeVries_2023_Nature = DeVries_2023_Nature.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Elmentaite_2021_Nature(
            Channel.value([
                [id: "Elmentaite_2021_Nature"],
                file("${baseDir}/analyses/01_dataloader/Elmentaite_2021_Nature.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Elmentaite_2021_Nature = Elmentaite_2021_Nature.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        GarridoTrigo_2023_Nat_Commun(
            Channel.value([
                [id: "GarridoTrigo_2023_Nat_Commun"],
                file("${baseDir}/analyses/01_dataloader/GarridoTrigo_2023_Nat_Commun.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_GarridoTrigo_2023_Nat_Commun = GarridoTrigo_2023_Nat_Commun.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Giguelay_2022_Theranostics(
            Channel.value([
                [id: "Giguelay_2022_Theranostics"],
                file("${baseDir}/analyses/01_dataloader/Giguelay_2022_Theranostics.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Giguelay_2022_Theranostics = Giguelay_2022_Theranostics.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Guo_2022_JCI_Insight(
            Channel.value([
                [id: "Guo_2022_JCI_Insight"],
                file("${baseDir}/analyses/01_dataloader/Guo_2022_JCI_Insight.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Guo_2022_JCI_Insight = Guo_2022_JCI_Insight.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Han_2020_Nature(
            Channel.value([
                [id: "Han_2020_Nature"],
                file("${baseDir}/analyses/01_dataloader/Han_2020_Nature.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Han_2020_Nature = Han_2020_Nature.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Harmon_2023_Nat_Cancer(
            Channel.value([
                [id: "Harmon_2023_Nat_Cancer"],
                file("${baseDir}/analyses/01_dataloader/Harmon_2023_Nat_Cancer.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Harmon_2023_Nat_Cancer = Harmon_2023_Nat_Cancer.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


         He_2020_Genome_Biol(
            Channel.value([
                [id: "He_2020_Genome_Biol"],
                file("${baseDir}/analyses/01_dataloader/He_2020_Genome_Biol.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_He_2020_Genome_Biol = He_2020_Genome_Biol.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
         HTAPP_HTAN(
            Channel.value([
                [id: "HTAPP_HTAN"],
                file("${baseDir}/analyses/01_dataloader/HTAPP_HTAN.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_HTAPP_HTAN = HTAPP_HTAN.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Huang_2024_Nat_Cancer(
            Channel.value([
                [id: "Huang_2024_Nat_Cancer"],
                file("${baseDir}/analyses/01_dataloader/Huang_2024_Nat_Cancer.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Huang_2024_Nat_Cancer = Huang_2024_Nat_Cancer.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        James_2020_Nat_Immunol(
            Channel.value([
                [id: "James_2020_Nat_Immunol"],
                file("${baseDir}/analyses/01_dataloader/James_2020_Nat_Immunol.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_James_2020_Nat_Immunol = James_2020_Nat_Immunol.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Ji_2024_Cancer_Lett(
            Channel.value([
                [id: "Ji_2024_Cancer_Lett"],
                file("${baseDir}/analyses/01_dataloader/Ji_2024_Cancer_Lett.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Ji_2024_Cancer_Lett = Ji_2024_Cancer_Lett.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Ji_2024_PLoS_Genet(
            Channel.value([
                [id: "Ji_2024_PLoS_Genet"],
                file("${baseDir}/analyses/01_dataloader/Ji_2024_PLoS_Genet.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Ji_2024_PLoS_Genet = Ji_2024_PLoS_Genet.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Joanito_2022_Nat_Genet(
            Channel.value([
                [id: "Joanito_2022_Nat_Genet"],
                file("${baseDir}/analyses/01_dataloader/Joanito_2022_Nat_Genet.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Joanito_2022_Nat_Genet = Joanito_2022_Nat_Genet.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Khaliq_2022_Genome_Biol(
            Channel.value([
                [id: "Khaliq_2022_Genome_Biol"],
                file("${baseDir}/analyses/01_dataloader/Khaliq_2022_Genome_Biol.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Khaliq_2022_Genome_Biol = Khaliq_2022_Genome_Biol.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        Kong_2023_Immunity(
            Channel.value([
                [id: "Kong_2023_Immunity"],
                file("${baseDir}/analyses/01_dataloader/Kong_2023_Immunity.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Kong_2023_Immunity = Kong_2023_Immunity.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }
        

        Lee_2020_Nat_Genet(
            Channel.value([
                [id: "Lee_2020_Nat_Genet"],
                file("${baseDir}/analyses/01_dataloader/Lee_2020_Nat_Genet.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Lee_2020_Nat_Genet = Lee_2020_Nat_Genet.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Li_2017_Nat_Genet(
            Channel.value([
                [id: "Li_2017_Nat_Genet"],
                file("${baseDir}/analyses/01_dataloader/Li_2017_Nat_Genet.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Li_2017_Nat_Genet = Li_2017_Nat_Genet.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Li_2023_Cancer_Cell(
            Channel.value([
                [id: "Li_2023_Cancer_Cell"],
                file("${baseDir}/analyses/01_dataloader/Li_2023_Cancer_Cell.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Li_2023_Cancer_Cell = Li_2023_Cancer_Cell.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Liu_2022_Cancer_Cell(
            Channel.value([
                [id: "Liu_2022_Cancer_Cell"],
                file("${baseDir}/analyses/01_dataloader/Liu_2022_Cancer_Cell.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Liu_2022_Cancer_Cell = Liu_2022_Cancer_Cell.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Liu_2024_Cancer_Res(
            Channel.value([
                [id: "Liu_2024_Cancer_Res"],
                file("${baseDir}/analyses/01_dataloader/Liu_2024_Cancer_Res.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Liu_2024_Cancer_Res = Liu_2024_Cancer_Res.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Masuda_2022_JCI_Insight(
            Channel.value([
                [id: "Masuda_2022_JCI_Insight"],
                file("${baseDir}/analyses/01_dataloader/Masuda_2022_JCI_Insight.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Masuda_2022_JCI_Insight = Masuda_2022_JCI_Insight.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Mazzurana_2021_Cell_Res(
            Channel.value([
                [id: "Mazzurana_2021_Cell_Res"],
                file("${baseDir}/analyses/01_dataloader/Mazzurana_2021_Cell_Res.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Mazzurana_2021_Cell_Res = Mazzurana_2021_Cell_Res.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        ch_MUI_Innsbruck_AbSeq = Channel.fromPath(params.own_datasets, type: 'dir')
            .concat(
                Channel.fromPath(params.annotation, type: 'dir'),
                Channel.fromPath(params.reference_meta)
            )
            .collect()
        MUI_Innsbruck_AbSeq(
            Channel.value([
                [id: "MUI_Innsbruck_AbSeq"],
                file("${baseDir}/analyses/01_dataloader/MUI_Innsbruck_AbSeq.py", checkIfExists: true)
                ]),
            ch_MUI_Innsbruck_AbSeq.map{ data_path, annotation, reference_meta ->
                [
                    "data_path": data_path.name,
                    "annotation": annotation.name,
                    "reference_meta": reference_meta.name
                ]},
            ch_MUI_Innsbruck_AbSeq
            )
        ch_MUI_Innsbruck_AbSeq = MUI_Innsbruck_AbSeq.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        ch_MUI_Innsbruck = Channel.fromPath(params.own_datasets, type: 'dir')
            .concat(
                Channel.fromPath(params.annotation, type: 'dir'),
                Channel.fromPath(params.reference_meta)
            )
            .collect()
        MUI_Innsbruck(
            Channel.value([
                [id: "MUI_Innsbruck"],
                file("${baseDir}/analyses/01_dataloader/MUI_Innsbruck.py", checkIfExists: true)
                ]),
            ch_MUI_Innsbruck.map{ data_path, annotation, reference_meta ->
                [
                    "data_path": data_path.name,
                    "annotation": annotation.name,
                    "reference_meta": reference_meta.name
                ]},
            ch_MUI_Innsbruck
            )
        ch_MUI_Innsbruck = MUI_Innsbruck.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Parikh_2019_Nature(
            Channel.value([
                [id: "Parikh_2019_Nature"],
                file("${baseDir}/analyses/01_dataloader/Parikh_2019_Nature.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Parikh_2019_Nature = Parikh_2019_Nature.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Pelka_2021_Cell(
            Channel.value([
                [id: "Pelka_2021_Cell"],
                file("${baseDir}/analyses/01_dataloader/Pelka_2021_Cell.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Pelka_2021_Cell = Pelka_2021_Cell.out.artifacts
            .flatten()
            .filter { it -> it.name.contains(".h5ad") }


        Qi_2022_Nat_Commun(
            Channel.value([
                [id: "Qi_2022_Nat_Commun"],
                file("${baseDir}/analyses/01_dataloader/Qi_2022_Nat_Commun.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Qi_2022_Nat_Commun = Qi_2022_Nat_Commun.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Qian_2020_Cell_Res(
            Channel.value([
                [id: "Qian_2020_Cell_Res"],
                file("${baseDir}/analyses/01_dataloader/Qian_2020_Cell_Res.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Qian_2020_Cell_Res = Qian_2020_Cell_Res.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Qin_2023_Cell_Rep_Med(
            Channel.value([
                [id: "Qin_2023_Cell_Rep_Med"],
                file("${baseDir}/analyses/01_dataloader/Qin_2023_Cell_Rep_Med.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Qin_2023_Cell_Rep_Med = Qin_2023_Cell_Rep_Med.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Sathe_2023_Clin_Cancer_Res(
            Channel.value([
                [id: "Sathe_2023_Clin_Cancer_Res"],
                file("${baseDir}/analyses/01_dataloader/Sathe_2023_Clin_Cancer_Res.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Sathe_2023_Clin_Cancer_Res = Sathe_2023_Clin_Cancer_Res.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        Scheid_2023_J_EXP_Med(
            Channel.value([
                [id: "Scheid_2023_J_EXP_Med"],
                file("${baseDir}/analyses/01_dataloader/Scheid_2023_J_EXP_Med.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Scheid_2023_J_EXP_Med = Scheid_2023_J_EXP_Med.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Terekhanova_2023_Nature(
            Channel.value([
                [id: "Terekhanova_2023_Nature"],
                file("${baseDir}/analyses/01_dataloader/Terekhanova_2023_Nature.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Terekhanova_2023_Nature = Terekhanova_2023_Nature.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Thomas_2024_Nat_Med(
            Channel.value([
                [id: "Thomas_2024_Nat_Med"],
                file("${baseDir}/analyses/01_dataloader/Thomas_2024_Nat_Med.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Thomas_2024_Nat_Med = Thomas_2024_Nat_Med.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Tian_2023_Nat_Med(
            Channel.value([
                [id: "Tian_2023_Nat_Med"],
                file("${baseDir}/analyses/01_dataloader/Tian_2023_Nat_Med.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Tian_2023_Nat_Med = Tian_2023_Nat_Med.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Uhlitz_2021_EMBO_Mol_Med(
            Channel.value([
                [id: "Uhlitz_2021_EMBO_Mol_Med"],
                file("${baseDir}/analyses/01_dataloader/Uhlitz_2021_EMBO_Mol_Med.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Uhlitz_2021_EMBO_Mol_Med = Uhlitz_2021_EMBO_Mol_Med.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        ch_UZH_Zurich = Channel.fromPath(params.own_datasets, type: 'dir')
            .concat(
                Channel.fromPath(params.annotation, type: 'dir'),
                Channel.fromPath(params.reference_meta)
            )
            .collect()
        UZH_Zurich(
            Channel.value([
                [id: "UZH_Zurich"],
                file("${baseDir}/analyses/01_dataloader/UZH_Zurich.py", checkIfExists: true)
                ]),
            ch_UZH_Zurich.map{ data_path, annotation, reference_meta ->
                [
                    "data_path": data_path.name,
                    "annotation": annotation.name,
                    "reference_meta": reference_meta.name
                ]},
            ch_UZH_Zurich
            )
        ch_UZH_Zurich = UZH_Zurich.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Wang_2020_J_Exp_Med(
            Channel.value([
                [id: "Wang_2020_J_Exp_Med"],
                file("${baseDir}/analyses/01_dataloader/Wang_2020_J_Exp_Med.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Wang_2020_J_Exp_Med = Wang_2020_J_Exp_Med.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Wang_2021_Adv_Sci(
            Channel.value([
                [id: "Wang_2021_Adv_Sci"],
                file("${baseDir}/analyses/01_dataloader/Wang_2021_Adv_Sci.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Wang_2021_Adv_Sci = Wang_2021_Adv_Sci.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Wang_2023_Sci_Adv(
            Channel.value([
                [id: "Wang_2023_Sci_Adv"],
                file("${baseDir}/analyses/01_dataloader/Wang_2023_Sci_Adv.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Wang_2023_Sci_Adv = Wang_2023_Sci_Adv.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Wu_2022_Cancer_Discov(
            Channel.value([
                [id: "Wu_2022_Cancer_Discov"],
                file("${baseDir}/analyses/01_dataloader/Wu_2022_Cancer_Discov.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Wu_2022_Cancer_Discov = Wu_2022_Cancer_Discov.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Wu_2024_Cell(
            Channel.value([
                [id: "Wu_2024_Cell"],
                file("${baseDir}/analyses/01_dataloader/Wu_2024_Cell.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Wu_2024_Cell = Wu_2024_Cell.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        
        Yang_2023_Front_Oncol(
            Channel.value([
                [id: "Yang_2023_Front_Oncol"],
                file("${baseDir}/analyses/01_dataloader/Yang_2023_Front_Oncol.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Yang_2023_Front_Oncol = Yang_2023_Front_Oncol.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Zhang_2018_Nature(
            Channel.value([
                [id: "Zhang_2018_Nature"],
                file("${baseDir}/analyses/01_dataloader/Zhang_2018_Nature.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Zhang_2018_Nature = Zhang_2018_Nature.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Zhang_2020_Cell(
            Channel.value([
                [id: "Zhang_2020_Cell"],
                file("${baseDir}/analyses/01_dataloader/Zhang_2020_Cell.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Zhang_2020_Cell = Zhang_2020_Cell.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        Zheng_2022_Signal_Transduct_Target_Ther(
            Channel.value([
                [id: "Zheng_2022_Signal_Transduct_Target_Ther"],
                file("${baseDir}/analyses/01_dataloader/Zheng_2022_Signal_Transduct_Target_Ther.py", checkIfExists: true)
            ]),
            dataloader_nxfvars_params,
            ch_dataloader
            )
        ch_Zheng_2022_Signal_Transduct_Target_Ther = Zheng_2022_Signal_Transduct_Target_Ther.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }


        // Combine all datasets into a single channel
        datasets = ch_Bian_2018_Science.concat(
            ch_Borras_2023_Cell_Discov,
            ch_Burclaff_2022_Cell_Mol_Gastroenterol_Hepatol,
            ch_Che_2021_Cell_Discov,
            ch_Chen_2021_Cell,
            ch_Chen_2024_Cancer_Cell,
            ch_Conde_2022_Science,
            ch_DeVries_2023_Nature,
            ch_Elmentaite_2021_Nature,
            ch_GarridoTrigo_2023_Nat_Commun,
            ch_Giguelay_2022_Theranostics,
            ch_Guo_2022_JCI_Insight,
            ch_Han_2020_Nature,
            ch_Harmon_2023_Nat_Cancer,
            ch_He_2020_Genome_Biol,
            ch_HTAPP_HTAN,
            ch_Huang_2024_Nat_Cancer,
            ch_James_2020_Nat_Immunol,
            ch_Ji_2024_Cancer_Lett,
            ch_Ji_2024_PLoS_Genet,
            ch_Joanito_2022_Nat_Genet,
            ch_Khaliq_2022_Genome_Biol,
            ch_Kong_2023_Immunity,
            ch_Lee_2020_Nat_Genet,
            ch_Li_2017_Nat_Genet,
            ch_Li_2023_Cancer_Cell,
          //ch_Liu_2022_Cancer_Cell,
            ch_Liu_2024_Cancer_Res,
          //ch_Masuda_2022_JCI_Insight,
            ch_Mazzurana_2021_Cell_Res,
            ch_MUI_Innsbruck_AbSeq,
            ch_MUI_Innsbruck,
            ch_Parikh_2019_Nature,
            ch_Pelka_2021_Cell,
            ch_Qi_2022_Nat_Commun,
            ch_Qian_2020_Cell_Res,
            ch_Qin_2023_Cell_Rep_Med,
            ch_Sathe_2023_Clin_Cancer_Res,
            ch_Scheid_2023_J_EXP_Med,
            ch_Terekhanova_2023_Nature,
            ch_Thomas_2024_Nat_Med,
            ch_Tian_2023_Nat_Med,
            ch_Uhlitz_2021_EMBO_Mol_Med,
            ch_UZH_Zurich,
            ch_Wang_2020_J_Exp_Med,
            ch_Wang_2021_Adv_Sci,
            ch_Wang_2023_Sci_Adv,
            ch_Wu_2022_Cancer_Discov,
            ch_Wu_2024_Cell,
            ch_Yang_2023_Front_Oncol,
            ch_Zhang_2018_Nature,
            ch_Zhang_2020_Cell,
            ch_Zheng_2022_Signal_Transduct_Target_Ther
            )


        ch_Harmonize_datasets = Channel.fromPath(params.annotation, type: 'dir')
            .concat(
                Channel.fromPath(params.reference_meta)
            )
            .collect()
        Harmonize_datasets(
            Channel.value([
                [id: "01-Harmonize_datasets"],
                file("${baseDir}/analyses/02_harmonize_datasets/Harmonize_datasets.py", checkIfExists: true)
                ]),
            ch_Harmonize_datasets.map{ annotation, reference_meta ->
                [
                    "annotation": annotation.name,
                    "reference_meta": reference_meta.name,
                    "datasets_path": "."
                ]},
            ch_Harmonize_datasets.mix(datasets
                .filter { path -> path.baseName.contains("-adata") })
                .collect()
            )


        ch_Harmonize_datasets = Harmonize_datasets.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains(".h5ad") }

        ch_adata_gtf = Harmonize_datasets.out.artifacts
            .flatten()
            .filter{ it -> it.name.contains("gtf.csv") }
            
    emit:
        datasets = ch_Harmonize_datasets
        adata_gtf = ch_adata_gtf
        
}
