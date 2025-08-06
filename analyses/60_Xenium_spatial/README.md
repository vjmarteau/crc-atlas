# Xenium spatial transcriptomics analysis

### Data availability

Raw and processed data can be downloaded from the BioImage Archive ([S-BIAD2208](https://doi.org/10.6019/S-BIAD2208))


### Environments

```bash
conda env create -n xenium -f 'envs/xenium.yml'
conda env create -n nichecompass -f 'envs/nichecompass.yml'
```

### Notebooks

Notebooks for data analysis: 
00_prepare_data.ipynb: Select and merge samples from Xenium output data 
01_preprocessing.ipynb: Preprocess the mergde data 
02_annotation.ipynb: Cell type annotation 
03_nichecompass.ipynb: NicheCompass analysis 
04_niches.ipynb: Niche analysis 
05_results.ipynb: Code to reproduce the figures from the processed data 
