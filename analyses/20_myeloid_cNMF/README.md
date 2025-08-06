# cNMF on Myeloid cells in single-cell colorectal cancer atlas

### Data availability

Data can be downloaded from the [project website](https://crc.icbi.at/)


### Environments

```bash
conda env create -n cNMF -f 'envs/cNMF.yaml'
```

### Notebooks

Notebooks for data analysis: 
* `01_preprocessing.ipynb`: Filtering cohorts, cells and genes
* `02_cNMF.ipynb`: consensus non-negative matrix factorization 
* `03_ORA_GSEA.ipynb`: ORA and GSEA on consensus genes programs