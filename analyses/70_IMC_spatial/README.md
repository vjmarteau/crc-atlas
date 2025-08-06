# Spatial analysis of IMC data

### Data availability

Data can be downloaded from the  *BioImage Archive* ([S-BIAD2208](https://doi.org/10.6019/S-BIAD2208)),
however, the provided code includes an automated download routine, so no manual download should be required.

### Environments

```bash
conda env create -n IMC -f '../../envs/IMC.yaml'
```

### Notebook and Script

Python notebook for cellular neighbohood analysis: 
* `IMC_CN_analysis.ipynb`: full analysis as iPython notebook

Python script for cellular neighbohood analysis: 
* `IMC_CN_analysis.py`: full analysis as standalone python script

For adjustments please edit the relevant parts in the `Settings` section, i.e.:

* `input_data_dir`: directory in the input data will be placed
* `output_dir`: directory in which the results will be stored