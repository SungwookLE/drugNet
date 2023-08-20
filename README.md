
# DrugNet

Preparing the Dacon '23 "신약개발 AI"...    
We are inspired by the paper "PredPS: attention-based graph neural network for predicting stability of compounds in human plasma" and took the baseline code structure below.
 1. https://bitbucket.org/krict-ai/predps/src/main/    
 2. https://github.com/chemprop/chemprop  
 3. https://github.com/SY575/CMPNN  
 
## Pre-requirement

This source code was developed in Ubuntu 20.04 LTS with Python v3.7 and PyTorch v1.7.1. And It has cuda of 10.1, cudnn of 7.6.3 version. You can install the proper pytorch(with gpu) version according to your cuda and cudnn version.

Create and activate conda environment
```
conda env create -f environment.yaml
conda activate drugnet
```

## Example (not completed yet)
- Run PredPS using sample input file  
```
python drugnet_pred.py --test_path ./test_input.csv --preds_path output_results.csv
``` 