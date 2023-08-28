
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
python drugnet_pred.py --test_path ./input/test.csv --preds_path ./output/submission.csv
python drugnet_train.py --data_path ./input/train.csv --features_generator morgan morgan_count --separate_test_path ./input/test.csv --dataset_type regression --metric rmse --cuda --split_type random --features_scaling --ensemble_size 1 --use_input_features
python drugnet_train.py --data_path ./input/train.csv --separate_test_path ./input/test.csv --dataset_type regression --metric rmse --cuda --split_type random --features_scaling --ensemble_size 1 
```