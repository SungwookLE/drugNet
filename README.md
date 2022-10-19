# PredPS #

Source code for our paper "PredPS: attention-based graph neural network for predicting stability of compounds in human plasma"
This code was built based on ChemProp (https://github.com/chemprop/chemprop) and CMPNN (https://github.com/SY575/CMPNN).


##Procedure

**Note**:
This source code was developed in Ubuntu 18.04.5 LTS with Python v3.7 and PyTorch v1.12.1.

1. Clone the repository

	git clone https://wdjang@bitbucket.org/krict-ai/predps.git

2. Create and activate a conda environment

	conda env create -f environment.yaml
	
	conda activate predps

##Example

- Run PredPS using sample input file

	python PredPS_pred.py --test_path ./test_input.csv --preds_path output_results.csv 




	
