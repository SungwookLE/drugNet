import random
import os

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rdkit import DataStructs
from rdkit.Chem import PandasTools, AllChem

from autogluon.tabular import TabularDataset, TabularPredictor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


class CustomDataset(Dataset):
    def __init__(self, df, target, is_test=False):
        self.df      = df
        self.target  = target # HLM or MLM
        self.feature = self.df[['AlogP', 'Molecular_Weight', 'Num_H_Acceptors', 'Num_H_Donors', 'Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea', 'plas']].values 
        self.label   = self.df[self.target].values

        self.is_test = is_test # train,valid / test

    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]

        if not self.is_test: # test가 아닌 경우(label 존재)
            return torch.tensor(feature, dtype=torch.float), torch.tensor(label, dtype=torch.float).unsqueeze(dim=-1) # feature, label
        else: # test인 경우
            return torch.tensor(feature, dtype=torch.float).float() # feature
        
    def __len__(self):
        return len(self.df)
    


if __name__ == "__main__":
    seed_everything(42) # Seed 고정
    train_d = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")

    train_plas= pd.read_csv("./input/train_out.csv")
    test_plas = pd.read_csv("./input/test_out.csv")

    # FPs column 추가
    train_d["plas"] = train_plas["pred_0"]
    test["plas"] = test_plas["pred_0"]

    train_d["AlogP"].fillna(value=train_d["AlogP"].mean(), inplace=True)
    test["AlogP"].fillna(value=train_d["AlogP"].mean(), inplace=True)

    print(train_d)
    # 사용할 column만 추출
    train_MLM = TabularDataset(train_d.drop(['id', "HLM"], axis=1))
    train_HLM = TabularDataset(train_d.drop(['id', "MLM"], axis=1))

    test = TabularDataset(test.drop(['id'], axis=1))

    
    # 이렇게 한 줄만 작성하면 내부에서 알아서 학습해줍니다.
    predictor_MLM = TabularPredictor(label='MLM', eval_metric='root_mean_squared_error',).fit(train_MLM)
    predictor_HLM = TabularPredictor(label='HLM', eval_metric='root_mean_squared_error',).fit(train_HLM)

    
    ld_board_MLM = predictor_MLM.leaderboard(train_MLM, silent=True)
    print(ld_board_MLM)

    ld_board_HLM = predictor_HLM.leaderboard(train_HLM, silent=True)
    print(ld_board_HLM)

    pred_MLM = predictor_MLM.predict(test)
    pred_HLM = predictor_HLM.predict(test)

    test["MLM"] = pred_MLM
    test["HLM"] = pred_HLM
    
    
    submission = test.to_csv("output/submission.csv", index=False)

