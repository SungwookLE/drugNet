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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

# 아래코드에선 일단 사용치않았음
def mol2fp(mol):
    fp = AllChem.GetHashedMorganFingerprint(mol, 6, nBits=4096)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar

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
    

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, out_size):
        super(Net, self).__init__()
        
        # fc 레이어 3개와 출력 레이어
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_size)
        
        # 정규화
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)       

        # 활성화 함수
        self.activation = nn.LeakyReLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
     
    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.ln3(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc_out(out)
        return out

def train(train_loader, valid_loader, model, criterion, optimizer, epochs):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if epoch % 100 == 0:
            valid_loss = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    output = model(inputs)
                    loss = criterion(output, targets)
                    valid_loss += loss.item()
                    
            print(f'Epoch: {epoch}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Valid Loss: {(valid_loss/len(valid_HLM_loader))}')
            model.train()
    
    return model


if __name__ == "__main__":
    seed_everything(42) # Seed 고정
    train_d = pd.read_csv("./input/train.csv")
    test = pd.read_csv("./input/test.csv")

    train_plas= pd.read_csv("./input/train_out.csv")
    test_plas = pd.read_csv("./input/test_out.csv")

    # FPs column 추가
    train_d["plas"] = train_plas["pred_0"]
    test["plas"] = test_plas["pred_0"]

    # 사용할 column만 추출
    train_d.drop(["SMILES", "id"], axis=1, inplace=True)
    test.drop(["SMILES", "id"], axis=1, inplace=True)

    train_d["AlogP"].fillna(value=train_d["AlogP"].mean(), inplace=True)
    test["AlogP"].fillna(value=train_d["AlogP"].mean(), inplace=True)


    train_MLM = CustomDataset(df=train_d, target='MLM', is_test=False)
    train_HLM = CustomDataset(df=train_d, target='HLM', is_test=False)

    input_size = train_MLM.feature.shape[1]

    # Hyperparameter
    CFG = {'BATCH_SIZE': 256,
        'EPOCHS': 1000,
        'INPUT_SIZE': input_size,
        'HIDDEN_SIZE': 1024,
        'OUTPUT_SIZE': 1,
        'DROPOUT_RATE': 0.8,
        'LEARNING_RATE': 0.001}
    
    # train,valid split
    train_MLM_dataset, valid_MLM_dataset = train_test_split(train_MLM, test_size=0.2, random_state=42)
    train_HLM_dataset, valid_HLM_dataset = train_test_split(train_HLM, test_size=0.2, random_state=42)


    train_MLM_loader = DataLoader(dataset=train_MLM_dataset,
                              batch_size=CFG['BATCH_SIZE'],
                              shuffle=True)

    valid_MLM_loader = DataLoader(dataset=valid_MLM_dataset,
                                batch_size=CFG['BATCH_SIZE'],
                                shuffle=False)

    train_HLM_loader = DataLoader(dataset=train_HLM_dataset,
                                batch_size=CFG['BATCH_SIZE'],
                                shuffle=True)

    valid_HLM_loader = DataLoader(dataset=valid_HLM_dataset,
                                batch_size=CFG['BATCH_SIZE'],
                                shuffle=False)

    model_MLM = Net(CFG['INPUT_SIZE'],CFG['HIDDEN_SIZE'],CFG['DROPOUT_RATE'],CFG['OUTPUT_SIZE'])
    model_HLM = Net(CFG['INPUT_SIZE'],CFG['HIDDEN_SIZE'],CFG['DROPOUT_RATE'],CFG['OUTPUT_SIZE'])

    criterion = nn.MSELoss()
    optimizer_MLM = torch.optim.Adam(model_MLM.parameters(), lr=CFG['LEARNING_RATE'])
    optimizer_HLM = torch.optim.Adam(model_HLM.parameters(), lr=CFG['LEARNING_RATE'])

    print("Training Start: MLM")
    print(model_MLM)
    model_MLM = train(train_MLM_loader, valid_MLM_loader, model_MLM, criterion, optimizer_MLM, epochs=CFG['EPOCHS'])

    print("Training Start: HLM")
    print(model_HLM)
    model_HLM = train(train_HLM_loader, valid_HLM_loader, model_HLM, criterion, optimizer_HLM, epochs=CFG['EPOCHS'])

    