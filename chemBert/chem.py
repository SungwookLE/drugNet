'''
## Chemical data 획득

1. atomic features
- `get_atomic_features` 함수를 통해서 분자의 atomic features를 획들할 수 있음
- features는 `mendeleev` 패키지에서 획득했고, `Chemical_feature_generator.mendeleev_atomic_f`를 확인하면 어떠한 atomic feature를 선택했는지 확인할 수 있음
- `generate_mol_atomic_features` 함수를 통해서 하나의 분자에 해당하는 atomic feature matrix를 획득할 수 있음
- `get_adj_matrix` 함수를 통해서 분자의 adjacency matrix를 획득하고, 이를 다시 sparse tensor의 형태로 변형함
- atomic feature와 함께 사용하면 GCN의 forward 함수에 활용할 수 있음
    - *deepchem MPNN featurizer가 제공하는 것과 많은 부분이 유사
- `get_atomic_feature_from_deepchem`을 통해서 더 다양한 범위의 atomic feature를 획득할 수 있으나, 대회에서 제공해준 모든 데이터에 유효한 동작을 하는진 확인 필요
- `encoder_smiles` 함수는 chemBERT의 입력을 위해 smiles tokenizing을 하는 함수임

2. molecular features
- `get_molecular_features` 함수를 통해서 분자의 molecular property를 획득할 수 있고 `rdKit`을 이용함
    - 이렇게 획드간 feature와 대회에서 제공된 동일한 특성 값 사이에 차이가 있음 
- `get_molecule_fingerprints` 함수를 통해서 분자의 FPs를 획득할 수 있음.
- `get_mol_feature_from_deepchem` 함수를 통해서 rdkit에서 제공하는 molecular descriptor의 대부분 feature를 획득할 수 있음

코드 참고:
https://dacon.io/competitions/official/236127/codeshare/8812?page=1&dtype=recent

'''

from mendeleev.fetch import fetch_table
from rdkit import Chem
from rdkit.Chem import (Descriptors,
                        Lipinski,
                        Crippen,
                        rdMolDescriptors,
                        MolFromSmiles,
                        AllChem,
                        PandasTools)

from torch import Tensor
from torch_geometric.utils.sparse import dense_to_sparse

from transformers import AutoTokenizer, AutoModel

import numpy as np

from sklearn.preprocessing import StandardScaler
from rdkit import DataStructs
from deepchem import feat

archieve = ['seyonec/PubChem10M_SMILES_BPE_450k', "DeepChem/ChemBERTa-77M-MTR", 'seyonec/ChemBERTa-zinc-base-v1', 'seyonec/ChemBERTa_zinc250k_v2_40k']
chosen = archieve[2]

class Chemical_feature_generator():
    def __init__(self) -> None:
        mendeleev_atomic_f = ['atomic_radius', 'atomic_radius_rahm', 'atomic_volume', 'atomic_weight', 'c6', 'c6_gb', 
            'covalent_radius_cordero', 'covalent_radius_pyykko', 'covalent_radius_pyykko_double', 'covalent_radius_pyykko_triple', 
            'density', 'dipole_polarizability', 'dipole_polarizability_unc', 'electron_affinity', 'en_allen', 'en_ghosh', 'en_pauling', 
            'heat_of_formation', 'is_radioactive', 'molar_heat_capacity', 'specific_heat_capacity', 'vdw_radius']
            # Others are fine
            # Heat of Formation: This reflects the energy associated with the formation of a molecule and might indirectly impact metabolic reactions.
            # Is Radioactive: This binary property may not be directly relevant to metabolic stability.
            # Molar Heat Capacity, Specific Heat Capacity: These properties relate to heat transfer but might not be directly tied to metabolic stability.

        self.mendeleev_atomic_f_table = fetch_table("elements")[mendeleev_atomic_f]

        self.DMPNNFeatureizer = feat.DMPNNFeaturizer()
        self.Mol2VecFingerprint = feat.Mol2VecFingerprint()
        self.BPSymmetryFunctionInput = feat.BPSymmetryFunctionInput(max_atoms=150)

        self.tokenizer = AutoTokenizer.from_pretrained(chosen)

    
    def get_atomic_features(self, atom):
        atomic_num = atom.GetAtomicNum() - 1 # -1 is offset
        mendel_atom_f = self.mendeleev_atomic_f_table.loc[atomic_num]
        # mendel_atom_f.is_radioactive = mendel_atom_f.is_radioactive.astype(int)
        mendel_atom_f = mendel_atom_f.to_numpy().astype(np.float32)

        rdkit_atom_f = [atom.GetDegree(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetIsAromatic()*1.,
                atom.GetNumImplicitHs(),
                atom.GetNumExplicitHs(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                atom.GetImplicitValence(),
                atom.GetExplicitValence(),
                atom.GetTotalValence(),
                atom.IsInRing()*1.]
        
        return mendel_atom_f, rdkit_atom_f
    
    def get_molecular_features(self, mol):
        ## 1. Molecular Descriptors 5
        MolWt = Descriptors.MolWt(mol)
        HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
        NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
        MolMR = Crippen.MolMR(mol)
        MolLogP = Crippen.MolLogP(mol)


        ## 2. Lipinski's Rule of Five 16
        FractionCSP3 = Lipinski.FractionCSP3(mol)
        HeavyAtomCount = Lipinski.HeavyAtomCount(mol)
        NHOHCount = Lipinski.NHOHCount(mol)
        NOCount = Lipinski.NOCount(mol)
        NumAliphaticCarbocycles = Lipinski.NumAliphaticCarbocycles(mol)
        NumAliphaticHeterocycles = Lipinski.NumAliphaticHeterocycles(mol)
        NumAliphaticRings = Lipinski.NumAliphaticRings(mol)
        NumAromaticCarbocycles = Lipinski.NumAromaticCarbocycles(mol)
        NumAromaticHeterocycles = Lipinski.NumAromaticHeterocycles(mol)
        NumAromaticRings = Lipinski.NumAromaticRings(mol)
        NumHAcceptors = Lipinski.NumHAcceptors(mol)
        NumHDonors = Lipinski.NumHDonors(mol)
        NumHeteroatoms = Lipinski.NumHeteroatoms(mol)
        NumRotatableBonds = Lipinski.NumRotatableBonds(mol)
        RingCount = Lipinski.RingCount(mol)
        CalcNumBridgeheadAtom = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        ## 3. Additional Features 11
        ExactMolWt = Descriptors.ExactMolWt(mol)
        NumRadicalElectrons = Descriptors.NumRadicalElectrons(mol)
        # MaxPartialCharge = Descriptors.MaxPartialCharge(mol) 
        # MinPartialCharge = Descriptors.MinPartialCharge(mol) 
        # MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol) 
        # MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)  
        NumSaturatedCarbocycles = Lipinski.NumSaturatedCarbocycles(mol)
        NumSaturatedHeterocycles = Lipinski.NumSaturatedHeterocycles(mol)
        NumSaturatedRings = Lipinski.NumSaturatedRings(mol)
        CalcNumAmideBonds = rdMolDescriptors.CalcNumAmideBonds(mol)
        CalcNumSpiroAtoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)

        num_carboxyl_groups = len(mol.GetSubstructMatches(MolFromSmiles("[C](=O)[OH]"))) # "[C;X3](=O)[OH1]" not working
        num_amion_groups = len(mol.GetSubstructMatches(MolFromSmiles("[NH2]")))
        num_ammonium_groups = len(mol.GetSubstructMatches(MolFromSmiles("[NH4+]")))
        num_sulfonic_acid_groups = len(mol.GetSubstructMatches(MolFromSmiles("[S](=O)(=O)[O-]")))
        num_alkoxy_groups = len(mol.GetSubstructMatches(MolFromSmiles('CO'))) # "[*]-O-[*]" not working

        return [MolWt,
                HeavyAtomMolWt,
                NumValenceElectrons,
                FractionCSP3,
                HeavyAtomCount,
                NHOHCount,
                NOCount,
                NumAliphaticCarbocycles,
                NumAliphaticHeterocycles,
                NumAliphaticRings,
                NumAromaticCarbocycles,
                NumAromaticHeterocycles,
                NumAromaticRings,
                NumHAcceptors,
                NumHDonors,
                NumHeteroatoms,
                NumRotatableBonds,
                RingCount,
                MolMR,
                CalcNumBridgeheadAtom,
                ExactMolWt,
                NumRadicalElectrons,
                # MaxPartialCharge,
                # MinPartialCharge,
                # MaxAbsPartialCharge,
                # MinAbsPartialCharge,
                NumSaturatedCarbocycles,
                NumSaturatedHeterocycles,
                NumSaturatedRings,
                MolLogP,
                CalcNumAmideBonds,
                CalcNumSpiroAtoms,
                num_carboxyl_groups,
                num_amion_groups,
                num_ammonium_groups,
                num_sulfonic_acid_groups,
                num_alkoxy_groups]
    
    def get_molecule_fingerprints(self, mol):
        ECFP12 = AllChem.GetHashedMorganFingerprint(mol, 6, nBits=2048) # 2048
        ECFP6 = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=2048) # 2048
        
        MACCS = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol) # 167
        RDK_fp = Chem.RDKFingerprint(mol) # 2048
        Layer_fp = Chem.rdmolops.LayeredFingerprint(mol) # 2048
        Pattern_fp = Chem.rdmolops.PatternFingerprint(mol) # 2048
        
        ecfp12 = np.zeros((1,), dtype=np.int8)
        ecfp6 = np.zeros((1,), dtype=np.int8)
        maccs = np.zeros((1,), dtype=np.int8)
        rdk_fp = np.zeros((1,), dtype=np.int8)
        layer_fp = np.zeros((1,), dtype=np.int8)
        pattern_fp = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(ECFP12, ecfp12)
        DataStructs.ConvertToNumpyArray(ECFP6, ecfp6)
        DataStructs.ConvertToNumpyArray(MACCS, maccs)
        DataStructs.ConvertToNumpyArray(RDK_fp, rdk_fp)
        DataStructs.ConvertToNumpyArray(Layer_fp, layer_fp)
        DataStructs.ConvertToNumpyArray(Pattern_fp, pattern_fp)
        return np.hstack([ecfp12, ecfp6, maccs, rdk_fp, layer_fp, pattern_fp]) 
    
    def get_mol_feature_from_deepchem(self, smiles):
        return self.Mol2VecFingerprint(smiles) # (1, 300)
    
    def encoder_smiles(self, smiles):
        inputs = self.tokenizer.encode_plus(smiles, padding=True, return_tensors='pt', add_special_tokens=True)
        return inputs['input_ids']
    
    def get_atomic_feature_from_deepchem(self, smiles):
        DMPNN_F = self.DMPNNFeatureizer(smiles)[0].node_features

        return DMPNN_F
    
    def generate_mol_atomic_features(self, smiles):

        mol = Chem.MolFromSmiles(smiles)

        # gathering atomic feature
        mendel_atom_features = []
        rdkit_atom_features = []
        for atom in mol.GetAtoms():
            mendel_atom_f, rdkit_atom_f = self.get_atomic_features(atom)
            mendel_atom_features.append(mendel_atom_f)
            rdkit_atom_features.append(rdkit_atom_f)

        
        dc_atomic = self.get_atomic_feature_from_deepchem(smiles)
        atomic_features = np.concatenate([mendel_atom_features, rdkit_atom_features, dc_atomic], axis= 1, dtype=np.float32)

        return atomic_features
    

    def get_adj_matrix(self, smiles):

        mol = MolFromSmiles(smiles)
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)

        edge_index, edge_attr =dense_to_sparse(Tensor(adj))

        return edge_index, edge_attr
    
    



if __name__ == '__main__' : 
    import pandas as pd 
    from tqdm import tqdm
    from deepchem.feat.molecule_featurizers import RDKitDescriptors
    
    train = pd.read_csv('./input/train.csv')
    test  = pd.read_csv('./input/test.csv')
    
    generator = Chemical_feature_generator()
    
    def process(df):
        molecular_f = [] 
        for sample in tqdm(df.SMILES):
            sample = Chem.MolFromSmiles(sample)
            molecular_features = generator.get_molecular_features(mol=sample)
            molecular_f.append(molecular_features)
        
            # for fp, name in zip(fps, ['ECFP12','ECFP6','MACCS','RDK_fp','Layer_fp','Pattern_fp']):
            #     print(name, len(fp))
        molecular_f = np.concatenate([molecular_f], axis=0)
        # print(molecular_f.shape)
        
        return pd.DataFrame(data=molecular_f, columns=['MolWt','HeavyAtomMolWt','NumValenceElectrons','FractionCSP3','HeavyAtomCount','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds','RingCount','MolMR','CalcNumBridgeheadAtom','ExactMolWt','NumRadicalElectrons','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','MolLogP','CalcNumAmideBonds','CalcNumSpiroAtoms','num_carboxyl_groups','num_amion_groups','num_ammonium_groups','num_sulfonic_acid_groups','num_alkoxy_groups',])
                                                   #    'ECFP12','ECFP6','MACCS','RDK_fp','Layer_fp','Pattern_fp' ])
    

    print(train.iloc[1, :])
    print(test.columns)
    train_molecular_f = process(train)
    train_merged = pd.concat([train, train_molecular_f], axis=1)
    
    test_molecular_f = process(test)
    test_merged = pd.concat([test, test_molecular_f], axis=1)
    
    train_merged.to_csv('./input/new_train.csv', index=False)
    test_merged.to_csv('./input/new_test.csv', index=False)



    def deepchem_rdkit(df):
        
        featurizer = RDKitDescriptors()
        rdkit_features = []
        
        for smiles in tqdm(df.SMILES):
            feature = featurizer(smiles)
            rdkit_features.append(feature)
            
        return np.concatenate(rdkit_features)
    
    
    features = deepchem_rdkit(train)
    column_means = np.mean(features, axis=0)
    non_zero_mean_columns = np.where(column_means != 0)[0]
    features = features[:, non_zero_mean_columns]
    features = np.concatenate([train.SMILES.values.reshape(-1,1), features], axis=1)
    features = pd.DataFrame(features).dropna(axis=1)
    pd.DataFrame(features).to_csv('./input/rdkit_train.csv', index=False)
    train_col = features.columns


    features = deepchem_rdkit(test)
    features = features[:, non_zero_mean_columns]
    features = np.concatenate([test.SMILES.values.reshape(-1,1), features], axis=1)
    features = pd.DataFrame(features).dropna(axis=1)
    features = features[train_col]

    pd.DataFrame(features).to_csv('./input/rdkit_test.csv', index=False)
    test_col = features.columns
    
    print(list(train_col), len(train_col))
    print(list(test_col), len(test_col))
    
