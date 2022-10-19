from argparse import ArgumentParser
import os
import torch
import numpy as np
import pandas as pd

from chemprop.parsing import modify_predict_args
from chemprop.train import make_predictions
from chemprop.features import get_available_features_generators

if __name__ == '__main__':
    # Predict arguments
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made',
                        default='../input/test.csv')
    parser.add_argument('--preds_path', type=str,
                        help='Path to CSV file where predictions will be saved',
                        default='test_pred')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)',
                        default='./ckpt')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')

    args = parser.parse_args()
    args.checkpoint_dir = 'predps_model'
    modify_predict_args(args)

    output_path = args.preds_path

    df = pd.read_csv(args.test_path)
    pred, smiles = make_predictions(args, df.smiles.tolist())
    df = pd.DataFrame({'smiles':smiles})
    predps_lst = []
    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]
        for item in pred:
            if item[i] >= 0.5:
                predps_lst.append("unstable")
            else:
                predps_lst.append("stable")
    df[f'predps_results_{i}'] = predps_lst
    df.to_csv(output_path, index=False)

