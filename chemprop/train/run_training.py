from argparse import Namespace
import csv
from logging import Logger
import os
from pprint import pformat
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions, evaluate_regression
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint
import pandas as pd
import random

"""
Training
To train a model, run:
`chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>`

where <path> is the path to a CSV file containing a dataset, <type> is one of [classification, regression, multiclass, spectra] depending on the type of the dataset, and <dir> is the directory where model checkpoints will be saved.
"""

def run_training(args: Namespace, fold_num: int, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # # for reproducibility
    # =============================================================================
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.manual_seed(args.seed)
    # if device == 'cuda':
    #     torch.cuda.manual_seed_all(args.seed)
    # print(f"Torch Seed is {torch.seed()}")
    # =============================================================================

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Basic pre-processing data
    debug('basic pre-processing data')
    data = pd.read_csv(args.data_path)
    debug(f"Data shape: {data.shape}")

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    if args.features_generator is not None:
        print(f"FPs calculated by {len(args.features_generator)}: the size is {data.features_size()}")

    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    ## 여기까지 (8/26.., AlogP, Molecular_Weight 등의 다른 칼럼들을 feature에 넣을까 아님 별도로 관리해줄까?)
    
    # Split data
    debug(f'Splitting data with seed {args.seed}')
    test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path, logger=logger)
    train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.9, 0.1, 0.0), seed=args.seed, args=args, logger=logger)

    
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.features_scaling:
        debug("feature scaling...")
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)
    
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler: target')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    """ (9/10: target scaler 일단 뺴보자) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    scaler = None
    """
    ###################################################################################################################


    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    best_score_arr = []
    for model_idx in range(args.ensemble_size):
        debug(f"torch seed: {torch.seed()}")
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)
        # Load/build model
        if args.checkpoint_paths is not None:
            temp = list()
            temp.append(args.checkpoint_paths)

            debug(f'Loading model {model_idx} from {temp[model_idx]}')
            model = load_checkpoint(temp[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx+1}/{args.ensemble_size}')
            model = build_model(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch:3d} / {args.epochs}: learning rate is {scheduler.get_lr()[0]:.6f}')

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate_regression(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            if avg_val_score < best_score :
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)        

        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx}/{args.ensemble_size} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
        best_score_arr.append(best_score)
        
        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    outputCsv = pd.read_csv("./input/test.csv")
    outputCsv = outputCsv.drop(["SMILES","AlogP","Molecular_Weight","Num_H_Acceptors","Num_H_Donors","Num_RotatableBonds","LogD","Molecular_PolarSurfaceArea"], axis='columns')
    outputCsv["MLM"] = np.array(avg_test_preds)[:,0]
    outputCsv["HLM"] = np.array(avg_test_preds)[:,1]
    ensemble_scores = np.mean(best_score_arr)
    outputCsv.to_csv(f"./output/submission_score{ensemble_scores:.2f}_fold{fold_num}.csv", index=False)


    """
    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_{args.metric}', avg_ensemble_test_score, 0)

    avg_acc_test_score = np.nanmean(perfs_acc)
    avg_acc_test_score_std = np.std(perfs_acc)
    info(f'Ensemble test accuracy = {avg_acc_test_score:.6f} +- {avg_acc_test_score_std:.6f}')
    debug(f'Ensemble test accuracy = {avg_acc_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_accuracy', avg_acc_test_score, 0)

    avg_spec_test_score = np.nanmean(perfs_specificity)
    avg_spec_test_score_std = np.std(perfs_specificity)
    info(f'Ensemble test specificity = {avg_spec_test_score:.6f} +- {avg_spec_test_score_std:.6f}')
    debug(f'Ensemble test specificity = {avg_spec_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_specificity', avg_spec_test_score, 0)

    avg_recall_test_score = np.nanmean(perfs_recall)
    avg_recall_test_score_std = np.std(perfs_recall)
    info(f'Ensemble test recall = {avg_recall_test_score:.6f} +- {avg_recall_test_score_std:.6f}')
    debug(f'Ensemble test recall = {avg_recall_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_recall', avg_recall_test_score, 0)

    avg_f1_test_score = np.nanmean(perfs_f1)
    avg_f1_test_score_std = np.std(perfs_f1)
    info(f'Ensemble test F1 = {avg_f1_test_score:.6f} +- {avg_f1_test_score_std:.6f}')
    debug(f'Ensemble test F1 = {avg_f1_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_F1', avg_f1_test_score, 0)

    avg_auroc_test_score = np.nanmean(perfs_auroc)
    avg_auroc_test_score_std = np.std(perfs_auroc)
    info(f'Ensemble test AUROC = {avg_auroc_test_score:.6f} +- {avg_auroc_test_score_std:.6f}')
    writer.add_scalar(f'ensemble_test_AUROC', avg_auroc_test_score, 0)

    avg_auprc_test_score = np.nanmean(perfs_auprc)
    info(f'Ensemble test AUPRC = {avg_auprc_test_score:.6f}')
    writer.add_scalar(f'ensemble_test_AUPRC', avg_auprc_test_score, 0)
    """


    return ensemble_scores