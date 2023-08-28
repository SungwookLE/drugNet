import logging
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
import numpy as np

from .predict import predict
from chemprop.data import MoleculeDataset, StandardScaler


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func: Callable,
                         dataset_type: str,
                         logger: logging.Logger = None) -> List[float]:
    """
    Evaluates predictions using a metric function and filtering out invalid targets.

    :param preds: A list of lists of shape (data_size, num_tasks) with model predictions.
    :param targets: A list of lists of shape (data_size, num_tasks) with targets.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param dataset_type: Dataset type.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    info = logger.info if logger is not None else print

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    # Filter out empty targets
    # valid_preds and valid_targets have shape (num_tasks, data_size)
    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    # Compute metric
    results = []
    perfs_acc = []
    perfs_specificity = []
    perfs_recall = []
    perfs_f1 = []
    perfs_auroc = []
    perfs_auprc = []
    for i in range(num_tasks):
        # # Skip if all targets or preds are identical, otherwise we'll crash during classification
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                info('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue
            
            if type(valid_preds[i]) == torch.Tensor:
                scores = valid_preds[i].cpu().detach().numpy()
            else:
                scores = np.array(valid_preds[i])
            if type(valid_targets[i]) == torch.Tensor:
                targets = valid_targets[i].cpu().detach().numpy()
            else:
                targets = np.array(valid_targets[i])
            
            fn = lambda x: 1 if x>0 else 0
            np_preds = np.array([fn(x) for x in scores])

            def np_sigmoid(x):
                return 1./(1. + np.exp(-x))
            
            #perfs_acc.append(accuracy_score(targets, np_preds))
            #perfs_precision.append(precision_score(targets, np_preds))
            #perfs_recall.append(recall_score(targets, np_preds))


            #hard_preds = [1 if p > 0.5 else 0 for p in np_preds]
            hard_preds = [1 if p > 0.5 else 0 for p in scores]
            
            conf_mat = confusion_matrix(targets, hard_preds)
            TN = conf_mat[0][0]
            FN = conf_mat[1][0]
            TP = conf_mat[1][1]
            FP = conf_mat[0][1]
            acc = float(TP+TN)/(TP+TN+FP+FN)
            #perfs_acc.append(accuracy_score(targets, hard_preds))
            perfs_acc.append(acc)
            mcc = float((TP*TN)-(FP*FN))/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
            PPV = float(TP)/(TP+FP)
            sensitivity = float(TP)/(TP+FN)
            specificity = float(TN)/(TN+FP)
            #print("acc, sensi, spec: ", acc, sensitivity, specificity)
            perfs_recall.append(sensitivity)
            perfs_specificity.append(specificity)
            perfs_f1.append(f1_score(targets, hard_preds))

            try:
                perfs_auroc.append(roc_auc_score(targets, np_sigmoid(scores)))
                perfs_auprc.append(average_precision_score(targets, np_sigmoid(scores)))
            except:
                perfs_auroc.append(0.)
                perfs_auprc.append(0.)

        if len(valid_targets[i]) == 0:
            continue

        if dataset_type == 'multiclass':
            results.append(metric_func(valid_targets[i], valid_preds[i], labels=list(range(len(valid_preds[i][0])))))
        else:
            results.append(metric_func(valid_targets[i], valid_preds[i]))
            #perfs_acc.append(accuracy_score(valid_targets[i], valid_preds[i]))

    return results, perfs_acc, perfs_specificity, perfs_recall, perfs_f1, perfs_auroc, perfs_auprc


def evaluate(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler
    )

    targets = data.targets()

    results, perfs_acc, perfs_precision, perfs_recall, perfs_f1, perfs_auroc, perfs_auprc = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
        logger=logger
    )
    return results, perfs_acc, perfs_precision, perfs_recall, perfs_f1, perfs_auroc, perfs_auprc


def evaluate_regression(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None,
             logger: logging.Logger = None) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param num_tasks: Number of tasks.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param batch_size: Batch size.
    :param dataset_type: Dataset type.
    :param scaler: A StandardScaler object fit on the training targets.
    :param logger: Logger.
    :return: A list with the score for each task based on `metric_func`.
    """
    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler
    )

    targets = data.targets()
    return metric_func(targets, preds)