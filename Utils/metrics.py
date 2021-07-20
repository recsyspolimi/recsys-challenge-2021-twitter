import dask
import dask.dataframe as dd
import gzip
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


def quantized_rce(quantiles, pred, gt,doMean=False):
    df = pd.DataFrame(
        {
            'quantiles': quantiles,
            'pred': pred,
            'gt': gt
        }
    )
    if doMean:
        return df.groupby('quantiles').apply(lambda x: compute_rce(x['pred'], x['gt'])).mean()
    return df.groupby('quantiles').apply(lambda x: compute_rce(x['pred'], x['gt']))
    


def quantized_average_precision(quantiles, pred, gt,doMean=False):
    df = pd.DataFrame(
        {
            'quantiles': quantiles,
            'pred': pred,
            'gt': gt
        }
    )
    if doMean:
        return df.groupby('quantiles').apply(lambda x: average_precision_score(x['gt'], x['pred'])).mean()
    return df.groupby('quantiles').apply(lambda x: average_precision_score(x['gt'], x['pred']))
