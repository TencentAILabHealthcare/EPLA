#!/usr/bin/env python
# encoding: utf-8
"""
@time: 2020/9/22 10:27
@authors: Fan Yang
@copywriter: Tencent
"""

from __future__ import print_function, division
import os
import time
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--PALHI_tbl', default='PALHI_wsi_pred.csv')
parser.add_argument('--BOW_tbl', default='BOW_wsi_pred.csv')
parser.add_argument('--result_folder', default='./results')

args = parser.parse_args()

result_folder = args.result_folder
PALHI_csv = args.PALHI_tbl
BOW_csv = args.BOW_tbl
os.makedirs(result_folder, exist_ok=True)


if __name__ == "__main__":

    since = time.time()

    modelCSVs = [PALHI_csv, BOW_csv]
    model_pred_dfs = pd.read_csv(modelCSVs[0])

    model_pred_dfs['Sample.ID'] = model_pred_dfs['Sample.ID'].apply(lambda x: str(os.path.basename(x))[:12])

    for idx, predCSV in enumerate(modelCSVs):
        if idx > 0:
            model_pred_dfs = model_pred_dfs.merge(
                pd.read_csv(predCSV), how='outer', on=['Sample.ID'])

    colNames = list(model_pred_dfs)
    colOfIns = [x for x in colNames if x[:9] == 'WSI.Score']
    print(np.array(colOfIns))

    weights = [0.5, 0.5]
    youden_criterion = 0.5  # could be custom

    modelPredScores = model_pred_dfs[colOfIns].apply(
        lambda x: np.inner(x, np.array(weights)), axis=1)

    model_pred_dfs['WSI.Score'] = modelPredScores
    model_pred_dfs['WSI.pred'] = model_pred_dfs['WSI.Score'].apply(lambda x: 1 if x >= youden_criterion else 0)

    print(model_pred_dfs.head(10))
    model_pred_dfs.to_csv(os.path.join(
        result_folder, 'EPLA_output.csv'), encoding='utf-8', index=False)

    time_elapsed = time.time() - since
    print('Predicting complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
