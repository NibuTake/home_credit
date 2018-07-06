#-----------------
# Import packages.
#-----------------

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import collections
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import metrics, cross_validation

#-----------------
# Define sunctions.
#-----------------

def parse_params(best, listed):
    best = best.copy()
    if len(listed) == 0:
        return best
    else :
        for key in best:
            if key in listed.keys():
                best[key] = listed[key][int(best[key])]
        return best

def optimize_LGB(x_train, y_train, params, max_evals):
    def objective(params):
        params['num_leaves'] = int(params['num_leaves'])
        params['bagging_freq'] = int(params['bagging_freq'])
        params['max_depth'] = int(params['max_depth'])
        skf = cross_validation.StratifiedKFold(
            y_train, # Samples to split in K folds
            n_folds=5, # Number of folds. Must be at least 2.
            shuffle=True, # Whether to shuffle each stratification of the data before splitting into batches.
            random_state=423 # pseudo-random number generator state used for shuffling
        )
        boost_rounds = []
        score = []

        for train, test in skf:
            _train_x, _test_x, _train_y, _test_y = \
                x_train.iloc[train], x_train.iloc[test], y_train[train], y_train[test]

            train_lgb = lgb.Dataset(np.array(_train_x),np.array(_train_y))
            test_lgb = lgb.Dataset(np.array(_test_x),np.array(_test_y),reference=train_lgb)

            model = lgb.train(
                params,
                train_lgb,
                num_boost_round=10000,
                valid_sets=test_lgb,
                early_stopping_rounds=300
            )

            boost_rounds.append(model.best_iteration)
            score.append(model.best_score)
            #score.append(-verify_accuracy(binary_predict(model.predict(_test_x), 0.5), _test_y))

        # print('nb_trees={} val_loss={}'.format(boost_rounds, score))
        # print(len(score))
        mean_score = np.mean([list(score[k]['valid_0'].values())[0]
                              for k in range(len(score))])
        #mean_score = np.mean(score)

        # print('average of best iteration:', np.average(boost_rounds))
        return {'loss': mean_score, 'status': STATUS_OK}

    trials = Trials()
    # minimize the objective over the space
    best_params = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    print("Done")
    return {'best_params': best_params,
            'trials': trials}


#-----------------
# Read csv.
#-----------------

#username = 'takes'
username = 'morinibu'
data_directory = 'C://Users//' + username + '//GitHub//kaggle//data//home_credit'
main_directory = 'C://Users//' + username + '//GitHub//kaggle//home_credit'

data = pd.read_csv(main_directory + '//' + 'conbined_data.csv')

#-----------------
# Preprocess.
#-----------------

data_test = data[data.TARGET.isnull()].reset_index(drop=True)
data_train = data[[not(k) for k in data.TARGET.isnull()]].reset_index(drop=True)

ID = data_test.SK_ID_CURR
data_train.drop(columns=['SK_ID_CURR'], inplace=True)
data_test.drop(columns=['SK_ID_CURR'], inplace=True)

y = data_train.TARGET
X = data_train
X.drop(columns=['TARGET'], inplace=True)

y_submit = data_test.TARGET
X_submit = data_test
X_submit.drop(columns=['TARGET'], inplace=True)

#-----------------
# Main process.
#-----------------

evals = 2000
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                   stratify=y)
listed = {
    'boosting_type': ['dart'],
    "objective": ['binary'],
    "metric": ['auc']
}

params_LGB = {
    # 'task': 'train',
    #'boosting_type': hp.choice('boosting_type', listed['boosting_type']),
    'objective': hp.choice('objective', listed['objective']),
    'metric': hp.choice('metric', listed['metric']),
    'num_leaves': hp.quniform('num_leaves', 32,64,2),
    'max_depth': hp.quniform('max_depth', 3,12,1),
    'learning_rate': hp.loguniform('learning_rate', -6, -1),
    'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.85),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 0.85),
    # "feature_fraction_seed": 30,
    "subsample": hp.uniform("subsample", 0.5, 0.8),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 0.8),
    "lambda_l2": hp.uniform('lambda_l2', 5, 10),
    "lambda_l1": hp.uniform('lambda_l1', 5, 10),
    #"drop_rate": hp.uniform('drop_rate', 0.15, 0.4),
    'bagging_freq': hp.quniform('bagging_freq', 1,10,1),
    'min_split_gain': hp.uniform('min_split_gain', 0.001, 0.1),
    'min_child_weight': hp.uniform('min_child_weight', 10, 50),
    'silent' : -1,
    'verbose': -1
}

results_LGB = optimize_LGB(X, y, params_LGB, evals)

param = parse_params(results_LGB['best_params'], listed)
param['bagging_freq'] = int(param['bagging_freq'])
param['num_leaves'] = int(param['num_leaves'])
param['max_depth'] = int(param['max_depth'])

param_df = pd.DataFrame(list(param.items()),columns=['name','value'])
param_df.to_csv('result.csv', index=False)
