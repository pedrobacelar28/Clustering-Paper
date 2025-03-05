Duas doenças 250 cada 1 lead

python scripts/run.py --dataset ecg --data_size 1 --gpu_enable
[info]run with options: {'dataset': 'ecg', 'n_splits': 5, 'nhidden': 8, 'nheads': 8, 'dropout': 0.2, 'relu': 0.2, 'data_size': 1, 'opt_metric': 'f1', 'niter': 1000, 'njobs': 10, 'batch_size': 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': False, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': False, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretrain': False, 'finetune': False}
Número de ecgs pra usar: 500
Número de ecgs que eram pra ser processados: 500
Número total de traçados processados: 500
(400, 4096, 1)
[info]original training shape (400, 4096, 1)
[info]basic statistics: max 17.6679, min -23.9936
[info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500, 'seg_length': 512, 'num_segment': 8, 'data_size': 1, 'warp': 2, 'tflag': True, 'gpu_enable': True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, 'sort': False, 'aggregate': True, 'n_features': 8, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2, 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standard_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=None, ...), 'ft_xgb': False, 'fm': <baselines.feature_based.FeatureModel object at 0x7fdce19fa730>, 'fm_scaler': MinMaxScaler(), 'feat_flag': False, 'feat_norm': False, 'pretrain': None, 'gat': GAT(
  (attention_0): GraphAttentionLayer (8 -> 8)
  (attention_1): GraphAttentionLayer (8 -> 8)
  (attention_2): GraphAttentionLayer (8 -> 8)
  (attention_3): GraphAttentionLayer (8 -> 8)
  (attention_4): GraphAttentionLayer (8 -> 8)
  (attention_5): GraphAttentionLayer (8 -> 8)
  (attention_6): GraphAttentionLayer (8 -> 8)
  (attention_7): GraphAttentionLayer (8 -> 8)
  (output): Sequential(
    (0): Linear(in_features=3200, out_features=6400, bias=True)
    (1): ReLU()
    (2): Linear(in_features=6400, out_features=2, bias=True)
  )
), 'lr': 0.001, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'debug': False, 'optimizer': 'Adam', 'measurement': 'gdtw', 'batch_size': 100, 'init': 0, 'niter': 1000, 'fastmode': False, 'tol': 0.0001}
[info]in this split: training 400 samples, with 204 positive
[info]train_size (400, 4096, 1), label size (400,)
[info]basic statistics before learn shapelets: max 17.6679, min -23.9936
[info]candidates with length 512 sampling done...00.00%
[info]totally 496 candidates with shape (496, 512, 1)
[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.002183
[info]training: 0.51 positive ratio with 400
[info]test: 0.46 positive ratio with 100
[info]fitting gat: accu 0.6575, prec 0.8317, recall 0.4118, f1 0.5508
[info]using default xgboost parameters
[info]fully-connected-layer res on training set: 0.6575, 0.8317, 0.4118, 0.5508
[info]out-classifier on training set: 1.0000, 1.0000, 1.0000, 1.0000
[info]res: accu 0.7400, prec 0.6667, recall 0.8696, f1 0.7547
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [23:28:38] WARNING: /workspace/src/c_api/c_api.cc:1374: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats.
  warnings.warn(smsg, UserWarning)


/////////////////////////////////////////////////////////
700 exames 1 lead 512

python scripts/run.py --dataset ecg --data_size 1 --gpu_enable
[info]run with options: {'dataset': 'ecg', 'n_splits': 5, 'nhidden': 8, 'nheads': 8, 'dropout': 0.2, 'relu': 0.2, 'data_size': 1, 'opt_metric': 'f1', 'niter': 1000, 'njobs': 10, 'batch_size': 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': False, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': False, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretrain': False, 'finetune': False}
Número de ecgs pra usar: 700
Número de ecgs que eram pra ser processados: 700
Número total de traçados processados: 700
(560, 4096, 1)
[info]original training shape (560, 4096, 1)
[info]basic statistics: max 16.6846, min -15.3332
[info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500, 'seg_length': 512, 'num_segment': 8, 'data_size': 1, 'warp': 2, 'tflag': True, 'gpu_enable': True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, 'sort': False, 'aggregate': True, 'n_features': 8, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2, 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standard_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=None, ...), 'ft_xgb': False, 'fm': <baselines.feature_based.FeatureModel object at 0x7fa942437d30>, 'fm_scaler': MinMaxScaler(), 'feat_flag': False, 'feat_norm': False, 'pretrain': None, 'gat': GAT(
  (attention_0): GraphAttentionLayer (8 -> 8)
  (attention_1): GraphAttentionLayer (8 -> 8)
  (attention_2): GraphAttentionLayer (8 -> 8)
  (attention_3): GraphAttentionLayer (8 -> 8)
  (attention_4): GraphAttentionLayer (8 -> 8)
  (attention_5): GraphAttentionLayer (8 -> 8)
  (attention_6): GraphAttentionLayer (8 -> 8)
  (attention_7): GraphAttentionLayer (8 -> 8)
  (output): Sequential(
    (0): Linear(in_features=3200, out_features=6400, bias=True)
    (1): ReLU()
    (2): Linear(in_features=6400, out_features=2, bias=True)
  )
), 'lr': 0.001, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'debug': False, 'optimizer': 'Adam', 'measurement': 'gdtw', 'batch_size': 100, 'init': 0, 'niter': 1000, 'fastmode': False, 'tol': 0.0001}
[info]in this split: training 560 samples, with 82 positive
[info]train_size (560, 4096, 1), label size (560,)
[info]basic statistics before learn shapelets: max 16.6846, min -15.3332
[info]candidates with length 512 sampling done...00.00%
[info]totally 496 candidates with shape (496, 512, 1)
[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.0022152207
[info]training: 0.15 positive ratio with 560
[info]test: 0.13 positive ratio with 140
[info]fitting gat: accu 0.8339, prec 0.3514, recall 0.1585, f1 0.2185
[info]using default xgboost parameters
[info]fully-connected-layer res on training set: 0.8339, 0.3514, 0.1585, 0.2185
[info]out-classifier on training set: 1.0000, 1.0000, 1.0000, 1.0000
[info]res: accu 0.8571, prec 0.4000, recall 0.2222, f1 0.2857
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [00:19:17] WARNING: /workspace/src/c_api/c_api.cc:1374: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats.
  warnings.warn(smsg, UserWarning)

////////////////////////////////
700 exames 12 leads 4096

python scripts/run.py --dataset ecg --data_size 12 --gpu_enable
[info]run with options: {'dataset': 'ecg', 'n_splits': 5, 'nhidden': 8, 'nheads': 8, 'dropout': 0.2, 'relu': 0.2, 'data_size': 12, 'opt_metric': 'f1', 'niter': 1000, 'njobs': 10, 'batch_size': 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': False, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': False, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretrain': False, 'finetune': False}
Número de ecgs pra usar: 700
Número de ecgs que eram pra ser processados: 700
Número total de traçados processados: 700
(700, 4096, 12)
(560, 4096, 12)
[info]original training shape (560, 4096, 12)
[info]basic statistics: max 40.1520, min -36.9318
[info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500, 'seg_length': 4096, 'num_segment': 1, 'data_size': 12, 'warp': 2, 'tflag': True, 'gpu_enable': True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, 'sort': False, 'aggregate': True, 'n_features': 1, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2, 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standard_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=None, ...), 'ft_xgb': False, 'fm': <baselines.feature_based.FeatureModel object at 0x7fd87fa5e0d0>, 'fm_scaler': MinMaxScaler(), 'feat_flag': False, 'feat_norm': False, 'pretrain': None, 'gat': GAT(
  (attention_0): GraphAttentionLayer (1 -> 8)
  (attention_1): GraphAttentionLayer (1 -> 8)
  (attention_2): GraphAttentionLayer (1 -> 8)
  (attention_3): GraphAttentionLayer (1 -> 8)
  (attention_4): GraphAttentionLayer (1 -> 8)
  (attention_5): GraphAttentionLayer (1 -> 8)
  (attention_6): GraphAttentionLayer (1 -> 8)
  (attention_7): GraphAttentionLayer (1 -> 8)
  (output): Sequential(
    (0): Linear(in_features=3200, out_features=6400, bias=True)
    (1): ReLU()
    (2): Linear(in_features=6400, out_features=2, bias=True)
  )
), 'lr': 0.001, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'debug': False, 'optimizer': 'Adam', 'measurement': 'gdtw', 'batch_size': 100, 'init': 0, 'niter': 1000, 'fastmode': False, 'tol': 0.0001}
[info]in this split: training 560 samples, with 82 positive
[info]train_size (560, 4096, 12), label size (560,)
[info]basic statistics before learn shapelets: max 40.1520, min -36.9318
[info]candidates with length 4096 sampling done...0.00%
[info]totally 500 candidates with shape (500, 4096, 12)
[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.049803
[info]training: 0.15 positive ratio with 560
[info]test: 0.13 positive ratio with 140
	/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
[info]fitting gat: accu 0.8536, prec 0.0000, recall 0.0000, f1 0.0000
[info]using default xgboost parameters
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
[info]fully-connected-layer res on training set: 0.8536, 0.0000, 0.0000, 0.0000
[info]out-classifier on training set: 0.9232, 0.9535, 0.5000, 0.6560
[info]res: accu 0.8429, prec 0.0000, recall 0.0000, f1 0.0000
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [19:10:17] WARNING: /workspace/src/c_api/c_api.cc:1374: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats.
  warnings.warn(smsg, UserWarning)

/////////////
700 exames 12 leads 1024

python scripts/run.py --dataset ecg --data_size 12 --gpu_enable
[info]run with options: {'dataset': 'ecg', 'n_splits': 5, 'nhidden': 8, 'nheads': 8, 'dropout': 0.2, 'relu': 0.2, 'data_size': 12, 'opt_metric': 'f1', 'niter': 1000, 'njobs': 10, 'batch_size': 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': False, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': False, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretrain': False, 'finetune': False}
Número de ecgs pra usar: 700
Número de ecgs que eram pra ser processados: 700
Número total de traçados processados: 700
(560, 4096, 12)
[info]original training shape (560, 4096, 12)
[info]basic statistics: max 40.1520, min -36.9318
[info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500, 'seg_length': 1024, 'num_segment': 4, 'data_size': 12, 'warp': 2, 'tflag': True, 'gpu_enable': True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, 'sort': False, 'aggregate': True, 'n_features': 4, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2, 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standard_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric='logloss',
              feature_types=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=None, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=None, ...), 'ft_xgb': False, 'fm': <baselines.feature_based.FeatureModel object at 0x7f8f7f25ed30>, 'fm_scaler': MinMaxScaler(), 'feat_flag': False, 'feat_norm': False, 'pretrain': None, 'gat': GAT(
  (attention_0): GraphAttentionLayer (4 -> 8)
  (attention_1): GraphAttentionLayer (4 -> 8)
  (attention_2): GraphAttentionLayer (4 -> 8)
  (attention_3): GraphAttentionLayer (4 -> 8)
  (attention_4): GraphAttentionLayer (4 -> 8)
  (attention_5): GraphAttentionLayer (4 -> 8)
  (attention_6): GraphAttentionLayer (4 -> 8)
  (attention_7): GraphAttentionLayer (4 -> 8)
  (output): Sequential(
    (0): Linear(in_features=3200, out_features=6400, bias=True)
    (1): ReLU()
    (2): Linear(in_features=6400, out_features=2, bias=True)
  )
), 'lr': 0.001, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'debug': False, 'optimizer': 'Adam', 'measurement': 'gdtw', 'batch_size': 100, 'init': 0, 'niter': 1000, 'fastmode': False, 'tol': 0.0001}
[info]in this split: training 560 samples, with 82 positive
[info]train_size (560, 4096, 12), label size (560,)
[info]basic statistics before learn shapelets: max 40.1520, min -36.9318
[info]candidates with length 1024 sampling done...0.00%
[info]totally 500 candidates with shape (500, 1024, 12)
[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.006250
[info]training: 0.15 positive ratio with 560
[info]test: 0.13 positive ratio with 140
[info]fitting gat: accu 0.8571, prec 1.0000, recall 0.0244, f1 0.0476
[info]using default xgboost parameters
[info]fully-connected-layer res on training set: 0.8571, 1.0000, 0.0244, 0.0476
[info]out-classifier on training set: 1.0000, 1.0000, 1.0000, 1.0000
[info]res: accu 0.8714, prec 0.5000, recall 0.1667, f1 0.2500
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [19:47:36] WARNING: /workspace/src/c_api/c_api.cc:1374: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats.
  warnings.warn(smsg, UserWarning)

/////////////////
7000 exames 12 leads 1024 pontos por segmento


python scripts/run.py --dataset ecg --data_size 12 --gpu_enable                                                                          [info]run with options: {'dataset': 'ecg', 'n_splits': 5, 'nhidden': 8, 'nheads': 8, 'dropout': 0.2, 'relu': 0.2, 'data_size': 12, 'opt_metric': 'f1', 'niter': 1000, 'njobs': 10, 'batch_size': 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': False, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': False, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretrain': False, 'finetune': False}                                                                                                 Número de ecgs pra usar: 7000                                                                                           Número de ecgs que eram pra ser processados: 7000                                                                       Número total de traçados processados: 7000                                                                              (5600, 4096, 12)                                                                                                        [info]original training shape (5600, 4096, 12)                                                                          [info]basic statistics: max 219.0648, min -44.9056                                                                      [info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500, 'seg_length': 1024, 'num_segment': 4, 'data_size': 12, 'warp': 2, 'tflag': True, 'gpu_enable': True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, 'sort': False, 'aggregate': True, 'n_features': 4, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2, 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standard_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callbacks=None,                                                                 colsample_bylevel=None, colsample_bynode=None,                                                                          colsample_bytree=None, device=None, early_stopping_rounds=None,                                                         enable_categorical=False, eval_metric='logloss',                                                                        feature_types=None, gamma=None, grow_policy=None,                                                                       importance_type=None, interaction_constraints=None,                                                                     learning_rate=None, max_bin=None, max_cat_threshold=None,                                                               max_cat_to_onehot=None, max_delta_step=None, max_depth=None,                                                            max_leaves=None, min_child_weight=None, missing=nan,

 and penalty 0.00[debug]91.07% steps, loss -0.176439 with -0.182583 and penalty 0.00[debug[debug]50.00% steps, loss -0.4[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.006261080.006858 and penalty 0.006263                  [info]training: 0.14 positive ratio with 5600                                                                           [info]test: 0.15 positive ratio with 1400                                                                               [info]fitting gat: accu 0.8593, prec 0.6000, recall 0.0038, f1 0.0076                                                   [info]using default xgboost parameters                                                                                  [info]fully-connected-layer res on training set: 0.8593, 0.6000, 0.0038, 0.0076                                         [info]out-classifier on training set: 0.9979, 1.0000, 0.9848, 0.9923                                                    [info]res: accu 0.8514, prec 0.5224, recall 0.1659, f1 0.2518                                                           /home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [02:45:39] WARNING: /workspace/src/c_api/c_api.cc:1374: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats.                                                             warnings.warn(smsg, UserWarning) 

////////////

700 exames 12 leads 512

python scripts/run.py --dataset ecg --data_size 12 --gpu_enable --n_splits 2         
[info]run with options: {'dataset': 'ecg', 'n_splits': 2, 'nhidden': 8, 'nheads': 8, 'dropout'
: 0.2, 'relu': 0.2, 'data_size': 12, 'opt_metric': 'f1', 'niter': 50, 'njobs': 10, 'batch_size
': 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': 
False, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': Fa
lse, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretr
ain': False, 'finetune': False}                                                               
Número de ecgs pra usar: 700                                                                  
Número de ecgs que eram pra ser processados: 700                                              
Número total de traçados processados: 700                                                     
(560, 4096, 12)                                                                               
[info]original training shape (560, 4096, 12)                                                 
[info]basic statistics: max 40.1520, min -36.9318                                             
[info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': 
False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500,
 'seg_length': 512, 'num_segment': 8, 'data_size': 12, 'warp': 2, 'tflag': True, 'gpu_enable':
 True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, 
'sort': False, 'aggregate': True, 'n_features': 8, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2
, 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standar
d_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callb
acks=None, 
              colsample_bytree=None, device=None, early_stopping_rounds=None,         [19/529]
              enable_categorical=False, eval_metric='logloss',                                
              feature_types=None, gamma=None, grow_policy=None,                               
              importance_type=None, interaction_constraints=None,                             
              learning_rate=None, max_bin=None, max_cat_threshold=None,                       
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,                    
              max_leaves=None, min_child_weight=None, missing=nan,                            
              monotone_constraints=None, multi_strategy=None, n_estimators=None,              
              n_jobs=None, num_parallel_tree=None, random_state=None, ...), 'ft_xgb': False, '
fm': <baselines.feature_based.FeatureModel object at 0x7ff3564fdd90>, 'fm_scaler': MinMaxScale
r(), 'feat_flag': False, 'feat_norm': False, 'pretrain': None, 'gat': GAT(                    
  (attention_0): GraphAttentionLayer (8 -> 8)                                                 
  (attention_1): GraphAttentionLayer (8 -> 8)                                                 
  (attention_2): GraphAttentionLayer (8 -> 8)                                                 
  (attention_3): GraphAttentionLayer (8 -> 8)                                                 
  (attention_4): GraphAttentionLayer (8 -> 8)                                                 
  (attention_5): GraphAttentionLayer (8 -> 8)                                                 
  (attention_6): GraphAttentionLayer (8 -> 8)                                                 
  (attention_7): GraphAttentionLayer (8 -> 8)                                                 
  (output): Sequential(                                                                       
    (0): Linear(in_features=3200, out_features=6400, bias=True)                               
    (1): ReLU()        
    (2): Linear(in_features=6400, out_features=2, bias=True)                                  
  )                    
), 'lr': 0.001, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'debug': False, 'optimizer': 'Adam', 'measurement': 'gdtw', 'batch_size': 100, 'init': 0, 'niter': 50, 'fastmode': False, 'tol': 0.0001}[info]in this split: training 560 samples, with 82 positive                                   
[info]train_size (560, 4096, 12), label size (560,)                                           
[info]basic statistics before learn shapelets: max 40.1520, min -36.9318                      
[info]candidates with length 512 sampling done...00.00%                                       
[info]totally 496 candidates with shape (496, 512, 12)                                        
^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A^[[A[debug]0[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.002225
[info]training: 0.15 positive ratio with 560                                                  
[info]test: 0.13 positive ratio with 140                                                      
^[[A^[[A^[[A^[[A^[[A^[[A^[[info]fitting gat: accu 0.8554, prec 0.6000, recall 0.0366, f1 0.0690
[info]using default xgboost parameters         
[info]fully-connected-layer res on training set: 0.8554, 0.6000, 0.0366, 0.0690
[info]out-classifier on training set: 1.0000, 1.0000, 1.0000, 1.0000                          
[info]res: accu 0.8571, prec 0.3333, recall 0.1111, f1 0.1667

//////////

700 exames 1 lead 1024

[info]run with options: {'dataset': 'ecg', 'n_splits': 5, 'nhidden': 8, 'nheads': 8, 'dropout'
: 0.2, 'relu': 0.2, 'data_size': 1, 'opt_metric': 'f1', 'niter': 50, 'njobs': 10, 'batch_size'
: 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': F
alse, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': Fal
se, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretra
in': False, 'finetune': False}                                                                
Número de ecgs pra usar: 700                                                                  
Número de ecgs que eram pra ser processados: 700                                              
Número total de traçados processados: 700                                                     
(560, 4096, 1)                                                                                
[info]original training shape (560, 4096, 1)                                                  
[info]basic statistics: max 16.6846, min -15.3332                                             
[info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': 
False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500,
 'seg_length': 1024, 'num_segment': 4, 'data_size': 1, 'warp': 2, 'tflag': True, 'gpu_enable':
 True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, 
'sort': False, 'aggregate': True, 'n_features': 4, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2
, 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standar
d_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callb
acks=None,                                                                                    
              colsample_bylevel=None, colsample_bynode=None,                                  
              colsample_bytree=None, device=None, early_stopping_rounds=None,                 
              enable_categorical=False, eval_metric='logloss',                                
              feature_types=None, gamma=None, grow_policy=None,                               
              importance_type=None, interaction_constraints=None,                             
              learning_rate=None, max_bin=None, max_cat_threshold=None,                       
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=None, ...), 'ft_xgb': False, '
fm': <baselines.feature_based.FeatureModel object at 0x7fa35baf74c0>, 'fm_scaler': MinMaxScale
r(), 'feat_flag': False, 'feat_norm': False, 'pretrain': None, 'gat': GAT(
  (attention_0): GraphAttentionLayer (4 -> 8)
  (attention_1): GraphAttentionLayer (4 -> 8)
  (attention_2): GraphAttentionLayer (4 -> 8)
  (attention_3): GraphAttentionLayer (4 -> 8)
  (attention_4): GraphAttentionLayer (4 -> 8)
  (attention_5): GraphAttentionLayer (4 -> 8)

), 'lr': 0.001, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'debug': False, 'optimizer': 'Adam', 'measurement': 'gdtw', 'batch_size': 100, 'init': 0, 'niter': 50, 'fastmode': False, 'tol': 0.0001}
[info]in this split: training 560 samples, with 82 positive
[info]train_size (560, 4096, 1), label size (560,)
[info]basic statistics before learn shapelets: max 16.6846, min -15.3332
[info]candidates with length 1024 sampling done...0.00%
[info]totally 500 candidates with shape (500, 1024, 1)
[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.006255
[info]training: 0.15 positive ratio with 560
[info]test: 0.13 positive ratio with 140
[info]fitting gat: accu 0.8518, prec 0.4000, recall 0.0244, f1 0.0460
[info]using default xgboost parameters
[info]fully-connected-layer res on training set: 0.8518, 0.4000, 0.0244, 0.0460
[info]out-classifier on training set: 1.0000, 1.0000, 1.0000, 1.0000
[info]res: accu 0.8357, prec 0.0000, recall 0.0000, f1 0.0000
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [18:54:27] WARNING: /workspace/src/c_api/c_api.cc:1374: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats.
  warnings.warn(smsg, UserWarning)

///////////
700 exames 1 lead 512

python scripts/run.py --dataset ecg --data_size 1 --gpu_enable                       
[info]run with options: {'dataset': 'ecg', 'n_splits': 5, 'nhidden': 8, 'nheads': 8, 'dropout'
: 0.2, 'relu': 0.2, 'data_size': 1, 'opt_metric': 'f1', 'niter': 50, 'njobs': 10, 'batch_size'
: 100, 'percentile': 80, 'diff': False, 'standard_scale': False, 'softmax': False, 'append': F
alse, 'sort': False, 'ft_xgb': False, 'aggregate': False, 'feat_flag': False, 'feat_norm': Fal
se, 'debug': False, 'model_cache': False, 'shapelet_cache': False, 'gpu_enable': True, 'pretra
in': False, 'finetune': False}                                                                
Número de ecgs pra usar: 700                                                                  
Número de ecgs que eram pra ser processados: 700                                              
Número total de traçados processados: 700                                                     
(560, 4096, 1)                                                                                
[info]original training shape (560, 4096, 1)                                                  
[info]basic statistics: max 16.6846, min -15.3332                                             
[info]initialize time2graph+ model with {'kernel': 'xgb', 'kwargs': {'njobs': 10, 'ggregate': 
False, 'kernel': 'xgb', 'mode': 'embedding', 'candidate_method': 'greedy'}, 'K': 50, 'C': 500,
 'seg_length': 512, 'num_segment': 8, 'data_size': 1, 'warp': 2, 'tflag': True, 'gpu_enable': 
True, 'cuda': True, 'shapelets': None, 'append': False, 'percentile': 80, 'threshold': None, '
sort': False, 'aggregate': True, 'n_features': 8, 'n_hidden': 8, 'n_heads': 8, 'dropout': 0.2,
 'lk_relu': 0.2, 'out_clf': True, 'softmax': False, 'dataset': 'ecg', 'diff': False, 'standard
_scale': False, 'opt_metric': 'f1', 'xgb': XGBClassifier(base_score=None, booster=None, callba
cks=None,                                                                                     
              colsample_bylevel=None, colsample_bynode=None,                                  
              colsample_bytree=None, device=None, early_stopping_rounds=None,                 
              enable_categorical=False, eval_metric='logloss',                                
              feature_types=None, gamma=None, grow_policy=None,                               
              importance_type=None, interaction_constraints=None,                             
              learning_rate=None, max_bin=None, max_cat_threshold=None,                [0/679]
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, random_state=None, ...), 'ft_xgb': False, 'fm': <baselines.feature_based.FeatureModel object at 0x7fdecc7b8310>, 'fm_scaler': MinMaxScaler(), 'feat_flag': False, 'feat_norm': False, 'pretrain': None, 'gat': GAT(
  (attention_0): GraphAttentionLayer (8 -> 8)
  (attention_1): GraphAttentionLayer (8 -> 8)
  (attention_2): GraphAttentionLayer (8 -> 8)
  (attention_3): GraphAttentionLayer (8 -> 8)
  (attention_4): GraphAttentionLayer (8 -> 8)
  (attention_5): GraphAttentionLayer (8 -> 8)
  (attention_6): GraphAttentionLayer (8 -> 8)
  (attention_7): GraphAttentionLayer (8 -> 8)
  (output): Sequential( 
    (0): Linear(in_features=3200, out_features=6400, bias=True)
    (1): ReLU()
    (2): Linear(in_features=6400, out_features=2, bias=True)
  )
), 'lr': 0.001, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'debug': False, 'optimizer': 'Adam', 'measurement': 'gdtw', 'batch_size': 100, 'init': 0, 'niter': 50, 'fastmode': False, 'tol': 0.0001}
[info]in this split: training 560 samples, with 82 positive
[info]train_size (560, 4096, 1), label size (560,)
[info]basic statistics before learn shapelets: max 16.6846, min -15.3332
[info]candidates with length 512 sampling done...00.00%
[info]totally 496 candidates with shape (496, 512, 1)
[info]learn time_aware shapelets done...cuted by 100.00%enalty 0.002208
[info]training: 0.15 positive ratio with 560
[info]test: 0.13 positive ratio with 140
[info]fitting gat: accu 0.8554, prec 1.0000, recall 0.0122, f1 0.0241
[info]using default xgboost parameters
[info]fully-connected-layer res on training set: 0.8554, 1.0000, 0.0122, 0.0241
[info]out-classifier on training set: 1.0000, 1.0000, 1.0000, 1.0000
[info]res: accu 0.8429, prec 0.2500, recall 0.1111, f1 0.1538
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [19:33:36] WARNING: /workspace/src/c_api/c_api.cc:1374: Saving model in the UBJSON format as default.  You can use file extension: `json`, `ubj` or `deprecated` to choose between formats.
  warnings.warn(smsg, UserWarning)