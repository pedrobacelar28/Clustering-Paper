ython scripts/run.py --dataset ecg --data_size 1 --gpu_enable --seg_length 512 --num_segment 8 --K 50 --C 500
[info]running with {'dataset': 'ecg', 'K': 50, 'C': 500, 'n_splits': 5, 'num_segment': 8, 'seg_length': 512, 'njobs': 8, 'data_size': 1, 'optimizer': 'Adam', 'alpha': 0.1, 'beta': 0.05, 'init': 0, 'gpu_enable': True, 'opt_metric': 'accuracy', 'cache': False, 'embed': 'aggregate', 'embed_size': 256, 'warp': 2, 'cmethod': 'greedy', 'kernel': 'xgb', 'percentile': 10, 'measurement': 'gdtw', 'batch_size': 50, 'tflag': True, 'scaled': False, 'norm': False, 'no_global': True}
Número de ecgs pra usar: 700
Número de ecgs que eram pra ser processados: 700
Número total de traçados processados: 700
(560, 4096, 1)
[info]training: 0.15 positive ratio with 560
[info]test: 0.13 positive ratio with 140
[info]initialize t2g model with {'kernel': 'xgb', 'kwargs': {'candidate_method': 'greedy', 'njobs': 8, 'optimizer': 'Adam', 'representation_size': 256}, 'K': 50, 'C': 500, 'seg_length': 512, 'warp': 2, 'tflag': True, 'opt_metric': 'accuracy', 'mode': 'aggregate', 'batch_size': 50, 'gpu_enable': True, 'percentile': 10, 'shapelets': None, 'sembeds': None, 'clf': None, 'lr': 0.01, 'p': 2, 'alpha': 0.1, 'beta': 0.05, 'multi_graph': False, 'debug': True, 'measurement': 'gdtw', 'verbose': False, 'global_flag': True}
[info]training ecg_mixed_model ...
[info]training embedding model...
[info]initialize ShapeletEmbedding model with ops: {'seg_length': 512, 'tflag': True, 'multi_graph': False, 'cache_dir': '/scratch/caio.grossi/Clustering-Paper/SBCAS/caiosgrossi/Time2Graph/scripts/cache/ecg/', 'tanh': False, 'debug': True, 'percentile': 10, 'dist_threshold': -1, 'measurement': 'gdtw', 'mode': 'aggregate', 'global_flag': True, 'deepwalk_args': {'candidate_method': 'greedy', 'njobs': 8, 'optimizer': 'Adam', 'representation_size': 256}, 'embed_size': 256, 'embeddings': None}
[info]fit shape: (478, 4096, 1)
[info]threshold(10) 0.031490332633256915, mean 0.7661189436912537
[info]2159 edges involved in shapelets graph
[info]embed_size: 256
[info]transition matrix size (1, 50, 50)
[info]run deepwalk with: deepwalk --input /scratch/caio.grossi/Clustering-Paper/SBCAS/caiosgrossi/Time2Graph/scripts/cache/ecg/0.edgelist --format edgelist --output /scratch/caio.grossi/Clustering-Paper/SBCAS/caiosgrossi/Time2Graph/scripts/cache/ecg/0.embeddings --representation-size 256
[info]embedding threshold 0.031490332633256915
[info]sdist size (560, 8, 50)
[info]extract mixed features done...d by 100.00%
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:42] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:42] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:43] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:39:44] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:38] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:38] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:38] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:38] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:38] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:38] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:39] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:40] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:40] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:41:40] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:43:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:20] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:45:21] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:47:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:31] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:49:32] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:51:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:48] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:53:49] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:09] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:09] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:56:09] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [15:58:27] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:01:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:28] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:29] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:04:30] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:07:19] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:34] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:35] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:10:36] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:23] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:13:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:06] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:06] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:06] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:06] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:06] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:07] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:16:08] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:24] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:25] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:19:26] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:16] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:17] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
/home/grad/si/23/caio.grossi/anaconda3/envs/t2g_py9/lib/python3.9/site-packages/xgboost/core.py:158: UserWarning: [16:22:18] WARNING: /workspace/src/learner.cc:740: 
Parameters: { "max_depth" } are not used.

  warnings.warn(smsg, UserWarning)
Modelo treinado
[info]embedding threshold 0.031490332633256915
[info]sdist size (140, 8, 50)
[info]result: accu 0.8643, prec 0.0000, recall 0.0000, f1 0.0000