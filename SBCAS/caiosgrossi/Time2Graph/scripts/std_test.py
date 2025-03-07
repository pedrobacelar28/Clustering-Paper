# -*- coding: utf-8 -*-
"""
    test scripts on three benchmark datasets: EQS, WTC, STB
"""
import argparse
import warnings
import os
from config import *
from archive.load_usr_dataset import load_usr_dataset_by_name
from time2graph.utils.base_utils import Debugger
from time2graph.core.model import Time2Graph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataloader import carregar_ecgs
import pandas as pd
import numpy as np



if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Earthquakes')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--model_cache', action='store_true', default=False)
    parser.add_argument('--shapelet_cache', action='store_true', default=False)
    parser.add_argument('--gpu_enable', action='store_true', default=False)
    args = parser.parse_args()
    Debugger.info_print('running with {}'.format(args.__dict__))

    # set default options
    general_options = {
        'kernel': 'xgb',
        'opt_metric': 'accuracy',
        'init': 0,
        'warp': 2,
        'tflag': True,
        'mode': 'embedding',
        'candidate_method': 'greedy'
    }
    model_options = model_args[args.dataset]
    xgb_options = xgb_args[args.dataset]

    # load benchmark dataset
    if args.dataset.startswith('ucr'):
        dataset = args.dataset.rstrip('\n\r').split('-')[-1]
        x_train, y_train, x_test, y_test = load_usr_dataset_by_name(
            fname=dataset, length=model_options['seg_length'] * model_options['num_segment'])
    elif args.dataset.startswith('ecg'):
        print("a")
        # Chama sua função para carregar os ECGs.
        # Aqui, por exemplo, você pode definir os parâmetros de quantidade desejada para cada rótulo.
        X, ids_ecgs = carregar_ecgs(
            unlabel=100,
            umdavb=0,
            rbbb=0,
            lbbb=0,
            sb=0,
            st=100,
            af=0,
            filtrado=True
        )
        
        # Supondo que cada traçado seja um array de shape (L, 12) e que X é um numpy.ndarray com shape (N, L, 12)
        # Se necessário, realize pré-processamento adicional, por exemplo, segmentação em batimentos.
        # Se você deseja segmentar cada ECG em batimentos, esse processamento pode ser feito aqui.
        #
        # Exemplo de pré-processamento (caso você já tenha os batimentos segmentados):
        # x_train = X  # ou algum método para dividir X em treino e teste
        # y_train = ...  # definir os rótulos (ou usar um esquema de classificação binária ou multiclasses)
        #
        # OBS.: Se o Time2Graph espera os dados com shape (N, L, data_size),
        # verifique que o formato de X esteja correto.
        
        # Carrega os metadados para obter os rótulos corretamente
        caminho_arquivo = "/scratch/caio.grossi/Clustering-Paper/Projeto/Database/exams.csv"
        dados = pd.read_csv(caminho_arquivo)
        
        # Para cada exam_id presente na amostra, verifica os metadados para definir o rótulo.
        # Neste exemplo, se a coluna de ST (índice 8) for True, atribui rótulo 1 (por exemplo, 'ST'),
        # caso contrário, atribui 0 (normal). Ajuste conforme sua definição de classes.
        labels = []
        for exam_id in ids_ecgs:
            # Filtra a linha do exame atual
            linha = dados[dados['exam_id'] == exam_id]
            if linha.empty:
                # Se o exam_id não for encontrado, atribui um rótulo padrão (por exemplo, -1)
                labels.append(-1)
            else:
                r = linha.iloc[0]
                # Supondo que o índice 8 corresponde à condição ST (True indica anormalidade)
                if r.iloc[8] == True:
                    labels.append(1)
                else:
                    labels.append(0)
        
        # Agora, usando os dados carregados e os rótulos extraídos, definimos x_train, y_train, etc.
        x_train, y_train = X, np.array(labels)
        x_test, y_test = X, np.array(labels)
    else:
        raise NotImplementedError()
    Debugger.info_print('training: {:.2f} positive ratio with {}'.format(
        float(sum(y_train) / len(y_train)), len(y_train)))
    Debugger.info_print('test: {:.2f} positive ratio with {}'.format(
        float(sum(y_test) / len(y_test)), len(y_test)))

    # initialize Time2Graph model
    m = Time2Graph(gpu_enable=args.gpu_enable, **model_options, **general_options,
                   shapelets_cache='{}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
                       module_path, args.dataset, general_options['candidate_method'],
                       model_options['K'], model_options['seg_length']))
    if args.model_cache:
        m.load_model(fpath='{}/scripts/cache/{}_embedding_t2g_model.cache'.format(module_path, args.dataset))
    if args.shapelet_cache:
        m.t2g.load_shapelets(fpath=m.shapelets_cache)
    res = np.zeros(4, dtype=np.float32)

    Debugger.info_print('training {}_tim2graph_model ...'.format(args.dataset))
    cache_dir = '{}/scripts/cache/{}'.format(module_path, args.dataset)

    if not path.isdir(cache_dir):
        os.mkdir(cache_dir)
    m.fit(X=x_train, Y=y_train, n_splits=args.n_splits, tuning=False, opt_args=xgb_options)
    y_pred = m.predict(X=x_test)[0]
    Debugger.info_print('classification result: accuracy {:.4f}, precision {:.4f}, recall {:.4f}, F1 {:.4f}'.format(
            accuracy_score(y_true=y_test, y_pred=y_pred),
            precision_score(y_true=y_test, y_pred=y_pred),
            recall_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred)
        ))
