# %%
#IMPORTS
import pandas as pd
import h5py
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import os
import scipy.signal as sgn
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
import pandas as pd
import numpy as np
import h5py
import random

def carregar_ecgs_goldstandard(unlabel, umdavb, rbbb, lbbb, sb, st, af, multilabel,
                               unlabel_offset=0, umdavb_offset=0, rbbb_offset=0,
                               lbbb_offset=0, sb_offset=0, st_offset=0, af_offset=0, multilabel_offset=0,
                               filtrado=False):
    """
    Carrega ECGs a partir do CSV 'gold_standard_filtered.csv' e retorna:
      - X: array numpy com os traçados de ECG, shape (N, 12, num_amostras)
      - ids_ecgs: lista com os exam_id correspondentes
      - labels: array numpy de shape (N, 6), onde cada linha contém [1dAVb, RBBB, LBBB, SB, ST, AF].
                Se todos forem 0, o exame é considerado normal.
    
    Parâmetros de quantidade:
      unlabel    : número de ECGs 'normais' (coluna normal == 1)
      umdavb     : número de ECGs com 1dAVb (apenas se essa for a única doença)
      rbbb       : número de ECGs com RBBB (apenas se essa for a única doença)
      lbbb       : número de ECGs com LBBB (apenas se essa for a única doença)
      sb         : número de ECGs com SB (apenas se essa for a única doença)
      st         : número de ECGs com ST (apenas se essa for a única doença)
      af         : número de ECGs com AF (apenas se essa for a única doença)
      multilabel : número de ECGs multilabel (com pelo menos duas doenças dentre as 6)
    
    Parâmetros de offset (para fatiamento sequencial):
      unlabel_offset    : índice inicial para pegar ECGs normais
      umdavb_offset     : índice inicial para ECGs 1dAVb
      rbbb_offset       : índice inicial para ECGs RBBB
      lbbb_offset       : índice inicial para ECGs LBBB
      sb_offset         : índice inicial para ECGs SB
      st_offset         : índice inicial para ECGs ST
      af_offset         : índice inicial para ECGs AF
      multilabel_offset : índice inicial para ECGs multilabel
    
    filtrado:
      Se True, carrega de '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest/ecg_tracings_filtered.hdf5'
      Se False, de '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/data/ecg_tracings.hdf5'
    
    Retorna:
      X        : array com os traçados (N, 12, num_amostras)
      ids_ecgs : lista de exam_ids selecionados
      labels   : array (N, 6) com as labels [1dAVb, RBBB, LBBB, SB, ST, AF]
    """
    csv_path = '/scratch2/guilherme.evangelista/Clustering-Paper/Grafo/dataset/gold_standard_filtered.csv'
    dados = pd.read_csv(csv_path)
    
    # Se a coluna 'exam_id' não existir, cria-a com IDs sequenciais de 1 até N
    if 'exam_id' not in dados.columns:
        dados.insert(0, 'exam_id', list(range(1, len(dados)+1)))
        dados.to_csv(csv_path, index=False)
        print("Coluna 'exam_id' criada com IDs sequenciais e CSV atualizado.")
    
    # Calcula a soma dos sinais de doença para identificar multilabel
    bool_sum = (dados['1dAVb'].astype(int) +
                dados['RBBB'].astype(int) +
                dados['LBBB'].astype(int) +
                dados['SB'].astype(int) +
                dados['ST'].astype(int) +
                dados['AF'].astype(int))
    
    # Seleciona linhas para os casos multilabel (pelo menos duas doenças)
    ecg_multilabel_linhas = dados.index[bool_sum >= 2]
    
    # Seleciona os exames de cada categoria, excluindo os multilabel
    ecg_normal_linhas  = dados.index[dados['normal'] == 1]
    ecg_umdavb_linhas  = dados.index[(dados['1dAVb'] == 1) & (~dados.index.isin(ecg_multilabel_linhas))]
    ecg_rbbb_linhas    = dados.index[(dados['RBBB'] == 1) & (~dados.index.isin(ecg_multilabel_linhas))]
    ecg_lbbb_linhas    = dados.index[(dados['LBBB'] == 1) & (~dados.index.isin(ecg_multilabel_linhas))]
    ecg_sb_linhas      = dados.index[(dados['SB'] == 1) & (~dados.index.isin(ecg_multilabel_linhas))]
    ecg_st_linhas      = dados.index[(dados['ST'] == 1) & (~dados.index.isin(ecg_multilabel_linhas))]
    ecg_af_linhas      = dados.index[(dados['AF'] == 1) & (~dados.index.isin(ecg_multilabel_linhas))]
    
    print("Número de linhas ecg_normal_linhas:", len(ecg_normal_linhas))
    print("Número de linhas ecg_umdavb_linhas:", len(ecg_umdavb_linhas))
    print("Número de linhas ecg_rbbb_linhas:", len(ecg_rbbb_linhas))
    print("Número de linhas ecg_lbbb_linhas:", len(ecg_lbbb_linhas))
    print("Número de linhas ecg_sb_linhas:", len(ecg_sb_linhas))
    print("Número de linhas ecg_st_linhas:", len(ecg_st_linhas))
    print("Número de linhas ecg_af_linhas:", len(ecg_af_linhas))
    print("Número de linhas ecg_multilabel_linhas:", len(ecg_multilabel_linhas))
    
    # Obtém os exam_ids para cada categoria
    ecg_normal_ids      = dados.loc[ecg_normal_linhas, 'exam_id'].tolist()
    ecg_umdavb_ids      = dados.loc[ecg_umdavb_linhas, 'exam_id'].tolist()
    ecg_rbbb_ids        = dados.loc[ecg_rbbb_linhas, 'exam_id'].tolist()
    ecg_lbbb_ids        = dados.loc[ecg_lbbb_linhas, 'exam_id'].tolist()
    ecg_sb_ids          = dados.loc[ecg_sb_linhas, 'exam_id'].tolist()
    ecg_st_ids          = dados.loc[ecg_st_linhas, 'exam_id'].tolist()
    ecg_af_ids          = dados.loc[ecg_af_linhas, 'exam_id'].tolist()
    ecg_multilabel_ids  = dados.loc[ecg_multilabel_linhas, 'exam_id'].tolist()
    
    # Função auxiliar para fatiar a lista com um offset
    def slice_ids(id_list, offset, count):
        if offset >= len(id_list):
            return []
        return id_list[offset : offset + count]
    
    # Seleciona os blocos desejados usando slicing (não aleatório)
    ecg_normal_sample     = slice_ids(ecg_normal_ids,     unlabel_offset,    unlabel)
    ecg_umdavb_sample     = slice_ids(ecg_umdavb_ids,     umdavb_offset,     umdavb)
    ecg_rbbb_sample       = slice_ids(ecg_rbbb_ids,       rbbb_offset,       rbbb)
    ecg_lbbb_sample       = slice_ids(ecg_lbbb_ids,       lbbb_offset,       lbbb)
    ecg_sb_sample         = slice_ids(ecg_sb_ids,         sb_offset,         sb)
    ecg_st_sample         = slice_ids(ecg_st_ids,         st_offset,         st)
    ecg_af_sample         = slice_ids(ecg_af_ids,         af_offset,         af)
    ecg_multilabel_sample = slice_ids(ecg_multilabel_ids, multilabel_offset, multilabel)
    
    # Se a soma dos exames desejados for igual ao total do CSV, pegue todos os IDs
    total_requested = unlabel + umdavb + rbbb + lbbb + sb + st + af + multilabel
    if total_requested == len(dados):
        ids_ecgs = dados['exam_id'].tolist()
    else:
        # Combina os exam_ids selecionados e remove duplicatas (usando a ordem de aparecimento)
        ids_ecgs = list(dict.fromkeys(
            ecg_normal_sample +
            ecg_umdavb_sample +
            ecg_rbbb_sample +
            ecg_lbbb_sample +
            ecg_sb_sample +
            ecg_st_sample +
            ecg_af_sample +
            ecg_multilabel_sample
        ))
    
    print("Número total de ECGs no CSV:", len(dados))
    print("Número de ECGs selecionados:", len(ids_ecgs))
    
    # Define o(s) arquivo(s) HDF5 a serem utilizados
    if filtrado:
        arquivos_hdf5 = ['/scratch2/guilherme.evangelista/Clustering-Paper/Grafo/dataset/ecg_tracings_filtered.hdf5']
    else:
        arquivos_hdf5 = ['/scratch/guilherme.evangelista/Clustering-Paper/Grafo/data/ecg_tracings.hdf5']
    
    # Função auxiliar para obter os dados do exame a partir do HDF5 usando exam_id
    def get_ecg_data(file_path, exam_id):
        with h5py.File(file_path, 'r') as f:
            if 'exam_id' in f.keys():
                exam_ids = np.array(f['exam_id'])
                exam_index = np.where(exam_ids == exam_id)[0]
                if len(exam_index) == 0:
                    raise ValueError(f"Exam ID {exam_id} não encontrado.")
                exam_index = exam_index[0]
            else:
                # Se não houver dataset exam_id, assume que a ordem é a mesma do CSV
                exam_index = np.where(dados['exam_id'] == exam_id)[0][0]
            exam_tracings = f['tracings'][exam_index]
            return exam_tracings
    
    # Carrega os traçados de ECG para cada exam_id selecionado
    all_tracings = []
    for exam_id in ids_ecgs:
        found = False
        for arquivo in arquivos_hdf5:
            try:
                tracings = get_ecg_data(arquivo, exam_id)
                if tracings is not None:
                    # Transpõe para o formato (12, N_amostras)
                    tracing_transposto = np.array(tracings).T
                    all_tracings.append(tracing_transposto)
                    found = True
                    break
            except Exception:
                pass
        if not found:
            print(f"Erro: exame ID {exam_id} não encontrado em nenhum dos arquivos.")
    
    print("Número de ECGs que eram pra ser processados:", len(ids_ecgs))
    print("Número total de traçados processados:", len(all_tracings))
    
    X = np.array(all_tracings)
    print("Shape de X:", X.shape)
    
    # Gerar labels: para cada exame, o label é um vetor de 6 posições: [1dAVb, RBBB, LBBB, SB, ST, AF]
    labels = []
    for eid in ids_ecgs:
        row = dados.loc[dados['exam_id'] == eid]
        if len(row) == 0:
            labels.append([0, 0, 0, 0, 0, 0])
        else:
            row = row.iloc[0]
            label = [
                int(row['1dAVb']),
                int(row['RBBB']),
                int(row['LBBB']),
                int(row['SB']),
                int(row['ST']),
                int(row['AF'])
            ]
            labels.append(label)
    labels = np.array(labels, dtype=int)
    
    return X, ids_ecgs, labels


# %%
# Exemplo de chamada da função:
X, ids_ecgs, labels = carregar_ecgs_goldstandard(
     unlabel=681,    unlabel_offset=0,
     umdavb=20,      umdavb_offset=0,
     rbbb=28,       rbbb_offset=0,
     lbbb=25,        lbbb_offset=0,
     sb=15,          sb_offset=0,
     st=35,          st_offset=0,
     af=11,         af_offset=0,
     multilabel=12,  multilabel_offset=0,
     filtrado=True
)

# %%
# Vamos imprimir um exemplo de 10 exames, mostrando seu exam_id e respectivo label
for i in range(min(10, len(ids_ecgs))):
    print(f"Exam ID: {ids_ecgs[i]}, Label: {labels[i]}")


# %%
import torch
import numpy as np
import neurokit2 as nk
import networkx as nx  # Para calcular PageRank, se necessário
from ts2vg import NaturalVG
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm

def get_middle_r_peak(lead_series, sampling_rate=400):
    peaks_dict = nk.ecg_findpeaks(lead_series, sampling_rate=sampling_rate)
    peaks = np.array(peaks_dict["ECG_R_Peaks"])
    if peaks.size == 0:
        return len(lead_series) // 2
    # Se número par, retorna o pico anterior ao meio; senão, o pico do meio
    if len(peaks) % 2 == 0:
        return peaks[len(peaks) // 2 - 1]
    else:
        return peaks[len(peaks) // 2]

def process_exam(ecg, exam_id, label, scaler=None):
    """
    Processa um ECG (12 leads) e retorna:
      - exam_id
      - um grafo baseado na lead1 (para a conectividade), com features combinadas das leads 0, 1, 6 e 11,
        onde o grau é calculado individualmente para cada lead.
      - a label associada a esse exame.
    """
    # Definir o segmento com base na lead1
    lead1_series = ecg[1]
    r_peak = get_middle_r_peak(lead1_series, sampling_rate=400)
    start_index = max(0, r_peak - 500)
    end_index = min(len(lead1_series), r_peak + 500)
    n = 1000  # comprimento esperado do segmento

    if (end_index - start_index) != n:
        node_features = np.zeros((n, 28))
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        valid = False
    else:
        # Constrói o grafo de visibilidade para conectividade com base na lead1
        lead1_segment = ecg[1][start_index:end_index]
        vg = NaturalVG()
        vg.build(lead1_segment)
        edges = vg.edges
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.int64)
        
        # Para as leads selecionadas: 0, 1, 6 e 11, calcular features individualmente,
        # incluindo o grau calculado com o grafo de visibilidade de cada lead.
        leads_indices = [0, 1, 6, 11]
        feature_list = []
        for lead in leads_indices:
            segment = ecg[lead][start_index:end_index]
            amplitude = segment.reshape(-1, 1)
            derivative = np.concatenate(([0], np.diff(segment))).reshape(-1, 1)
            
            # Calcula o grau para o segmento da lead atual
            vg_lead = NaturalVG()
            vg_lead.build(segment)
            edges_lead = vg_lead.edges
            if edges_lead:
                edges_lead_array = np.array(edges_lead)
                counts = np.bincount(np.concatenate([edges_lead_array[:, 0], edges_lead_array[:, 1]]), minlength=n)
                degree = counts.reshape(-1, 1).astype(float)
            else:
                degree = np.zeros((n, 1))
            
            # Vetor one-hot para identificar a lead (dimensão 4)
            one_hot = np.zeros((n, 4))
            one_hot[:, leads_indices.index(lead)] = 1.0
            
            # Concatena: [amplitude, derivada, grau, one_hot] => 3 contínuas + 4 one-hot = 7 features
            features_this_lead = np.hstack([amplitude, derivative, degree, one_hot])
            feature_list.append(features_this_lead)
        
        # Concatena as features de todas as 4 leads: resultado final (n, 28)
        node_features = np.hstack(feature_list)
        valid = True

    # Normalizar separadamente cada grupo de features contínuas, se scaler for fornecido.
    if scaler is not None:
        amp_idx = np.array([0, 7, 14, 21])
        deriv_idx = np.array([1, 8, 15, 22])
        degree_idx = np.array([2, 9, 16, 23])
        node_features[:, amp_idx] = (node_features[:, amp_idx] - scaler["amplitude"][0]) / scaler["amplitude"][1]
        node_features[:, deriv_idx] = (node_features[:, deriv_idx] - scaler["derivative"][0]) / scaler["derivative"][1]
        node_features[:, degree_idx] = (node_features[:, degree_idx] - scaler["degree"][0]) / scaler["degree"][1]

    data = Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=edge_index)
    return exam_id, data, label, valid


if __name__ == '__main__':
    # Primeira passagem sem normalização para coletar as features e calcular os parâmetros
    from joblib import Parallel, delayed
    from tqdm import tqdm

    results_temp = Parallel(n_jobs=20, verbose=10)(
        delayed(process_exam)(ecg, ids_ecgs[idx], labels[idx], scaler=None)
        for idx, ecg in enumerate(tqdm(X, desc="Processando exames (temp)"))
    )
    
    # Colete as features de todos os exames válidos
    all_features = []
    for exam_id, data, lbl, valid in results_temp:
        if valid:
            all_features.append(data.x.numpy())
    all_features = np.vstack(all_features)  # forma (N_total, 28)
    
    # Índices para cada grupo
    amp_idx = np.array([0, 7, 14, 21])
    deriv_idx = np.array([1, 8, 15, 22])
    degree_idx = np.array([2, 9, 16, 23])
    
    mean_amp = np.mean(all_features[:, amp_idx])
    std_amp = np.std(all_features[:, amp_idx])
    mean_deriv = np.mean(all_features[:, deriv_idx])
    std_deriv = np.std(all_features[:, deriv_idx])
    mean_degree = np.mean(all_features[:, degree_idx])
    std_degree = np.std(all_features[:, degree_idx])
    
    scaler = {"amplitude": (mean_amp, std_amp),
              "derivative": (mean_deriv, std_deriv),
              "degree": (mean_degree, std_degree)}
    
    print("Parâmetros de normalização:")
    print("Amplitude - Média:", mean_amp, "Std:", std_amp)
    print("Derivada  - Média:", mean_deriv, "Std:", std_deriv)
    print("Grau      - Média:", mean_degree, "Std:", std_degree)
    
    # Agora, processe novamente os exames aplicando a normalização
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_exam)(ecg, ids_ecgs[idx], labels[idx], scaler=scaler)
        for idx, ecg in enumerate(tqdm(X, desc="Processando exames (normalizados)"))
    )
    
    graphs_by_exam = {}
    count_invalid = 0
    for exam_id, data, lbl, valid in results:
        graphs_by_exam[exam_id] = {
            "grafo": data,  # Grafo com features normalizadas
            "label": lbl
        }
        if not valid:
            count_invalid += 1

    # Salvar os grafos e labels em um arquivo HDF5
    output_filename = "codetesthd.hdf5"
    with h5py.File(output_filename, "w") as f:
        grp = f.create_group("grafos")
        for exam_id, content in graphs_by_exam.items():
            exam_grp = grp.create_group(str(exam_id))
            exam_grp.create_dataset("x", data=content["grafo"].x.numpy())
            exam_grp.create_dataset("edge_index", data=content["grafo"].edge_index.numpy())
            exam_grp.create_dataset("label", data=np.array(content["label"]))
    print(f"\nGrafos salvos em {output_filename}")
    print(f"Quantidade de exames com segmento inválido (≠ 1000 pontos): {count_invalid}")

