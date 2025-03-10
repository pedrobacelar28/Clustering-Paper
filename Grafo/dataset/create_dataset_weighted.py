# %%
# IMPORTS

import pandas as pd
import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import neurokit2 as nk
import random
import networkx as nx
import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx # Visualization
from torch.nn import Linear                   # Define layers
from torch_geometric.nn import GCNConv
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d 
from scipy.signal import medfilt
from ts2vg import NaturalVG
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import os
import matplotlib.colors as mcolors


# %%

# ================================================
# Carregar os dados
# ================================================

def carregar_ecgs(unlabel, umdavb, rbbb, lbbb, sb, st, af, multilabel,
                  unlabel_offset=0, umdavb_offset=0, rbbb_offset=0,
                  lbbb_offset=0, sb_offset=0, st_offset=0, af_offset=0, multilabel_offset=0,
                  filtrado=False):
    """
    Carrega os ECGs e retorna:
      - X: array numpy com os traçados de ECG, shape (N, 12, num_amostras_por_sinal)
      - ids_ecgs: lista com os exam_id correspondentes
      - labels: array numpy de shape (N, 6), onde cada linha contém [UMdAVB, RBBB, LBBB, SB, ST, AF].
                Se todos forem 0, significa ECG normal (unlabel).

    Parâmetros:
      unlabel    : quantidade de ECGs normais
      umdavb     : quantidade de ECGs com UMdAVB (apenas essa doença)
      rbbb       : quantidade de ECGs com RBBB  (apenas essa doença)
      lbbb       : quantidade de ECGs com LBBB  (apenas essa doença)
      sb         : quantidade de ECGs com SB    (apenas essa doença)
      st         : quantidade de ECGs com ST    (apenas essa doença)
      af         : quantidade de ECGs com AF    (apenas essa doença)
      multilabel : quantidade de ECGs com pelo menos duas doenças simultâneas

      unlabel_offset    : índice de início (offset) para pegar ECGs normais
      umdavb_offset     : índice de início para UMdAVB
      rbbb_offset       : índice de início para RBBB
      lbbb_offset       : índice de início para LBBB
      sb_offset         : índice de início para SB
      st_offset         : índice de início para ST
      af_offset         : índice de início para AF
      multilabel_offset : índice de início para ECGs multilabel

      filtrado   : se True, carrega arquivos de ECG filtrados; caso contrário, carrega os brutos

    Exemplo de uso para pegar os primeiros 1000:
      carregar_ecgs(
        unlabel=1000, unlabel_offset=0,   # do 0 ao 999
        ...
      )
    E para depois pegar os próximos 1000:
      carregar_ecgs(
        unlabel=1000, unlabel_offset=1000, # do 1000 ao 1999
        ...
      )
    """
    caminho_arquivo = "../../Projeto/Database/exams.csv"
    dados = pd.read_csv(caminho_arquivo)

    # Arquivos HDF5 que vamos considerar
    arquivos_usados = [
        "exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5",
        "exams_par4.hdf5",  "exams_part5.hdf5", "exams_part6.hdf5", "exams_part7.hdf5",
        "exams_par8.hdf5",  "exams_part9.hdf5", "exams_part10.hdf5", "exams_part11.hdf5",
        "exams_part12.hdf5","exams_part13.hdf5","exams_part14.hdf5","exams_part15.hdf5",
        "exams_part16.hdf5","exams_part17.hdf5"
    ]

    # ======================
    # 1) Filtrar pelo col14 nos arquivos_usados e col13=False, etc.
    # ======================
    ecg_normal_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (dados.iloc[:, 4] == False) &
        (dados.iloc[:, 5] == False) &
        (dados.iloc[:, 6] == False) &
        (dados.iloc[:, 7] == False) &
        (dados.iloc[:, 8] == False) &
        (dados.iloc[:, 9] == False)
    ]
    ecg_umdavb_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (dados.iloc[:, 4] == True)  &
        (dados.iloc[:, 5] == False) &
        (dados.iloc[:, 6] == False) &
        (dados.iloc[:, 7] == False) &
        (dados.iloc[:, 8] == False) &
        (dados.iloc[:, 9] == False) &
        (dados.iloc[:, 13] == False)
    ]
    ecg_rbbb_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (dados.iloc[:, 4] == False) &
        (dados.iloc[:, 5] == True)  &
        (dados.iloc[:, 6] == False) &
        (dados.iloc[:, 7] == False) &
        (dados.iloc[:, 8] == False) &
        (dados.iloc[:, 9] == False) &
        (dados.iloc[:, 13] == False)
    ]
    ecg_lbbb_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (dados.iloc[:, 4] == False) &
        (dados.iloc[:, 5] == False) &
        (dados.iloc[:, 6] == True)  &
        (dados.iloc[:, 7] == False) &
        (dados.iloc[:, 8] == False) &
        (dados.iloc[:, 9] == False) &
        (dados.iloc[:, 13] == False)
    ]
    ecg_sb_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (dados.iloc[:, 4] == False) &
        (dados.iloc[:, 5] == False) &
        (dados.iloc[:, 6] == False) &
        (dados.iloc[:, 7] == True)  &
        (dados.iloc[:, 8] == False) &
        (dados.iloc[:, 9] == False) &
        (dados.iloc[:, 13] == False)
    ]
    ecg_st_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (dados.iloc[:, 4] == False) &
        (dados.iloc[:, 5] == False) &
        (dados.iloc[:, 6] == False) &
        (dados.iloc[:, 7] == False) &
        (dados.iloc[:, 8] == True)  &
        (dados.iloc[:, 9] == False) &
        (dados.iloc[:, 13] == False)
    ]
    ecg_af_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (dados.iloc[:, 4] == False) &
        (dados.iloc[:, 5] == False) &
        (dados.iloc[:, 6] == False) &
        (dados.iloc[:, 7] == False) &
        (dados.iloc[:, 8] == False) &
        (dados.iloc[:, 9] == True)  &
        (dados.iloc[:, 13] == False)
    ]

    # Multilabel = pelo menos 2 doenças
    bool_sum = (
        dados.iloc[:, 4].astype(int) +
        dados.iloc[:, 5].astype(int) +
        dados.iloc[:, 6].astype(int) +
        dados.iloc[:, 7].astype(int) +
        dados.iloc[:, 8].astype(int) +
        dados.iloc[:, 9].astype(int)
    )
    ecg_multilabel_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (bool_sum >= 2) &
        (dados.iloc[:, 13] == False)
    ]

    print("Número de linhas ecg_normal_linhas:", len(ecg_normal_linhas))
    print("Número de linhas ecg_umdavb_linhas:", len(ecg_umdavb_linhas))
    print("Número de linhas ecg_rbbb_linhas:", len(ecg_rbbb_linhas))
    print("Número de linhas ecg_lbbb_linhas:", len(ecg_lbbb_linhas))
    print("Número de linhas ecg_sb_linhas:", len(ecg_sb_linhas))
    print("Número de linhas ecg_st_linhas:", len(ecg_st_linhas))
    print("Número de linhas ecg_af_linhas:", len(ecg_af_linhas))
    print("Número de linhas ecg_multilabel_linhas:", len(ecg_multilabel_linhas))

    # ======================
    # 2) Excluir exames com interferência
    # ======================
    caminho_interferencias = "../../Projeto/Database/resultados_interferencia.csv"
    interferencias = pd.read_csv(caminho_interferencias)
    interferencias_ids = interferencias['exam_id'].tolist()

    ecg_normal_linhas     = ecg_normal_linhas[~dados.loc[ecg_normal_linhas, 'exam_id'].isin(interferencias_ids)]
    ecg_umdavb_linhas     = ecg_umdavb_linhas[~dados.loc[ecg_umdavb_linhas, 'exam_id'].isin(interferencias_ids)]
    ecg_rbbb_linhas       = ecg_rbbb_linhas[~dados.loc[ecg_rbbb_linhas, 'exam_id'].isin(interferencias_ids)]
    ecg_lbbb_linhas       = ecg_lbbb_linhas[~dados.loc[ecg_lbbb_linhas, 'exam_id'].isin(interferencias_ids)]
    ecg_sb_linhas         = ecg_sb_linhas[~dados.loc[ecg_sb_linhas, 'exam_id'].isin(interferencias_ids)]
    ecg_st_linhas         = ecg_st_linhas[~dados.loc[ecg_st_linhas, 'exam_id'].isin(interferencias_ids)]
    ecg_af_linhas         = ecg_af_linhas[~dados.loc[ecg_af_linhas, 'exam_id'].isin(interferencias_ids)]
    ecg_multilabel_linhas = ecg_multilabel_linhas[~dados.loc[ecg_multilabel_linhas, 'exam_id'].isin(interferencias_ids)]

    print("\nTirando Interferência:")
    print("Número de linhas ecg_normal_linhas:", len(ecg_normal_linhas))
    print("Número de linhas ecg_umdavb_linhas:", len(ecg_umdavb_linhas))
    print("Número de linhas ecg_rbbb_linhas:", len(ecg_rbbb_linhas))
    print("Número de linhas ecg_lbbb_linhas:", len(ecg_lbbb_linhas))
    print("Número de linhas ecg_sb_linhas:", len(ecg_sb_linhas))
    print("Número de linhas ecg_st_linhas:", len(ecg_st_linhas))
    print("Número de linhas ecg_af_linhas:", len(ecg_af_linhas))
    print("Número de linhas ecg_multilabel_linhas:", len(ecg_multilabel_linhas))

    # ======================
    # 3) Obter exam_id de cada grupo
    # ======================
    ecg_normal_id      = dados.iloc[ecg_normal_linhas, 0].tolist()
    ecg_umdavb_id      = dados.iloc[ecg_umdavb_linhas, 0].tolist()
    ecg_rbbb_id        = dados.iloc[ecg_rbbb_linhas, 0].tolist()
    ecg_lbbb_id        = dados.iloc[ecg_lbbb_linhas, 0].tolist()
    ecg_sb_id          = dados.iloc[ecg_sb_linhas, 0].tolist()
    ecg_st_id          = dados.iloc[ecg_st_linhas, 0].tolist()
    ecg_af_id          = dados.iloc[ecg_af_linhas, 0].tolist()
    ecg_multilabel_id  = dados.iloc[ecg_multilabel_linhas, 0].tolist()

    # ======================
    # 4) Em vez de random.sample(...), usamos slicing com offset
    #    Ex.: ecg_normal_id[unlabel_offset : unlabel_offset + unlabel]
    # ======================
    # Se a lista for menor do que o offset, devolvemos lista vazia
    # Se a lista ainda tiver espaço após offset, pegamos a fatia
    def slice_ids(id_list, offset, count):
        if offset >= len(id_list):
            return []  # não há nada para pegar se offset estiver além do tamanho da lista
        return id_list[offset : offset + count]

    ecg_normal_sample     = slice_ids(ecg_normal_id,     unlabel_offset,    unlabel)
    ecg_umdavb_sample     = slice_ids(ecg_umdavb_id,     umdavb_offset,     umdavb)
    ecg_rbbb_sample       = slice_ids(ecg_rbbb_id,       rbbb_offset,       rbbb)
    ecg_lbbb_sample       = slice_ids(ecg_lbbb_id,       lbbb_offset,       lbbb)
    ecg_sb_sample         = slice_ids(ecg_sb_id,         sb_offset,         sb)
    ecg_st_sample         = slice_ids(ecg_st_id,         st_offset,         st)
    ecg_af_sample         = slice_ids(ecg_af_id,         af_offset,         af)
    ecg_multilabel_sample = slice_ids(ecg_multilabel_id, multilabel_offset, multilabel)

    # ======================
    # 5) Combina todos os IDs (ordem é a dada pela concatenação simples)
    # ======================
    ids_ecgs = (
        ecg_normal_sample +
        ecg_umdavb_sample +
        ecg_rbbb_sample +
        ecg_lbbb_sample +
        ecg_sb_sample +
        ecg_st_sample +
        ecg_af_sample +
        ecg_multilabel_sample
    )

    print("\nNúmero total de ECGs selecionados:", len(ids_ecgs))

    # ======================
    # 6) Selecionar caminhos HDF5 (filtrado ou não)
    # ======================
    if filtrado:
        arquivos_hdf5 = [
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_0_1.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_2_3.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_4_5.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_6_7.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_8_9.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_10_11.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_12_13.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_14_15.hdf5",
            "/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_16_17.hdf5"
        ]
    else:
        arquivos_hdf5 = [
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part0.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part1.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part2.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part3.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part4.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part5.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part6.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part7.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part8.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part9.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part10.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part11.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part12.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part13.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part14.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part15.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part16.hdf5',
            '/scratch/pedro.bacelar/Clustering-Paper/Projeto/Database/exams_part17.hdf5'
        ]

    # ======================
    # 7) Função auxiliar para ler o exame no HDF5
    # ======================
    def get_ecg_data(file_path, exam_id):
        with h5py.File(file_path, 'r') as f:
            exam_ids = np.array(f['exam_id'])
            exam_index = np.where(exam_ids == exam_id)[0]
            if len(exam_index) == 0:
                raise ValueError("Exam ID não encontrado.")
            exam_index = exam_index[0]
            exam_tracings = f['tracings'][exam_index]
            return exam_tracings

    # ======================
    # 8) Carrega os traçados
    # ======================
    all_tracings = []
    for exam_id in ids_ecgs:
        found = False
        for arquivo in arquivos_hdf5:
            try:
                tracings = get_ecg_data(arquivo, exam_id)
                if tracings is not None:
                    # Transpõe para shape (12, n_amostras)
                    tracing_transposto = np.array(tracings).T
                    all_tracings.append(tracing_transposto)
                    found = True
                    break
            except ValueError:
                # Se não achou esse exam_id nesse arquivo, pula
                pass
            except Exception as e:
                # Se houver outro erro, também só ignore
                pass

        if not found:
            print(f"Erro: exame ID {exam_id} não encontrado em nenhum dos arquivos.")

    print("\nNúmero de ecgs que eram pra ser processados:", len(ids_ecgs))
    print(f"Número total de traçados efetivamente carregados: {len(all_tracings)}")

    # ======================
    # 9) Monta X e as labels
    # ======================
    # X -> (N, 12, num_amostras)
    X = np.array(all_tracings)

    # labels -> (N, 6) => [UMdAVB, RBBB, LBBB, SB, ST, AF]
    labels = []
    for eid in ids_ecgs:
        row = dados.loc[dados['exam_id'] == eid]
        if len(row) == 0:
            labels.append([0, 0, 0, 0, 0, 0])
        else:
            row = row.iloc[0]
            label = [
                int(row.iloc[4]),  # UMdAVB
                int(row.iloc[5]),  # RBBB
                int(row.iloc[6]),  # LBBB
                int(row.iloc[7]),  # SB
                int(row.iloc[8]),  # ST
                int(row.iloc[9])   # AF
            ]
            labels.append(label)

    labels = np.array(labels, dtype=int)

    return X, ids_ecgs, labels

X, ids_ecgs, labels = carregar_ecgs(unlabel=10,umdavb=10, rbbb=5, lbbb=0, sb=0, st=0, af=0, multilabel=0,unlabel_offset=0, umdavb_offset=0, rbbb_offset=0,
                                    lbbb_offset=0, sb_offset=0, st_offset=0, af_offset=0, multilabel_offset=0,filtrado=True)

# Vamos imprimir um exemplo de 10 exames, mostrando seu exam_id e respectivo label
for i in range(min(1, len(ids_ecgs))):
    print(f"Exam ID: {ids_ecgs[i]}, Label: {labels[i]}")


# %%

# ================================================
# Funções para a Construção do Grafo
# ================================================


def compute_node_features(time_series, edges):
    """
    Calcula as features de cada nó:
      - amplitude: valor da amostra;
      - derivada: diferença com o nó anterior (primeiro nó = 0);
      - grau: número de arestas incidentes;
      - pagerank: valor de pagerank calculado via NetworkX.
    
    Parâmetros:
      time_series: numpy array de forma (n,) com os valores da lead.
      edges: lista de tuplas (i, j) definindo as arestas do grafo.
      
    Retorna:
      features: numpy array de forma (n, 4) com
                [amplitude, derivada, grau, pagerank] para cada nó.
    """
    n = len(time_series)
    amplitude = time_series.reshape(-1, 1)
    derivative = np.concatenate(([0], np.diff(time_series))).reshape(-1, 1)
    
    if edges:
        edges_array = np.array(edges)
        # Separa nós de origem e destino
        u, v = edges_array[:, 0], edges_array[:, 1]
        counts = np.bincount(np.concatenate([u, v]), minlength=n)
    else:
        counts = np.zeros(n)
    degree = counts.reshape(-1, 1)
    
    # Cálculo do PageRank usando NetworkX
    if edges:
        G = nx.Graph()
        G.add_nodes_from(range(n))    # garante que todos os nós estejam no grafo
        G.add_edges_from(edges)
        pr_values = nx.pagerank(G)    # dicionário {nó: pagerank}
        pagerank_arr = np.array([pr_values[i] for i in range(n)]).reshape(-1, 1)
    else:
        pagerank_arr = np.zeros((n, 1))

    features = np.hstack([amplitude, derivative, degree, pagerank_arr])
    return features

def compute_edge_weights(edges, time_series):
    """
    Calcula os pesos das arestas como a distância euclidiana entre os nós conectados.

    Parâmetros:
      edges: lista de tuplas (i, j) definindo as arestas do grafo.
      time_series: numpy array com os valores da série temporal.

    Retorna:
      edge_weights: Tensor contendo os pesos das arestas.
    """
    edge_weights = []
    for i, j in edges:
        dist = np.abs(time_series[i] - time_series[j])  # Distância euclidiana em 1D
        edge_weights.append(dist)

    return torch.tensor(edge_weights, dtype=torch.float32)

def get_middle_r_peak(lead_series, sampling_rate=400):
    """
    Detecta os picos R na lead utilizando nk.ecg_findpeaks do NeuroKit e retorna o pico "do meio".
    Caso nenhum pico seja encontrado, retorna o índice central da série.
    """
    peaks_dict = nk.ecg_findpeaks(lead_series, sampling_rate=sampling_rate)
    peaks = np.array(peaks_dict["ECG_R_Peaks"])
    if peaks.size == 0:
        return len(lead_series) // 2
    
    if len(peaks) % 2 == 0:
        middle_index = peaks[len(peaks) // 2 - 1]
    else:
        middle_index = peaks[len(peaks) // 2]
    return middle_index

def process_exam(ecg, exam_id, label):
    """
    Processa um ECG (12 leads) e retorna:
      - exam_id
      - grafo da lead1 com features concatenadas de todas as 12 leads (48 features por nó)
      - label associada a esse exame.
      
    A segmentação é baseada na lead1. Se o segmento não tiver 1000 pontos,
    as features serão um array de zeros de forma (1000, 48) e o grafo terá edge_index vazio.
    """
    # Usar a lead1 para determinar o segmento
    lead1_series = ecg[1]
    r_peak = get_middle_r_peak(lead1_series, sampling_rate=400)
    start_index = max(0, r_peak - 500)
    end_index = min(len(lead1_series), r_peak + 500)
    segment_length = end_index - start_index

    if segment_length != 1000:
        # Caso o segmento não possua 1000 pontos: features nulas e grafo vazio.
        node_features = np.zeros((1000, 48))
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        edge_weights = torch.empty((0,), dtype=torch.float32)
        valid = False
    else:
        features_list = []
        # Para cada uma das 12 leads, extrai o segmento com os mesmos índices
        for lead in range(12):
            lead_segment = ecg[lead][start_index:end_index]
            # Para cada lead, calcula as features usando seu próprio grafo de visibilidade.
            vg = NaturalVG()
            vg.build(lead_segment)
            edges = vg.edges
            feat = compute_node_features(lead_segment, edges)
            features_list.append(feat)
            # Para a lead1, usaremos o grafo para definir o edge_index do Data
            if lead == 1:
                if edges:
                    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
                    edge_weights = compute_edge_weights(edges, lead_segment)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.int64)
                    edge_weights = torch.empty((0,), dtype=torch.float32)
        # Concatena as features de todas as leads (eixo das colunas)
        node_features = np.hstack(features_list)  # Resultado: (1000, 48)
        valid = True

    data = Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_weights)
    return exam_id, data, label, valid

# %%

# ================================================
# Main
# ================================================


# SUPOSIÇÕES:
#  - X, ids_ecgs e labels estão definidos e têm mesmo tamanho N.
#  - X: (N, 12, num_amostras)
#  - ids_ecgs: lista com N exam_ids
#  - labels: array/list com as N labels

if __name__ == '__main__':
    print("Criando grafos de visibilidade para cada ECG...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_exam)(ecg, ids_ecgs[idx], labels[idx])
        for idx, ecg in enumerate(tqdm(X, desc="Processando exames"))
    )

    graphs_by_exam = {}
    count_invalid = 0
    for (exam_id, data, lbl, valid) in results:
        graphs_by_exam[exam_id] = {
            "grafo": data,  
            "label": lbl
        }
        if not valid:
            count_invalid += 1

    dados_salvos = {"grafos": graphs_by_exam}
    output_filename = "dataset_with_weights.pt"
    torch.save(dados_salvos, output_filename)
    print(f"\nGrafos salvos em {output_filename}")
    print(f"Quantidade de exames que não possuem 1000 pontos: {count_invalid}")

    # Carregar o arquivo salvo e exibir 5 exemplos
    loaded_data = torch.load(output_filename, weights_only=False)
    exam_keys = list(loaded_data["grafos"].keys())
    print("\nExemplos de 5 exames:")
    for key in exam_keys[:5]:
        exame = loaded_data["grafos"][key]
        print(f"Exam ID: {key}")
        print(f"Label: {exame['label']}")
        print(f"Grafo (lead1) Data:")
        print(f"  x shape: {exame['grafo'].x.shape}")
        print(f"  edge_index shape: {exame['grafo'].edge_index.shape}")
        print(f"  edge_attr shape: {exame['grafo'].edge_attr.shape}\n")





# ================================================
# TESTANDO
# ================================================




# ================================================
# 1) Carregar o dataset salvo
# ================================================
dataset_filename = "dataset_with_weights.pt"

try:
    loaded_data = torch.load(dataset_filename, map_location=torch.device('cpu'), weights_only=False)
    graphs_by_exam = loaded_data["grafos"]
    print(f"\nDataset '{dataset_filename}' carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# ================================================
# 2) Verificar a Estrutura do Dataset
# ================================================
exam_keys = list(graphs_by_exam.keys())

print("\n========================= INFORMAÇÕES DO DATASET =========================")
print(f"Total de exames no dataset: {len(exam_keys)}")

# Pegamos um exemplo para inspecionar
exam_id = exam_keys[0]
exam_data = graphs_by_exam[exam_id]

grafo = exam_data["grafo"]
label = exam_data["label"]

# Verificando as formas dos arrays e atributos do grafo
print(f"\nExam ID: {exam_id}")
print(f"Label: {label}")
print(f"Formato de x (features dos nós): {grafo.x.shape}")
print(f"Formato de edge_index (arestas): {grafo.edge_index.shape}")
print(f"Formato de edge_attr (pesos das arestas): {grafo.edge_attr.shape}")

# Contar grafos vazios (sem conexões)
empty_graphs = sum(1 for g in graphs_by_exam.values() if g["grafo"].edge_index.shape[1] == 0)
print(f"\nQuantidade de grafos vazios (sem arestas): {empty_graphs}/{len(exam_keys)}")


# ================================================
# 3) Função para Visualizar Exemplos
# ================================================

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import os
import matplotlib.colors as mcolors

# ================================================
# 1) Carregar o dataset salvo
# ================================================
dataset_filename = "dataset_with_weights.pt"

try:
    loaded_data = torch.load(dataset_filename, map_location=torch.device('cpu'), weights_only=False)
    graphs_by_exam = loaded_data["grafos"]
    print(f"\nDataset '{dataset_filename}' carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# ================================================
# 2) Criar diretório para salvar figuras
# ================================================
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# ================================================
# 3) Função para Visualizar e Salvar Exemplos com 3 Imagens
# ================================================

import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import os

def plot_ecg_and_graph(exam_id):
    """
    Plota e salva um exemplo de ECG e a estrutura do grafo correspondente,
    em uma figura com 3 subplots:
      1) Sinal de ECG original.
      2) Sinal de ECG com nós e arestas sobrepostos (coloridos pelos pesos).
      3) Grafo desenhado em formato de rede, com cores baseadas na 4ª feature dos nós.
    """
    if exam_id not in graphs_by_exam:
        print(f"Exam ID {exam_id} não encontrado.")
        return
    
    exam_data = graphs_by_exam[exam_id]
    grafo = exam_data["grafo"]
    
    # Verificar se tem arestas
    if grafo.edge_index.shape[1] == 0:
        print(f"Exam ID {exam_id} tem um grafo vazio. Pulando visualização.")
        return
    
    # Criar o grafo NetworkX para visualização
    G = to_networkx(grafo, edge_attrs=["edge_attr"])

    # Criar figura com 3 subplots (um em cima do outro)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # =======================
    # 1️⃣ Subplot: Sinal de ECG cru
    # =======================
    signal = grafo.x[:, 0].numpy()
    axes[0].plot(signal, color="black")
    axes[0].set_xlabel("Amostra")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title("Sinal de ECG (Original)")

    # =======================
    # 2️⃣ Subplot: Sinal com nós e arestas
    # =======================
    axes[1].plot(signal, color="black", alpha=0.5)  # Sinal de ECG como fundo
    axes[1].set_title("ECG com Arestas do Grafo")

    # Obter pesos das arestas
    edge_weights = [d['edge_attr'] for _, _, d in G.edges(data=True)]
    
    # Normalizar os pesos para escala de cores
    if edge_weights:
        norm_edges = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        cmap_edges = plt.cm.Reds  # Branco → Vermelho
    else:
        norm_edges = None
        cmap_edges = "gray"

    # Criar nós espaçados (a cada 10 pontos)
    step = 10  # Define o espaçamento
    nodes_x = np.arange(0, len(signal), step)  # Posição dos nós espaçados
    nodes_y = signal[nodes_x]  # Altura dos nós espaçados

    # Adicionar nós ao gráfico
    axes[1].scatter(nodes_x, nodes_y, color="blue", s=10, label="Nós (Espaçados)")

    # Adicionar arestas espaçadas
    for (u, v, data) in G.edges(data=True):
        if u % step == 0 and v % step == 0:  # Apenas desenha arestas entre nós espaçados
            weight = data["edge_attr"]
            color = cmap_edges(norm_edges(weight)) if norm_edges else "gray"
            axes[1].plot([nodes_x[u // step], nodes_x[v // step]], 
                         [nodes_y[u // step], nodes_y[v // step]], 
                         color=color, alpha=0.7, lw=1)

    axes[1].legend()
    axes[1].set_xlim(0, len(signal))  # Define o eixo X maior que o Y

    # =======================
    # 3️⃣ Subplot: Grafo desenhado separadamente
    # =======================
    axes[2].set_title("Grafo de Visibilidade (Cores pela 4ª Feature)")

    # Garantir que o layout do grafo está bem definido
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42)  # Layout fixo para consistência
    else:
        print(f"⚠️  Warning: Grafo do Exam {exam_id} não possui nós!")
        return

    # Obter a quarta feature dos nós
    node_features = grafo.x[:, 3].numpy()  # 4ª feature (índice 3)

    # Normalizar as cores dos nós para a escala de azul
    if len(node_features) > 0:
        norm_nodes = mcolors.Normalize(vmin=min(node_features), vmax=max(node_features))
        cmap_nodes = plt.cm.Blues
        node_colors = [cmap_nodes(norm_nodes(f)) for f in node_features]
    else:
        node_colors = "gray"

    # Obter pesos das arestas novamente para este subplot
    edge_colors = [cmap_edges(norm_edges(w)) for w in edge_weights] if norm_edges else "gray"

    # Desenhar o grafo
    nx.draw(
        G, pos, ax=axes[2],
        with_labels=False, node_size=100, node_color=node_colors, edge_color=edge_colors, edge_cmap=plt.cm.Reds
    )

    # Adicionar barra de cores para os nós
    if len(node_features) > 0:
        sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm_nodes)
        sm_nodes.set_array([])
        cbar_nodes = plt.colorbar(sm_nodes, ax=axes[2], fraction=0.03, pad=0.02)
        cbar_nodes.set_label("Valor da 4ª Feature (Nós)")

    # Adicionar barra de cores para as arestas
    if edge_weights:
        sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm_edges)
        sm_edges.set_array([])
        cbar_edges = plt.colorbar(sm_edges, ax=axes[2], fraction=0.03, pad=0.08)
        cbar_edges.set_label("Peso das Arestas")

    # =======================
    # Salvar Figura
    # =======================
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Exam_{exam_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figura salva: {save_path}")

    plt.close()  # Fecha a figura para evitar sobreposição de gráficos


# ================================================
# 4) Visualizar e Salvar Exemplos do Dataset
# ================================================
print("\nGerando e salvando exemplos de ECG e seus grafos...")
for i in range(min(5, len(graphs_by_exam))):  # Exibir até 5 exemplos
    print(f"Processando Exam {i+1}/{min(5, len(graphs_by_exam))} - ID: {list(graphs_by_exam.keys())[i]}")
    plot_ecg_and_graph(list(graphs_by_exam.keys())[i])

print("\nFinalizado! As figuras foram salvas na pasta 'figures/'.")



