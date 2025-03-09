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
import matplotlib.pyplot as plt
from torch.nn import Linear                   # Define layers
from torch_geometric.nn import GCNConv
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d 
from scipy.signal import medfilt

# %%
import pandas as pd
import numpy as np
import random
import h5py

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




# %%

X, ids_ecgs, labels = carregar_ecgs(unlabel=20000,umdavb=0, rbbb=0, lbbb=0, sb=0, st=0, af=0, multilabel=0,unlabel_offset=20000, umdavb_offset=0, rbbb_offset=0,
                                    lbbb_offset=0, sb_offset=0, st_offset=0, af_offset=0, multilabel_offset=0,filtrado=True)


# %%

# Vamos imprimir um exemplo de 10 exames, mostrando seu exam_id e respectivo label
for i in range(min(1, len(ids_ecgs))):
    print(f"Exam ID: {ids_ecgs[i]}, Label: {labels[i]}")


# %%
import torch
import numpy as np
import neurokit2 as nk
import networkx as nx  # <- Import para calcular PageRank
from ts2vg import NaturalVG
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm

def compute_node_features(time_series, edges):
    """
    Calcula as features de cada nó:
      - amplitude: valor da amostra;
      - derivada: diferença com o nó anterior (primeiro nó = 0);
      - grau: número de arestas incidentes;
      - pagerank: valor de pagerank calculado via networkx.
    
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


def process_lead(lead_series):
    """
    Processa uma única lead:
      - Detecta o pico R "do meio" (NeuroKit).
      - Seleciona 1000 pontos centrados no R (500 antes, 500 depois).
      - Constrói o grafo de visibilidade (ts2vg.NaturalVG).
      - Calcula as features [amplitude, derivada, grau, pagerank].
      - Retorna Data(x=node_features, edge_index).
    """
    r_peak = get_middle_r_peak(lead_series, sampling_rate=400)

    start_index = max(0, r_peak - 500)
    end_index = min(len(lead_series), r_peak + 500)
    lead_segment = lead_series[start_index:end_index]

    vg = NaturalVG()
    vg.build(lead_segment)
    edges = vg.edges  # lista de arestas (i, j)
    
    node_features = compute_node_features(lead_segment, edges)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)

    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    return Data(x=node_features_tensor, edge_index=edge_index)


def process_exam(ecg, exam_id, label):
    """
    Processa um ECG (12 leads) e retorna:
      - exam_id
      - dicionário com 12 grafos
      - label associada a esse exame.
    """
    exam_graphs = {
        f"lead_{lead_index}": process_lead(ecg[lead_index])
        for lead_index in range(12)
    }
    return exam_id, exam_graphs, label


if __name__ == '__main__':
    # ==========================================================================
    # SUPOSIÇÕES:
    #  - X, ids_ecgs e labels estão definidos e têm mesmo tamanho N.
    #  - X: (N, 12, num_amostras)
    #  - ids_ecgs: lista com N exam_ids
    #  - labels: array/list com as N labels
    # ==========================================================================
    exam_ids_list = ids_ecgs
    labels_list   = labels

    print("Iniciando a criação dos grafos de visibilidade para cada ECG e cada lead...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_exam)(ecg, exam_ids_list[idx], labels_list[idx])
        for idx, ecg in enumerate(tqdm(X, desc="Processando exames"))
    )

    graphs_by_exam = {
        exam_id: {
            "grafos": exam_graphs,  # as 12 leads como Data PyG
            "label": lbl
        }
        for (exam_id, exam_graphs, lbl) in results
    }

    dados_salvos = {"grafos": graphs_by_exam}

    output_filename = "parte4.pt"
    torch.save(dados_salvos, output_filename)
    print(f"Grafos (com labels e PageRank) salvos em {output_filename}")

