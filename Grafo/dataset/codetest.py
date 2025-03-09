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

sys.path.append('/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Pre-processings')  
from filters import ecg_filtrado

# %%
import pandas as pd
import h5py
import numpy as np
import sys

# Adiciona o caminho para importar a função ecg_filtrado
sys.path.append('/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Pre-processing')  
from filters import ecg_filtrado

# Caminhos dos arquivos
csv_path = '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/data/gold_standard.csv'
hdf5_path = '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/data/ecg_tracings.hdf5'

# 1. Carrega o CSV
df = pd.read_csv(csv_path)

# Colunas das anomalias
anomalia_cols = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

# 2. Cria a coluna 'normal': 1 se todas as anomalias forem 0, 0 caso contrário
df['normal'] = (df[anomalia_cols].sum(axis=1) == 0).astype(int)

# (Removido o filtro por quantidade de anomalias)

# Mantemos todo o DataFrame
df_filtered = df.copy()

# Guarda todos os índices dos exames
indices_validos = df.index.tolist()

# 4. Carrega os traçados do arquivo HDF5
with h5py.File(hdf5_path, "r") as f:
    ecg_data = np.array(f['tracings'])

print("Número original de exames:", ecg_data.shape[0])
print("Número de exames mantidos (sem filtro):", len(indices_validos))

# Filtra os traçados (na prática aqui não filtra nada, pois pegamos todos)
ecg_filtrado_array = ecg_data[indices_validos]

# 5. Aplica o pré-processamento (função ecg_filtrado) em cada lead de cada exame
ecg_processed = np.array([
    np.array([ecg_filtrado(lead) for lead in exame.T]).T
    for exame in ecg_filtrado_array
])

# 6. Salva os novos arquivos
# Salva o CSV filtrado (com a coluna 'normal' adicionada)
df_filtered.to_csv('gold_standard_filtered.csv', index=False)
print("Arquivo CSV (sem filtro por quantidade de doenças) salvo como 'gold_standard_filtered.csv'.")

# Salva o dataset processado em um novo arquivo HDF5
with h5py.File('ecg_tracings_filtered.hdf5', "w") as f_out:
    f_out.create_dataset('tracings', data=ecg_processed)
print("Arquivo HDF5 (sem filtro por quantidade de doenças) salvo como 'ecg_tracings_filtered.hdf5'.")


# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Caminho para o arquivo HDF5 filtrado
hdf5_path = '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/ecg_tracings_filtered.hdf5'

# Carregar os traçados do arquivo
with h5py.File(hdf5_path, 'r') as f:
    ecg_data = np.array(f['tracings'])

print(f"Número de exames no arquivo: {ecg_data.shape[0]}")

# Número de exames que queremos plotar (exemplo: 3 exames)
num_exames_plot = 3

# Seleciona aleatoriamente alguns índices para plotar
indices_plot = np.random.choice(ecg_data.shape[0], num_exames_plot, replace=False)

for idx in indices_plot:
    exame = ecg_data[idx]  # Shape: (4096, 12)
    
    # Cria uma figura para o exame atual
    plt.figure(figsize=(12, 18))
    
    # Plota cada uma das 12 derivações em um subplot diferente
    for lead in range(12):
        plt.subplot(12, 1, lead + 1)
        plt.plot(exame[:, lead])
        plt.ylabel(f"Lead {lead+1}")
        if lead == 0:
            plt.title(f"Exame ID: {idx}")
        if lead < 11:
            plt.xticks([])  # Remove os ticks do eixo x para subplots intermediários
    plt.xlabel("Amostras")
    plt.tight_layout()
    plt.show()


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
      Se True, carrega de '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/ecg_tracings_filtered.hdf5'
      Se False, de '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/data/ecg_tracings.hdf5'

    Retorna:
      X        : array com os traçados (N, 12, num_amostras)
      ids_ecgs : lista de exam_ids selecionados
      labels   : array (N, 6) com as labels [1dAVb, RBBB, LBBB, SB, ST, AF]
    """
    csv_path = '/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest/gold_standard_filtered.csv'
    dados = pd.read_csv(csv_path)
    
    # Se a coluna 'exam_id' não existir, cria-a com números aleatórios
    if 'exam_id' not in dados.columns:
        dados.insert(0, 'exam_id', [random.randint(10000, 99999) for _ in range(len(dados))])
        dados.to_csv(csv_path, index=False)
        print("Coluna 'exam_id' criada e CSV atualizado.")
    
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

    # Combina os exam_ids selecionados. Note que as amostras multilabel são adicionadas de forma separada.
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
        arquivos_hdf5 = ['/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest/ecg_tracings_filtered.hdf5']
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

#

# %%
# Exemplo de chamada da função:
X, ids_ecgs, labels = carregar_ecgs_goldstandard(
     unlabel=681,    unlabel_offset=0,
     umdavb=20,      umdavb_offset=0,
     rbbb=28,       rbbb_offset=0,
     lbbb=25,        lbbb_offset=0,
     sb=15,          sb_offset=0,
     st=37,          st_offset=0,
     af=11,         af_offset=0,
     multilabel=12,  multilabel_offset=0,
     filtrado=True
)

# %%
# Vamos imprimir um exemplo de 10 exames, mostrando seu exam_id e respectivo label
for i in range(min(100, len(ids_ecgs))):
    print(f"Exam ID: {ids_ecgs[i]}, Label: {labels[i]}")


# %%
import torch
import numpy as np
import neurokit2 as nk
import networkx as nx  # Para calcular PageRank
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
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.int64)
        # Concatena as features de todas as leads (eixo das colunas)
        node_features = np.hstack(features_list)  # Resultado: (1000, 48)
        valid = True

    data = Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=edge_index)
    return exam_id, data, label, valid

if __name__ == '__main__':
    # SUPOSIÇÕES:
    #  - X, ids_ecgs e labels estão definidos e têm mesmo tamanho N.
    #  - X: (N, 12, num_amostras)
    #  - ids_ecgs: lista com N exam_ids
    #  - labels: array/list com as N labels

    exam_ids_list = ids_ecgs
    labels_list   = labels

    print("Iniciando a criação dos grafos de visibilidade para cada ECG (armazenando apenas a lead1 com 48 features)...")

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_exam)(ecg, exam_ids_list[idx], labels_list[idx])
        for idx, ecg in enumerate(tqdm(X, desc="Processando exames"))
    )

    graphs_by_exam = {}
    count_invalid = 0
    for (exam_id, data, lbl, valid) in results:
        graphs_by_exam[exam_id] = {
            "grafo": data,  # Apenas a lead1, com features concatenadas de todas as 12 leads
            "label": lbl
        }
        if not valid:
            count_invalid += 1

    dados_salvos = {"grafos": graphs_by_exam}

    output_filename = "codetest.pt"
    torch.save(dados_salvos, output_filename)
    print(f"\nGrafos (com labels e 48 features na lead1) salvos em {output_filename}")
    print(f"Quantidade de exames que não possuem 1000 pontos: {count_invalid}")

    # Carregar o arquivo salvo e exibir 5 exemplos de exames
    loaded_data = torch.load(output_filename, weights_only=False)
    exam_keys = list(loaded_data["grafos"].keys())
    print("\nExemplos de 5 exames:")
    for key in exam_keys[:5]:
        exame = loaded_data["grafos"][key]
        print(f"Exam ID: {key}")
        print(f"Label: {exame['label']}")
        print(f"Grafo (lead1) Data:")
        print(f"  x shape: {exame['grafo'].x.shape}")
        print(f"  edge_index shape: {exame['grafo'].edge_index.shape}\n")


# %%
import torch

# Caminho do arquivo .pt (ajuste conforme necessário)
file_path = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest.pt"

try:
    # Carrega o arquivo com o checkpoint completo (weights_only=False)
    dados = torch.load(file_path, weights_only=False)
except Exception as e:
    print(f"Erro ao carregar o arquivo {file_path}: {e}")
    exit()

# Verifica o formato dos dados e conta os exames
if isinstance(dados, dict) and "grafos" in dados:
    exames = dados["grafos"]
    print(f"Número de exames (dict): {len(exames)}")
elif isinstance(dados, list):
    print(f"Número de exames (lista): {len(dados)}")
else:
    print("Formato dos dados não reconhecido.")


# %%
import torch

def obter_informacoes_exame(exam_id, pt_file="codetest.pt"):
    """
    Carrega o arquivo .pt, localiza o exam_id e retorna:
      - label associada ao exame
      - dicionário de grafos das 12 leads (cada lead é um objeto Data do PyG)
    """
    # Carrega o conteúdo do .pt
    dados = torch.load(pt_file, weights_only=False)

    # 'dados' é um dicionário contendo "grafos": {exam_id: {"grafos": dict_de_leads, "label": label}}
    if "grafos" not in dados:
        print("Formato de arquivo inesperado. Chave 'grafos' não encontrada.")
        return None, None

    # Tenta recuperar o dicionário para o exam_id desejado
    info_exame = dados["grafos"].get(exam_id)
    if info_exame is None:
        print(f"Exam ID {exam_id} não encontrado no arquivo.")
        return None, None

    # Pega a label e o dicionário das 12 leads
    label_exame = info_exame["label"]
    grafos_12_leads = info_exame["grafos"]

    return label_exame, grafos_12_leads


# ======================
# Exemplo de uso:
# ======================
if __name__ == "__main__":
    meu_exam_id = 138  # substitua pelo ID que você quer inspecionar
    label, grafos = obter_informacoes_exame(meu_exam_id, pt_file="codetest.pt")

    if grafos is not None:
        print(f"Label do exame {meu_exam_id}:", label)
        print("Leads disponíveis:", list(grafos.keys()))
        
        # Podemos inspecionar a estrutura de uma lead específica (ex: lead_0)
        lead0_data = grafos["lead_0"]  # objeto Data do PyTorch Geometric
        print("Nó x shape:", lead0_data.x.shape)          # (num_nós, 3) -> amplitude, derivada, grau
        print("Edge index shape:", lead0_data.edge_index.shape)  # (2, num_arestas)
        print("Exemplo de 'x':\n", lead0_data.x[:250])      # 5 primeiras linhas das features



