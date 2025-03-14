# %% IMPORTS
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
from torch_geometric.utils import to_networkx  # Visualização
from torch.nn import Linear                   # Define layers
from torch_geometric.nn import GCNConv
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d 
from scipy.signal import medfilt
from joblib import Parallel, delayed
from tqdm import tqdm

# Escolha o formato de salvamento: "pt" ou "hdf5"
SAVE_FORMAT = "pt"  # ou "hdf5"

# %% Função para carregar ECGs
def carregar_ecgs(unlabel, umdavb, rbbb, lbbb, sb, st, af, multilabel,
                  unlabel_offset=0, umdavb_offset=0, rbbb_offset=0,
                  lbbb_offset=0, sb_offset=0, st_offset=0, af_offset=0, multilabel_offset=0,
                  filtrado=False):
    """
    Carrega os ECGs e retorna:
      - X: array numpy com os traçados de ECG, shape (N, 12, num_amostras_por_sinal)
      - ids_ecgs: lista com os exam_id correspondentes
      - labels: array numpy de shape (N, 6)
    """
    caminho_arquivo = "../../Projeto/Database/exams.csv"
    dados = pd.read_csv(caminho_arquivo)

    # Lista de arquivos HDF5 usados
    arquivos_usados = [
        "exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5",
        "exams_par4.hdf5",  "exams_part5.hdf5", "exams_part6.hdf5", "exams_part7.hdf5",
        "exams_par8.hdf5",  "exams_part9.hdf5", "exams_part10.hdf5", "exams_part11.hdf5",
        "exams_part12.hdf5","exams_part13.hdf5","exams_part14.hdf5","exams_part15.hdf5",
        "exams_part16.hdf5","exams_part17.hdf5"
    ]

    # 1) Filtrar os exames (conforme as condições)
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
    bool_sum = (dados.iloc[:, 4].astype(int) +
                dados.iloc[:, 5].astype(int) +
                dados.iloc[:, 6].astype(int) +
                dados.iloc[:, 7].astype(int) +
                dados.iloc[:, 8].astype(int) +
                dados.iloc[:, 9].astype(int))
    ecg_multilabel_linhas = dados.index[
        (dados.iloc[:, 14].isin(arquivos_usados)) &
        (bool_sum >= 2) &
        (dados.iloc[:, 13] == False)
    ]

    print("Número de linhas ecg_normal_linhas:", len(ecg_normal_linhas))

    # 2) Excluir exames com interferência
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

    # 3) Obter exam_id para cada grupo
    ecg_normal_id      = dados.iloc[ecg_normal_linhas, 0].tolist()
    ecg_umdavb_id      = dados.iloc[ecg_umdavb_linhas, 0].tolist()
    ecg_rbbb_id        = dados.iloc[ecg_rbbb_linhas, 0].tolist()
    ecg_lbbb_id        = dados.iloc[ecg_lbbb_linhas, 0].tolist()
    ecg_sb_id          = dados.iloc[ecg_sb_linhas, 0].tolist()
    ecg_st_id          = dados.iloc[ecg_st_linhas, 0].tolist()
    ecg_af_id          = dados.iloc[ecg_af_linhas, 0].tolist()
    ecg_multilabel_id  = dados.iloc[ecg_multilabel_linhas, 0].tolist()

    # 4) Selecionar exames via slicing com offset
    def slice_ids(id_list, offset, count):
        if offset >= len(id_list):
            return []
        return id_list[offset : offset + count]

    ecg_normal_sample     = slice_ids(ecg_normal_id,     unlabel_offset,    unlabel)
    ecg_umdavb_sample     = slice_ids(ecg_umdavb_id,     umdavb_offset,     umdavb)
    ecg_rbbb_sample       = slice_ids(ecg_rbbb_id,       rbbb_offset,       rbbb)
    ecg_lbbb_sample       = slice_ids(ecg_lbbb_id,       lbbb_offset,       lbbb)
    ecg_sb_sample         = slice_ids(ecg_sb_id,         sb_offset,         sb)
    ecg_st_sample         = slice_ids(ecg_st_id,         st_offset,         st)
    ecg_af_sample         = slice_ids(ecg_af_id,         af_offset,         af)
    ecg_multilabel_sample = slice_ids(ecg_multilabel_id, multilabel_offset, multilabel)

    # 5) Combinar todos os exam_ids
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

    # 6) Selecionar caminhos dos arquivos HDF5 (filtrado ou não)
    if filtrado:
        arquivos_hdf5 = [
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_0_1.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_2_3.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_4_5.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_6_7.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_8_9.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_10_11.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_12_13.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_14_15.hdf5",
            "/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/filtered_exams_16_17.hdf5"
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

    # 7) Função auxiliar para ler um exame em um arquivo HDF5
    def get_ecg_data(file_path, exam_id):
        with h5py.File(file_path, 'r') as f:
            exam_ids = np.array(f['exam_id'])
            exam_index = np.where(exam_ids == exam_id)[0]
            if len(exam_index) == 0:
                raise ValueError("Exam ID não encontrado.")
            exam_index = exam_index[0]
            exam_tracings = f['tracings'][exam_index]
            return exam_tracings

    # 8) Carregar os traçados
    all_tracings = []
    for exam_id in ids_ecgs:
        found = False
        for arquivo in arquivos_hdf5:
            try:
                tracings = get_ecg_data(arquivo, exam_id)
                if tracings is not None:
                    tracing_transposto = np.array(tracings).T  # shape (12, n_amostras)
                    all_tracings.append(tracing_transposto)
                    found = True
                    break
            except ValueError:
                pass
            except Exception as e:
                pass
        if not found:
            print(f"Erro: exame ID {exam_id} não encontrado em nenhum dos arquivos.")

    print("\nNúmero de ECGs processados:", len(ids_ecgs))
    print(f"Número total de traçados carregados: {len(all_tracings)}")

    # 9) Montar X e as labels
    X = np.array(all_tracings)  # shape (N, 12, num_amostras)
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

# %% Carregamento dos ECGs
n = 0
if n == 0:
    X, ids_ecgs, labels = carregar_ecgs(unlabel=50000, umdavb=3651, rbbb=6703, lbbb=4122, sb=4248, st=6038, af=4804, multilabel=3169,
                                        unlabel_offset=0, umdavb_offset=0, rbbb_offset=0,
                                        lbbb_offset=0, sb_offset=0, st_offset=0, af_offset=0, multilabel_offset=0,
                                        filtrado=True)
else:
    X, ids_ecgs, labels = carregar_ecgs(unlabel=0, umdavb=0, rbbb=0, lbbb=0, sb=0, st=0, af=0, multilabel=0,
                                        unlabel_offset=0, umdavb_offset=0, rbbb_offset=0,
                                        lbbb_offset=0, sb_offset=0, st_offset=0, af_offset=0, multilabel_offset=0,
                                        filtrado=True)

# %% Exibir um exemplo
for i in range(min(1, len(ids_ecgs))):
    print(f"Exam ID: {ids_ecgs[i]}, Label: {labels[i]}")

import torch
import numpy as np
import neurokit2 as nk
import networkx as nx
from ts2vg import NaturalVG
from torch_geometric.data import Data

def get_middle_r_peak(lead_series, sampling_rate=400):
    peaks_dict = nk.ecg_findpeaks(lead_series, sampling_rate=sampling_rate)
    peaks = np.array(peaks_dict["ECG_R_Peaks"])
    if peaks.size == 0:
        return len(lead_series) // 2
    if len(peaks) % 2 == 0:
        return peaks[len(peaks) // 2 - 1]
    else:
        return peaks[len(peaks) // 2]

def process_exam(ecg, exam_id, label, scaler=None):
    """
    Processa um ECG (12 leads) e retorna:
      - exam_id
      - um grafo baseado na lead1 para conectividade, com features combinadas de leads 0, 1, 6 e 11
      - a label associada.
    """
    lead1_series = ecg[1]
    r_peak = get_middle_r_peak(lead1_series, sampling_rate=400)
    start_index = max(0, r_peak - 500)
    end_index = min(len(lead1_series), r_peak + 500)
    n = 1000

    if (end_index - start_index) != n:
        node_features = np.zeros((n, 28))
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        valid = False
    else:
        lead1_segment = ecg[1][start_index:end_index]
        vg = NaturalVG()
        vg.build(lead1_segment)
        edges = vg.edges
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.int64)
        
        leads_indices = [0, 1, 6, 11]
        feature_list = []
        for lead in leads_indices:
            segment = ecg[lead][start_index:end_index]
            amplitude = segment.reshape(-1, 1)
            derivative = np.concatenate(([0], np.diff(segment))).reshape(-1, 1)
            vg_lead = NaturalVG()
            vg_lead.build(segment)
            edges_lead = vg_lead.edges
            if edges_lead:
                edges_lead_array = np.array(edges_lead)
                counts = np.bincount(np.concatenate([edges_lead_array[:, 0], edges_lead_array[:, 1]]), minlength=n)
                degree = counts.reshape(-1, 1).astype(float)
            else:
                degree = np.zeros((n, 1))
            one_hot = np.zeros((n, 4))
            one_hot[:, leads_indices.index(lead)] = 1.0
            features_this_lead = np.hstack([amplitude, derivative, degree, one_hot])
            feature_list.append(features_this_lead)
        
        node_features = np.hstack(feature_list)
        valid = True

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
    # Primeira passagem (sem normalização)
    results_temp = Parallel(n_jobs=20, verbose=10)(
        delayed(process_exam)(ecg, ids_ecgs[idx], labels[idx], scaler=None)
        for idx, ecg in enumerate(tqdm(X, desc="Processando exames (temp)"))
    )
    
    all_features = []
    for exam_id, data, lbl, valid in results_temp:
        if valid:
            all_features.append(data.x.numpy())
    all_features = np.vstack(all_features)
    
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
    
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_exam)(ecg, ids_ecgs[idx], labels[idx], scaler=scaler)
        for idx, ecg in enumerate(tqdm(X, desc="Processando exames (normalizados)"))
    )
    
    graphs_by_exam = {}
    count_invalid = 0
    for exam_id, data, lbl, valid in results:
        graphs_by_exam[exam_id] = {"grafo": data, "label": lbl}
        if not valid:
            count_invalid += 1

    # Salvar com base na opção escolhida
    if SAVE_FORMAT == "pt":
        output_filename = "normalized.pt"
        dados_salvos = {"grafos": graphs_by_exam}
        torch.save(dados_salvos, output_filename)
        print(f"\nGrafos salvos em {output_filename} (torch .pt)")
    else:
        output_filename = "normalized2.hdf5"
        with h5py.File(output_filename, "w") as f:
            grp = f.create_group("grafos")
            for exam_id, content in graphs_by_exam.items():
                exam_grp = grp.create_group(str(exam_id))
                exam_grp.create_dataset("x", data=content["grafo"].x.numpy())
                exam_grp.create_dataset("edge_index", data=content["grafo"].edge_index.numpy())
                exam_grp.create_dataset("label", data=np.array(content["label"]))
        print(f"\nGrafos salvos em {output_filename} (HDF5)")

    print(f"Quantidade de exames com segmento inválido (≠ 1000 pontos): {count_invalid}")
