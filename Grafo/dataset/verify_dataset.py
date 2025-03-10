import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import os

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

def plot_ecg_and_graph(exam_id):
    """
    Plota e salva um exemplo de ECG e a estrutura do grafo correspondente.
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

    # ======= Criar Figura =======
    plt.figure(figsize=(12, 5))
    
    # ======= Subplot 1: Sinal de ECG =======
    plt.subplot(1, 2, 1)
    plt.plot(grafo.x[:, 0].numpy(), label="Amplitude (Lead1)")
    plt.xlabel("Amostra")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"ECG - Exam ID {exam_id}")
    plt.legend()
    
    # ======= Subplot 2: Grafo =======
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(G)  # Layout para visualização
    edge_weights = [d['edge_attr'] for _, _, d in G.edges(data=True)]
    
    nx.draw(G, pos, with_labels=False, node_size=30, edge_color=edge_weights, edge_cmap=plt.cm.Blues)
    plt.title(f"Grafo - Exam ID {exam_id}")
    
    # ======= Salvar Figura =======
    save_path = os.path.join(output_dir, f"Exam_{exam_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figura salva: {save_path}")
    
    plt.close()  # Fecha a figura para evitar sobreposição de gráficos

# ================================================
# 4) Visualizar alguns exemplos do dataset
# ================================================

print("\nGerando e salvando exemplos de ECG e seus grafos...")
for i in range(min(5, len(exam_keys))):  # Exibir até 5 exemplos
    print(f"Processando Exam {i+1}/{min(5, len(exam_keys))} - ID: {exam_keys[i]}")
    plot_ecg_and_graph(exam_keys[i])

print("\nFinalizado! As figuras foram salvas na pasta 'figures/'.")