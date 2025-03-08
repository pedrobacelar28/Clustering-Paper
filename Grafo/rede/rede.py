import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# =============================================================================
# 1. Definir o mapeamento entre os arquivos e os índices dos rótulos
# =============================================================================
# Supondo a seguinte correspondência:
# "umdavb.pt" -> 0 (equivalente a "1dAVb")
# "rbbb.pt"   -> 1 ("RBBB")
# "lbbb.pt"   -> 2 ("LBBB")
# "sb.pt"     -> 3 ("SB")
# "st.pt"     -> 4 ("ST")
# "af.pt"     -> 5 ("AF")
# "unlabel.pt"-> 6 ("normal")
class_files = {
    "umdavb.pt": 0,
    "rbbb.pt":   1,
    "lbbb.pt":   2,
    "sb.pt":     3,
    "st.pt":     4,
    "af.pt":     5,
    "unlabel.pt":6,
}

# =============================================================================
# 2. Implementação do Dataset utilizando InMemoryDataset (alterado)
# =============================================================================
class ECGDataset(InMemoryDataset):
    def __init__(self, root, class_files, force_process=False, transform=None, pre_transform=None):
        self.class_files = class_files
        self.force_process = force_process

        # Define o caminho do arquivo de cache no diretório atual
        processed_path = os.path.join(os.getcwd(), 'data.pt')
        if self.force_process and os.path.exists(processed_path):
            os.remove(processed_path)
            print("Arquivo de cache removido. Reprocessamento forçado.")

        super(ECGDataset, self).__init__(root, transform, pre_transform)
        # Carrega os dados processados (cache) do diretório atual
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        # Os arquivos brutos devem estar na pasta raw (mesmo diretório especificado por 'root/raw')
        return list(self.class_files.keys())
    
    @property
    def processed_file_names(self):
        # Nome do arquivo cacheado com os dados processados
        return ['data.pt']
    
    @property
    def processed_dir(self):
        # Sobrescreve o diretório padrão de dados processados para ser o diretório atual
        return os.getcwd()
    
    @property
    def processed_paths(self):
        # Define os caminhos dos arquivos processados com base no processed_dir (diretório atual)
        return [os.path.join(self.processed_dir, self.processed_file_names[0])]
    
    def download(self):
        # Se os arquivos brutos já estiverem no diretório, nada a fazer
        pass
    
    def process(self):
        dataset = []
        raw_dir = self.root
        for file_name, label in self.class_files.items():
            file_path = os.path.join(raw_dir, file_name)
            try:
                # Carrega o arquivo .pt
                dados = torch.load(file_path)
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")
                continue

            # Se os grafos estiverem dentro de um dicionário com a chave 'grafos'
            if isinstance(dados, dict) and "grafos" in dados:
                grafos_dict = dados["grafos"]
                for exam_id, graph_info in grafos_dict.items():
                    try:
                        # Usamos a estrutura (edge_index) da lead_0
                        graph0 = graph_info["lead_0"]
                        # Número de nós esperado baseado na lead_0
                        num_nodes_expected = graph0.x.shape[0]
                        # Lista para armazenar as features de cada lead (supondo 12 leads: lead_0 a lead_11)
                        features_list = []
                        for i in range(12):
                            key = f"lead_{i}"
                            if key not in graph_info:
                                raise KeyError(f"{key} não encontrado")
                            # Extrai as features (assumindo que cada lead possui um tensor .x com shape [num_nodes, 3])
                            current_features = graph_info[key].x
                            # Se o número de nós não bater com o esperado, substitui por tensor de zeros
                            if current_features.shape[0] != num_nodes_expected:
                                print(f"Tamanho inesperado para {key} em exam_id {exam_id} ({current_features.shape[0]} != {num_nodes_expected}). Substituindo por zeros.")
                                current_features = torch.zeros(num_nodes_expected, current_features.shape[1],
                                                               dtype=current_features.dtype, device=current_features.device)
                            features_list.append(current_features)
                        # Concatena as features de todas as leads ao longo da dimensão das features
                        # Resultado: cada nó terá 3 * 12 = 36 features
                        x_concat = torch.cat(features_list, dim=1)
                    except KeyError as e:
                        print(f"Erro: {e} para exam_id {exam_id} em {file_name}. Ignorando.")
                        continue

                    data = Data(
                        x = x_concat,                    # features concatenadas de todas as leads
                        edge_index = graph0.edge_index,    # estrutura (conectividade) da lead_0
                        y = torch.tensor(label, dtype=torch.long)
                    )
                    data.exam_id = exam_id  # opcional, para referência
                    dataset.append(data)
            # Se os dados já forem uma lista de objetos Data (caso não haja dicionário com 'grafos')
            elif isinstance(dados, list):
                for data in dados:
                    data.y = torch.tensor(label, dtype=torch.long)
                    dataset.append(data)
            else:
                print(f"Formato dos dados em {file_name} não reconhecido.")
        
        data, slices = self.collate(dataset)
        # Salva o dataset processado no arquivo de cache (diretório atual)
        torch.save((data, slices), self.processed_paths[0])






# Defina o diretório raiz para o dataset.
# Estrutura esperada:
#   ../dataset/
#       raw/   -> aqui devem estar os arquivos: "umdavb.pt", "rbbb.pt", etc.
#       processed/ -> será criado automaticamente
root_dir = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset"

# Cria o dataset utilizando a classe customizada (alterado)
dataset = ECGDataset(root=root_dir, class_files=class_files, force_process=False)
print(f"Total de amostras no dataset: {len(dataset)}")

# =============================================================================
# 3. Dividir o dataset em treino, validação e teste (inalterado)
# =============================================================================
train_data, temp_data = train_test_split(dataset, test_size=0.10, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)

print(f"Número de amostras de treino: {len(train_data)}")
print(f"Número de amostras de validação: {len(val_data)}")
print(f"Número de amostras de teste: {len(test_data)}")

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# =============================================================================
# 4. Definir o modelo – arquitetura baseada em GIN para classificação exclusiva (inalterado)
# =============================================================================
class GINExclusive(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.2):
        super(GINExclusive, self).__init__()
        # Projeção residual para ajustar a dimensão da entrada na primeira camada
        self.res_conv1 = nn.Linear(num_features, hidden_channels) if num_features != hidden_channels else nn.Identity()

        # --- Bloco 1 ---
        self.mlp1 = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # --- Bloco 2 ---
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # --- Bloco 3 ---
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv3 = GINConv(self.mlp3)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        # --- Bloco 4 ---
        self.mlp4 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv4 = GINConv(self.mlp4)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        # Agregação com Jumping Knowledge: concatena as saídas dos 4 blocos e reduz para hidden_channels
        self.lin_jump = nn.Linear(hidden_channels * 4, hidden_channels)

        # Camadas finais para classificação
        self.lin1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.bn_final = nn.BatchNorm1d(hidden_channels * 2)
        self.lin2 = nn.Linear(hidden_channels * 2, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # --- Bloco 1 ---
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1 + self.res_conv1(x))
        x1 = self.bn1(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # --- Bloco 2 ---
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2 + x1)
        x2 = self.bn2(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # --- Bloco 3 ---
        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3 + x2)
        x3 = self.bn3(x3)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        # --- Bloco 4 ---
        x4 = self.conv4(x3, edge_index)
        x4 = F.relu(x4 + x3)
        x4 = self.bn4(x4)
        # Neste bloco, optamos por não aplicar dropout para preservar a informação para o Jumping Knowledge

        # --- Jumping Knowledge ---
        # Concatena as saídas de todos os blocos
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x_cat = F.relu(self.lin_jump(x_cat))

        # Agregação global dos nós (global mean pooling)
        x_global = global_mean_pool(x_cat, batch)

        # Camadas finais para a classificação
        x_global = self.lin1(x_global)
        x_global = self.bn_final(x_global)
        x_global = F.relu(x_global)
        x_global = F.dropout(x_global, p=self.dropout, training=self.training)
        out = self.lin2(x_global)
        return out




# Número de classes: 7 (conforme o mapeamento definido)
num_classes = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINExclusive(num_features=36, hidden_channels=128, num_classes=num_classes, dropout=0.2).to(device)

# =============================================================================
# 5. Preparar treinamento e avaliação (inalterado)
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_epoch(loader):
    model.train()
    epoch_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * data.num_graphs
    return epoch_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    total_loss = 0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            all_logits.append(out.cpu())
            all_targets.append(data.y.cpu())
    avg_loss = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return avg_loss, all_logits, all_targets

# =============================================================================
# 6. Treinar o modelo e plotar as curvas de loss (inalterado)
# =============================================================================
num_epochs = 50
train_losses = []
val_losses = []

for epoch in tqdm(range(1, num_epochs+1), desc="Treinamento"):
    train_loss = train_epoch(train_loader)
    val_loss, _, _ = evaluate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, label="Treino")
plt.plot(range(1, num_epochs+1), val_losses, label="Validação")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de Loss de Treino e Validação")
plt.legend()
plt.grid(True)
plt.savefig("curva_loss.png")  # Salva a figura
# plt.show() pode ser mantido se o ambiente suportar exibição


# =============================================================================
# 7. Avaliação no conjunto de teste e cálculo do F1 Score (inalterado)
# =============================================================================
test_loss, test_logits, test_targets = evaluate(test_loader)
print(f"Loss no teste: {test_loss:.4f}")

test_probs = F.softmax(test_logits, dim=1)
test_preds = test_probs.argmax(dim=1)

test_preds_np = test_preds.numpy()
test_targets_np = test_targets.numpy()

# Mapeamento dos índices para nomes das classes (para exibição)
classes_list = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal"]
f1_per_class = {}
for i, cls in enumerate(classes_list):
    f1 = f1_score(test_targets_np == i, test_preds_np == i, zero_division=1)
    f1_per_class[cls] = f1

print("\nF1 Score por classe:")
for cls in classes_list:
    print(f"{cls}: {f1_per_class[cls]:.4f}")

f1_macro = f1_score(test_targets_np, test_preds_np, average='macro', zero_division=1)
print(f"\nF1 Macro: {f1_macro:.4f}")

num_examples = 5
print("\nExemplos do conjunto de teste:")
for i in range(num_examples):
    print(f"Exemplo {i+1}:")
    print("  Rótulo verdadeiro: ", test_targets_np[i])
    print("  Predição (classe): ", test_preds_np[i])
    print("  Probabilidades: ", test_probs[i].detach().numpy())
