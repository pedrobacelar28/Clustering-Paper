import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# =============================================================================
# 1. Definir o mapeamento entre os arquivos e os índices dos rótulos
# =============================================================================
class_files = {
    "umdavb.pt": 0,
    "rbbb.pt":   1,
    "lbbb.pt":   2,
    "sb.pt":     3,
    "st.pt":     4,
    "af.pt":     5,
    "unlabel.pt":6,
}

# Pasta onde os arquivos estão armazenados
dataset_dir = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/"

# =============================================================================
# 2. Carregar os grafos de cada arquivo e construir o dataset do PyG
# =============================================================================
dataset = []

for file_name, label in class_files.items():
    file_path = os.path.join(dataset_dir, file_name)
    try:
        # Carrega os dados salvos com torch.save()
        dados = torch.load(file_path)
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        continue

    # Verifica se os dados estão em formato de dicionário com a chave 'grafos'
    if isinstance(dados, dict) and "grafos" in dados:
        grafos_dict = dados["grafos"]
        for exam_id, graph_info in grafos_dict.items():
            try:
                # === MODIFICAÇÃO AQUI ===
                # Ao invés de usar apenas lead_0, concatenamos as features de todas as 12 leads.
                features_list = [graph_info[f"lead_{i}"].x for i in range(12)]
                combined_features = torch.cat(features_list, dim=1)  # Dimensão: (N, 36)

                # Usa a conectividade de "lead_0" como referência.
                edge_index = graph_info["lead_0"].edge_index

                data = Data(
                    x = combined_features,                
                    edge_index = edge_index,
                    y = torch.tensor(label, dtype=torch.long)
                )
                data.exam_id = exam_id  # Opcional, para referência
                dataset.append(data)
            except KeyError:
                print(f"Alguma das leads não foi encontrada para o exam_id {exam_id} em {file_name}. Ignorando.")
                continue

    # Caso os dados já estejam salvos como uma lista de objetos Data
    elif isinstance(dados, list):
        for data in dados:
            data.y = torch.tensor(label, dtype=torch.long)
            dataset.append(data)
    else:
        print(f"Formato dos dados em {file_name} não reconhecido.")

print(f"Total de amostras no dataset: {len(dataset)}")

# =============================================================================
# 3. Dividir o dataset em treino, validação e teste
# =============================================================================
labels = [data.y.item() for data in dataset]
train_data, temp_data = train_test_split(dataset, test_size=0.30, random_state=42, stratify=labels)
temp_labels = [data.y.item() for data in temp_data]
val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42, stratify=temp_labels)

print(f"Número de amostras de treino: {len(train_data)}")
print(f"Número de amostras de validação: {len(val_data)}")
print(f"Número de amostras de teste: {len(test_data)}")

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# =============================================================================
# 4. Definir o modelo – arquitetura baseada em GIN para classificação exclusiva
# =============================================================================
class GINExclusive(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.2):
        super(GINExclusive, self).__init__()
        # Primeira camada GIN
        self.mlp1 = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)

        # Segunda camada GIN
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)

        # Terceira camada GIN
        self.mlp3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv3 = GINConv(self.mlp3)

        # Camadas finais para converter o embedding global em logits
        self.lin1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.bn = nn.BatchNorm1d(hidden_channels * 2)
        self.lin2 = nn.Linear(hidden_channels * 2, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Agrega os embeddings dos nós de cada grafo em um único vetor
        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)
        return out

# === MODIFICAÇÃO AQUI ===
# Número de features passa de 3 para 36, pois usamos 12 leads concatenadas (12 * 3 = 36)
num_classes = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINExclusive(num_features=36, hidden_channels=128, num_classes=num_classes, dropout=0.2).to(device)

# =============================================================================
# 5. Preparar treinamento e avaliação
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
# 6. Treinar o modelo e plotar as curvas de loss
# =============================================================================
num_epochs = 70
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

plt.savefig("loss_plot.png", dpi=300)
plt.show()

# =============================================================================
# 7. Avaliação no conjunto de teste e cálculo do F1 Score
# =============================================================================
test_loss, test_logits, test_targets = evaluate(test_loader)
print(f"Loss no teste: {test_loss:.4f}")

test_probs = F.softmax(test_logits, dim=1)
test_preds = test_probs.argmax(dim=1)

test_preds_np = test_preds.numpy()
test_targets_np = test_targets.numpy()

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
