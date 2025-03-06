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
# Supondo a seguinte correspondência:
# "umdavb.pt" -> 0 (equivalente a "1dAVb")
# "rbbb.pt"   -> 1 ("RBBB")
# "lbbb.pt"   -> 2 ("LBBB")
# "sf.pt"     -> 3 ("SB")     # Se "sf" equivale a "SB"
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

# Pasta onde os arquivos estão armazenados
dataset_dir = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/"  # ajuste o caminho conforme necessário

# =============================================================================
# 2. Carregar os grafos de cada arquivo e construir o dataset do PyG
# =============================================================================
dataset = []

for file_name, label in class_files.items():
    file_path = os.path.join(dataset_dir, file_name)
    try:
        # Supondo que cada arquivo seja salvo com torch.save() e contenha um dicionário
        # com a chave 'grafos' que é, por exemplo, outro dicionário ou uma lista.
        dados = torch.load(file_path)
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        continue

    # Dependendo de como os grafos estão salvos, adapte a iteração.
    # Se os grafos estiverem armazenados em um dicionário, por exemplo:
    if isinstance(dados, dict) and "grafos" in dados:
        grafos_dict = dados["grafos"]
        for exam_id, graph_info in grafos_dict.items():
            try:
                # Se cada "graph_info" for um dicionário com chaves (por exemplo, "lead_0")
                # você pode escolher qual lead utilizar.
                graph = graph_info["lead_0"]
            except KeyError:
                print(f"Lead_0 não encontrado para o exam_id {exam_id} em {file_name}. Ignorando.")
                continue

            data = Data(
                x = graph.x,                 # features dos nós
                edge_index = graph.edge_index,  # conectividade
                y = torch.tensor(label, dtype=torch.long)
            )
            data.exam_id = exam_id  # opcional, para referência
            dataset.append(data)
    # Se os dados já forem uma lista de objetos Data, você pode iterar diretamente
    elif isinstance(dados, list):
        for data in dados:
            # Certifique-se de que cada objeto Data possua os atributos necessários
            data.y = torch.tensor(label, dtype=torch.long)
            dataset.append(data)
    else:
        print(f"Formato dos dados em {file_name} não reconhecido.")

print(f"Total de amostras no dataset: {len(dataset)}")

# =============================================================================
# 3. Dividir o dataset em treino, validação e teste
# =============================================================================
# Criamos uma lista de rótulos (labels) extraída de cada objeto Data.
labels = [data.y.item() for data in dataset]

# Divide o dataset em treino (70%) e um conjunto temporário (30%), estratificando pela classe
train_data, temp_data = train_test_split(dataset, test_size=0.30, random_state=42, stratify=labels)

# Cria a lista de rótulos para o conjunto temporário
temp_labels = [data.y.item() for data in temp_data]

# Divide o conjunto temporário igualmente em validação (15%) e teste (15%), mantendo o balanceamento
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

# Número de classes: 7 (conforme o mapeamento definido)
num_classes = 7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINExclusive(num_features=3, hidden_channels=128, num_classes=num_classes, dropout=0.2).to(device)

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
num_epochs = 100
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

# Salva a figura como um arquivo PNG com 300 dpi de resolução
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
