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
# 1. Definir os mapeamentos entre os arquivos e os índices dos rótulos
# =============================================================================
# Arquivos para treino e validação
class_files_train = {
    "umdavb.pt": 0,
    "rbbb.pt":   1,
    "lbbb.pt":   2,
    "sb.pt":     3,
    "st.pt":     4,
    "af.pt":     5,
    "unlabel.pt":6,
}

# Arquivos para teste
class_files_test = {
    "umdavbtest.pt": 0, 
    "rbbbtest.pt":   1, 
    "lbbbtest.pt":   2, 
    "sbtest.pt":     3, 
    "sttest.pt":     4, 
    "aftest.pt":     5, 
    "unlabeltest.pt":6,
}

# =============================================================================
# 2. Implementação do Dataset utilizando InMemoryDataset
# =============================================================================
class ECGDataset(InMemoryDataset):
    def __init__(self, root, class_files, split="train", transform=None, pre_transform=None):
        self.class_files = class_files
        self.split = split  # 'train_val' ou 'test'
        super(ECGDataset, self).__init__(root, transform, pre_transform)
        # Permitir o carregamento seguro dos globals: DataEdgeAttr, DataTensorAttr e GlobalStorage
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        # Carrega os dados processados (cache)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        # Os arquivos brutos devem estar na pasta raw (ou ajuste se necessário)
        return list(self.class_files.keys())
    
    @property
    def processed_file_names(self):
        # O nome do arquivo cacheado difere conforme o split para evitar conflito
        return [f'/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/processed/{self.split}_data.pt']
    
    def download(self):
        # Se os arquivos brutos já estiverem no diretório, nada a fazer
        pass
    
    def process(self):
        dataset = []
        raw_dir = self.root  # Se os arquivos estiverem diretamente em self.root

        # Importar os globals necessários
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage

        for file_name, label in self.class_files.items():
            file_path = os.path.join(raw_dir, file_name)
            try:
                # Carrega o arquivo com weights_only=False para obter o checkpoint completo
                dados = torch.load(file_path, weights_only=False)
            except Exception as e:
                print(f"Erro ao carregar {file_path}: {e}")
                continue

            # Se os grafos estiverem dentro de um dicionário com a chave 'grafos'
            if isinstance(dados, dict) and "grafos" in dados:
                grafos_dict = dados["grafos"]
                for exam_id, graph_info in grafos_dict.items():
                    try:
                        # Seleciona o lead_0
                        graph = graph_info["lead_1"]
                    except KeyError:
                        print(f"Lead_0 não encontrado para o exam_id {exam_id} em {file_name}. Ignorando.")
                        continue

                    data = Data(
                        x=graph.x,                # features dos nós
                        edge_index=graph.edge_index,  # conectividade
                        y=torch.tensor(label, dtype=torch.long)
                    )
                    data.exam_id = exam_id  # opcional, para referência
                    dataset.append(data)
            # Se os dados já forem uma lista de objetos Data
            elif isinstance(dados, list):
                for data in dados:
                    data.y = torch.tensor(label, dtype=torch.long)
                    dataset.append(data)
            else:
                print(f"Formato dos dados em {file_name} não reconhecido.")

        # Verifica se algum dado foi carregado
        if len(dataset) == 0:
            raise ValueError("Nenhum dado foi carregado. Verifique os arquivos e os erros acima.")

        data, slices = self.collate(dataset)
        # Salva o dataset processado no arquivo de cache
        torch.save((data, slices), self.processed_paths[0])

# =============================================================================
# 3. Carregar os datasets para treino/validação e teste
# =============================================================================
# Diretório raiz do dataset
root_dir = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset"

# Dataset de treino e validação
dataset_train_val = ECGDataset(root=root_dir, class_files=class_files_train, split="train_val")
print(f"Total de amostras no dataset de treino/validação: {len(dataset_train_val)}")

# Dataset de teste
dataset_test = ECGDataset(root=root_dir, class_files=class_files_test, split="test")
print(f"Total de amostras no dataset de teste: {len(dataset_test)}")

# Dividir dataset_train_val em 95% treino e 5% validação
train_data, val_data = train_test_split(dataset_train_val, test_size=0.05, random_state=42)
print(f"Número de amostras de treino: {len(train_data)}")
print(f"Número de amostras de validação: {len(val_data)}")

# Todos os dados do dataset_test serão usados para teste
test_data = dataset_test

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

# Número de classes: 7 (conforme os mapeamentos definidos)
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
# 6. Treinar o modelo, calcular e exibir métricas por época, e plotar as curvas de loss e F1 Macro
# =============================================================================
num_epochs = 60
train_losses = []
val_losses = []
train_f1_macros = []
val_f1_macros = []

for epoch in tqdm(range(1, num_epochs+1), desc="Treinamento"):
    # Treinamento
    train_loss = train_epoch(train_loader)
    
    # Avaliação no conjunto de treino para métricas
    train_eval_loss, train_logits, train_targets = evaluate(train_loader)
    train_preds = train_logits.argmax(dim=1)
    train_f1_macro = f1_score(train_targets.numpy(), train_preds.numpy(), average='macro', zero_division=1)
    
    # Avaliação no conjunto de validação para métricas
    val_loss, val_logits, val_targets = evaluate(val_loader)
    val_preds = val_logits.argmax(dim=1)
    val_f1_macro = f1_score(val_targets.numpy(), val_preds.numpy(), average='macro', zero_division=1)
    
    # Armazenando as métricas para plotagem
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_f1_macros.append(train_f1_macro)
    val_f1_macros.append(val_f1_macro)
    
    # Exibindo as métricas por época
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train F1 Macro: {train_f1_macro:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val F1 Macro: {val_f1_macro:.4f}")

# Plot e salvamento da curva de loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_losses, label="Treino")
plt.plot(range(1, num_epochs+1), val_losses, label="Validação")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de Loss de Treino e Validação")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")  # Salva a imagem
plt.show()

# Plot e salvamento da curva do F1 Macro
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs+1), train_f1_macros, label="Train F1 Macro")
plt.plot(range(1, num_epochs+1), val_f1_macros, label="Val F1 Macro")
plt.xlabel("Epoch")
plt.ylabel("F1 Macro")
plt.title("Curva do F1 Macro por Epoch")
plt.legend()
plt.grid(True)
plt.savefig("f1_macro_curve.png")  # Salva a imagem
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
