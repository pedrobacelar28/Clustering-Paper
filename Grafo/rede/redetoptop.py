import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.data import Subset, ConcatDataset  # Adicionado ConcatDataset
import torch.nn as nn
import torch.nn.functional as F

# ================================================
# 1. Dataset: Carregamento completo do arquivo de exames
# ================================================
class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.data = []
        # Carrega o arquivo inteiro (já pré-processado)
        loaded = torch.load(file_path, map_location="cpu", weights_only=False)
        if "grafos" not in loaded:
            raise ValueError(f"Formato inesperado no arquivo {file_path}: 'grafos' não encontrado.")
        for exam_id, exam_info in loaded["grafos"].items():
            if "grafo" not in exam_info or "label" not in exam_info:
                print(f"Exame {exam_id} sem 'grafo' ou 'label' no arquivo {file_path}.")
                continue
            data_obj = exam_info["grafo"]
            # Adiciona o label como atributo do Data, com shape [1, 6]
            data_obj.y = torch.tensor(exam_info["label"], dtype=torch.float).unsqueeze(0)
            data_obj.exam_id = exam_id
            self.data.append(data_obj)
        print(f"Foram carregados {len(self.data)} exames a partir de {file_path}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ================================================
# 2. Carregando os datasets de treino/validação e teste (Arquivos)
# ================================================
# Arquivo original de treino/validação
train_val_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/dataset.pt"
# Arquivo adicional de treino/validação
train_val_file2 = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/dataset2.pt"
train_val_file3 = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/dataset3.pt"
# Arquivo de teste
test_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest.pt"

# Carrega os datasets individualmente
dataset1 = ECGDataset(train_val_file)
dataset2 = ECGDataset(train_val_file2)
dataset3 = ECGDataset(train_val_file3)
# Concatena os datasets de treino/validação
dataset_train_val = ConcatDataset([dataset1, dataset2, dataset3])
dataset_test = ECGDataset(test_file)

print(f"\nTotal de exames em TREINO/VAL: {len(dataset_train_val)}")
print(f"Total de exames em TESTE: {len(dataset_test)}")

# ================================================
# 3. Função para calcular a distribuição de classes
# ================================================
def compute_class_distribution(dataset):
    labels = []
    for data in tqdm(dataset, desc="Computando distribuição de classes"):
        labels.append(data.y)
    if len(labels) == 0:
        return None, None
    labels_tensor = torch.cat(labels, dim=0)  # shape: [N, num_classes]
    positive_counts = labels_tensor.sum(dim=0)  # soma dos positivos por classe
    total_examples = labels_tensor.size(0)
    return positive_counts, total_examples

counts, total = compute_class_distribution(dataset_train_val)
if counts is not None:
    print(f"\nDistribuição de classes no dataset TREINO/VAL (Total: {total} exemplos):")
    for i, count in enumerate(counts):
        print(f"  Classe {i}: {int(count.item())} (proporção: {count.item()/total:.2f})")

# ================================================
# 4. Separando em treino e validação usando Subset
# ================================================
if len(dataset_train_val) == 0:
    print("Nenhum exame encontrado no dataset de treino/val.")
    train_data = []
    val_data = []
else:
    indices = np.arange(len(dataset_train_val))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_data = Subset(dataset_train_val, train_idx)
    val_data = Subset(dataset_train_val, val_idx)
    print(f"\nExames em TREINO: {len(train_data)}")
    print(f"Exames em VAL: {len(val_data)}")
    print(f"Exames em TESTE: {len(dataset_test)}")

    for name, ds in {"TREINO": train_data, "VALIDAÇÃO": val_data, "TESTE": dataset_test}.items():
        counts, total = compute_class_distribution(ds)
        if counts is not None:
            print(f"\n{name} - Total de exemplos: {total}")
            for i, count in enumerate(counts):
                print(f"  Classe {i}: {int(count.item())} (proporção: {count.item()/total:.2f})")
        else:
            print(f"\n{name}: Nenhum exemplo encontrado.")

    # ================================================
    # 5. Preparando DataLoaders (carregamento rápido e eficiente)
    # ================================================
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # ================================================
    # 6. Definição do modelo GIN com 4 blocos
    # ================================================
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

    # ================================================
    # 7. Instanciar modelo, critério e otimizador
    # ================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = 48  # As features já vêm concatenadas (48 por nó)
    hidden_dim   = 128
    num_outputs  = 6   # 6 rótulos
    model = GINExclusive(num_features, hidden_dim, num_outputs, dropout=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ================================================
    # 8. Funções de treinamento e avaliação
    # ================================================
    def train_epoch(loader):
        model.train()
        epoch_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.num_graphs
        return epoch_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_loss = 0
        all_logits = []
        all_targets = []
        for data in loader:
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y)
            total_loss += loss.item() * data.num_graphs
            all_logits.append(logits.cpu())
            all_targets.append(data.y.cpu())
        
        avg_loss = total_loss / len(loader.dataset)
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return avg_loss, all_logits, all_targets

    def compute_f1_macro_multi_label(logits, targets, thresholds=0.5):
        probs = torch.sigmoid(logits)
        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds] * logits.shape[1]
        thresholds = torch.tensor(thresholds, dtype=torch.float32)
        preds = torch.zeros_like(probs)
        for class_idx in range(logits.shape[1]):
            preds[:, class_idx] = (probs[:, class_idx] >= thresholds[class_idx]).int()
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        f1_macro = f1_score(targets_np, preds_np, average='macro', zero_division=1)
        return f1_macro

    # ================================================
    # 9. Loop de treinamento e ajuste de threshold
    # ================================================
    num_epochs = 50
    train_losses = []
    val_losses   = []
    train_f1s    = []
    val_f1s      = []

    for epoch in tqdm(range(1, num_epochs+1), desc="Treinamento"):
        train_loss = train_epoch(train_loader)
        train_eval_loss, train_logits, train_targets = evaluate(train_loader)
        train_f1_05 = compute_f1_macro_multi_label(train_logits, train_targets)

        val_loss, val_logits, val_targets = evaluate(val_loader)
        val_f1_05 = compute_f1_macro_multi_label(val_logits, val_targets)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1_05)
        val_f1s.append(val_f1_05)
        
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f}, Train F1@0.5: {train_f1_05:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val F1@0.5: {val_f1_05:.4f}")

        plt.clf()  # Limpa a figura atual
        plt.plot(range(1, epoch+1), train_losses, label="Treino", color="blue")
        plt.plot(range(1, epoch+1), val_losses, label="Validação", color="orange")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.title("Curva de Loss (Treino vs Validação)")
        plt.legend()
        plt.savefig("loss_curve_multilabel.png")

        plt.clf()  # Limpa a figura atual
        plt.plot(range(1, epoch+1), train_f1s, label="Treino", color="blue")
        plt.plot(range(1, epoch+1), val_f1s, label="Validação", color="orange")
        plt.xlabel("Época")
        plt.ylabel("F1 Macro")
        plt.title("Curva de F1 Macro (Treino vs Validação)")
        plt.legend()
        plt.savefig("f1_curve_multilabel.png")
        
    # Ajuste de threshold na validação
    val_loss, val_logits, val_targets = evaluate(val_loader)
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_thresholds = np.zeros(num_outputs)
    best_f1s = np.zeros(num_outputs)

    for class_idx in range(num_outputs):
        best_f1 = 0.0
        best_thr = 0.5
        for thr in thresholds:
            f1_val = f1_score(val_targets[:, class_idx].numpy(), 
                               (torch.sigmoid(val_logits[:, class_idx]) >= thr).int().numpy(),
                               zero_division=1)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_thr = thr
        best_thresholds[class_idx] = best_thr
        best_f1s[class_idx] = best_f1

    print(f"\nMelhores thresholds por classe: {best_thresholds}")
    print(f"Melhores F1-scores por classe: {best_f1s}")

    # ================================================
    # 10. Plot das curvas de Loss e F1
    # ================================================
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label="Treino")
    plt.plot(range(1, num_epochs+1), val_losses,   label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Curva de Loss (Treino vs Validação)")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve_multilabel.png")
    plt.show()

    plt.figure()
    plt.plot(range(1, num_epochs+1), train_f1s, label="Treino")
    plt.plot(range(1, num_epochs+1), val_f1s,   label="Validação")
    plt.xlabel("Época")
    plt.ylabel("F1 Macro (thr=0.5)")
    plt.title("Curva de F1 Macro (Treino vs Validação)")
    plt.legend()
    plt.grid(True)
    plt.savefig("f1_curve_multilabel.png")
    plt.show()

    # ================================================
    # 11. Avaliação final no conjunto de teste
    # ================================================
    test_loss, test_logits, test_targets = evaluate(test_loader)
    test_f1 = compute_f1_macro_multi_label(test_logits, test_targets, best_thresholds)

    print(f"\nLoss no Teste: {test_loss:.4f}")
    print(f"F1 Macro no Teste: {test_f1:.4f}")

    # F1 por classe
    test_probs = torch.sigmoid(test_logits)
    test_preds = torch.zeros_like(test_probs)
    for class_idx in range(num_outputs):
        test_preds[:, class_idx] = (test_probs[:, class_idx] >= best_thresholds[class_idx]).int()
    test_preds_np = test_preds.numpy()
    test_targets_np = test_targets.numpy()

    print("\nF1 por classe (colunas 0..5):")
    for class_idx in range(num_outputs):
        f1_class = f1_score(test_targets_np[:, class_idx], test_preds_np[:, class_idx], zero_division=1)
        print(f"  Classe {class_idx}: {f1_class:.4f}")

    # Exemplos de previsões
    num_examples = 5
    print(f"\nExemplos aleatórios de {num_examples} previsões do conjunto de teste:")
    indices = np.random.choice(len(test_preds_np), size=min(num_examples, len(test_preds_np)), replace=False)
    for i, idx in enumerate(indices, 1):
        print(f"Exemplo {i}:")
        print("  rótulo real:", test_targets_np[idx])
        print("  predição   :", test_preds_np[idx])
        print("  (logits)   :", test_logits[idx].detach().numpy())
        print("  (sigmoid)  :", test_probs[idx].detach().numpy())
