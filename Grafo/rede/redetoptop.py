import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
from sklearn.model_selection import GroupShuffleSplit  # Usado para split baseado em grupos
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from torch.utils.data import Subset, ConcatDataset  
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json

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

CLASS_NAMES = ["umdavb", "rbbb", "lbbb", "sb", "st", "af"]

# ================================================
# 2. Carregando os datasets de treino/validação e teste (Arquivos)
# ================================================
gorgona = 8
if gorgona == 8:
    # Arquivo original de treino/validação
    train_val_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/lead1.pt"
    # Arquivo adicional de treino/validação
    # Arquivo de teste
    test_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest2.pt"
elif gorgona == 1:
    train_val_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/train_val/dataset.pt"
    train_val_file2 = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/train_val/dataset2.pt"
    train_val_file3 = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/train_val/dataset3.pt"
    test_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest/codetest.pt"

print("Carregando os datasets:")
# Carrega os datasets individualmente
dataset1 = ECGDataset(train_val_file)
#dataset2 = ECGDataset(train_val_file2)
#dataset3 = ECGDataset(train_val_file3)
# Concatena os datasets de treino/validação
dataset_train_val = ConcatDataset([dataset1])
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
# 4. Separando em treino e validação usando GroupShuffleSplit
#     (garantindo que exames do mesmo paciente não se separem)
# ================================================
if len(dataset_train_val) == 0:
    print("Nenhum exame encontrado no dataset de treino/val.")
    train_data = []
    val_data = []
else:
    # Carrega os metadados do CSV com as colunas: exam_id, age, is_male, nn_predicted_age,
    # 1dAVb, RBBB, LBBB, SB, ST, AF, patient_id, death, timey, normal_ecg, trace_file
    metadata = pd.read_csv("/scratch/guilherme.evangelista/Clustering-Paper/Projeto/Database/exams.csv")  # ajuste o caminho conforme necessário

    # Cria um dicionário para mapear exam_id para patient_id
    exam_to_patient = dict(zip(metadata["exam_id"], metadata["patient_id"]))

    # Cria a lista de grupos (patient_id) para cada exame no dataset_train_val
    groups = []
    for data in dataset_train_val:
        patient_id = exam_to_patient.get(data.exam_id)
        if patient_id is None:
            raise ValueError(f"Exam_id {data.exam_id} não encontrado nos metadados.")
        groups.append(patient_id)

    # Realiza o split garantindo que exames do mesmo paciente fiquem juntos
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    indices = np.arange(len(dataset_train_val))
    train_idx, val_idx = next(gss.split(indices, groups=groups))
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
# 6. Definição do modelo GAT+GIN MultiLabel com 4 blocos
# ================================================
class GAT_GINMultiLabel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_outputs, dropout=0.3):
        super(GAT_GINMultiLabel, self).__init__()
        # Bloco 1: GAT seguido de GIN (a primeira GAT processa os features iniciais)
        self.gat_conv = GATConv(num_features, hidden_channels, heads=4, concat=False)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.ln1 = nn.LayerNorm(hidden_channels)  # LayerNorm no lugar do BatchNorm
        
        # Bloco 2: Nova camada GAT e GIN
        self.gat_conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)
        self.ln2 = nn.LayerNorm(hidden_channels)
        
        # Jumping Knowledge: concatenação das saídas dos 2 blocos
        self.lin_jump = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Camadas finais para classificação
        self.lin1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.ln_final = nn.LayerNorm(hidden_channels * 2)
        self.lin2 = nn.Linear(hidden_channels * 2, num_outputs)
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Bloco 1: GAT -> GIN com conexão residual estilo Transformer
        x_gat1 = self.gat_conv(x, edge_index)
        x_gat1 = F.relu(x_gat1)
        x1_sub = self.conv1(x_gat1, edge_index)
        # Soma residual e dropout antes da normalização
        x1 = x_gat1 + F.dropout(x1_sub, p=self.dropout, training=self.training)
        x1 = self.ln1(x1)
        
        # Bloco 2: GAT -> GIN com conexão residual estilo Transformer
        x_gat2 = self.gat_conv2(x1, edge_index)
        x_gat2 = F.relu(x_gat2)
        x2_sub = self.conv2(x_gat2, edge_index)
        x2 = x_gat2 + F.dropout(x2_sub, p=self.dropout, training=self.training)
        x2 = self.ln2(x2)
        
        # Jumping Knowledge: concatenação dos recursos dos 2 blocos
        x_cat = torch.cat([x1, x2], dim=1)
        x_cat = F.relu(self.lin_jump(x_cat))
        x_global = global_mean_pool(x_cat, batch)
        
        # Camadas finais para classificação
        x_global = self.lin1(x_global)
        x_global = self.ln_final(x_global)
        x_global = F.relu(x_global)
        x_global = F.dropout(x_global, p=self.dropout, training=self.training)
        out = self.lin2(x_global)
        return out



# ================================================
# 7. Instanciar modelo, critério e otimizador
# ================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_features = 7  # As features já vêm concatenadas (48 por nó)
hidden_dim   = 64
num_outputs  = 6   # 6 rótulos
model = GAT_GINMultiLabel(num_features, hidden_dim, num_outputs, dropout=0.2).to(device)

criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Criar diretório para salvar os resultados
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_dir = f"results/{timestamp}/"
os.makedirs(results_dir, exist_ok=True)
confusion_matrix_file = f"{results_dir}/confusion_matrix.csv"

print(f"Os resultados serão salvos em: {results_dir}")

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

def adjust_thresholds(val_logits, val_targets, num_outputs, step=0.01):
    thresholds = np.arange(0.0, 1.01, step)
    best_thresholds = np.zeros(num_outputs)
    best_f1s = np.zeros(num_outputs)

    for class_idx in range(num_outputs):
        best_f1 = 0.0
        best_thr = 0.5  
        for thr in thresholds:
            preds = (torch.sigmoid(val_logits[:, class_idx]) >= thr).int().numpy()
            f1_val = f1_score(val_targets[:, class_idx].numpy(), preds, zero_division=1)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_thr = thr
        best_thresholds[class_idx] = best_thr
        best_f1s[class_idx] = best_f1

    print(f"\nMelhores thresholds por classe: {best_thresholds}")
    print(f"Melhores F1-scores por classe: {best_f1s}")
    return best_thresholds

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

def compute_precision_recall(logits, targets, thresholds):
    probs = torch.sigmoid(logits)
    preds = torch.zeros_like(probs)
    for class_idx in range(logits.shape[1]):
        preds[:, class_idx] = (probs[:, class_idx] >= thresholds[class_idx]).int()
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    precision = precision_score(targets_np, preds_np, average='macro', zero_division=1)
    recall = recall_score(targets_np, preds_np, average='macro', zero_division=1)
    return precision, recall

def save_confusion_matrix(epoch, preds, targets, phase):
    # Reduz os rótulos multi-label para a classe mais ativada
    right_labels = targets.argmax(axis=1)
    pred_labels = preds.argmax(axis=1)
    conf_matrix = confusion_matrix(right_labels, pred_labels, labels=range(6))
    conf_list = []
    for true_class in range(6):
        row = [epoch, phase, CLASS_NAMES[true_class]] + conf_matrix[true_class].tolist()
        conf_list.append(row)
    column_names = ["Epoch", "Phase", "Right_Label"] + [f"Pred_{name}" for name in CLASS_NAMES]
    conf_df = pd.DataFrame(conf_list, columns=column_names)
    if not os.path.exists(confusion_matrix_file):
        conf_df.to_csv(confusion_matrix_file, index=False)
    else:
        conf_df.to_csv(confusion_matrix_file, mode='a', header=False, index=False)

def plot_metric(metric_name, train_values, val_values, test_values, epoch, results_dir, ylabel):
    plt.clf()
    plt.plot(range(1, epoch+1), train_values, label="Treino", color="blue")
    plt.plot(range(1, epoch+1), val_values, label="Validação", color="orange")
    plt.plot(range(1, epoch+1), test_values, label="Teste", color="red")
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.title(f"Curva de {metric_name} (Treino vs Validação)")
    plt.legend()
    plt.savefig(f"{results_dir}/{metric_name.lower().replace(' ', '_')}_curve_multilabel.png")

def apply_thresholds(logits, thresholds):
    probs = torch.sigmoid(logits)
    preds = torch.zeros_like(probs)
    for class_idx in range(logits.shape[1]):
        preds[:, class_idx] = (probs[:, class_idx] >= thresholds[class_idx]).int()
    return preds.numpy()

# ================================================
# 9. Loop de treinamento e ajuste de threshold
# ================================================
num_epochs = 30
train_losses = []
val_losses   = []
test_losses = []
train_f1s    = []
val_f1s      = []
test_f1s = []
train_precisions = []
val_precisions = []
test_precisions = []
train_recalls = []
val_recalls = []
test_recalls = []

for epoch in tqdm(range(1, num_epochs+1), desc="Treinamento"):
    # Treinamento
    train_loss = train_epoch(train_loader)
    train_eval_loss, train_logits, train_targets = evaluate(train_loader)
    train_f1_05 = compute_f1_macro_multi_label(train_logits, train_targets)
    
    # Validação
    val_loss, val_logits, val_targets = evaluate(val_loader)
    val_f1_05 = compute_f1_macro_multi_label(val_logits, val_targets)
    
    # Ajuste dos thresholds com base na validação
    best_thresholds = adjust_thresholds(val_logits, val_targets, num_outputs)
    
    # Teste usando os thresholds ajustados
    test_loss, test_logits, test_targets = evaluate(test_loader)
    test_f1 = compute_f1_macro_multi_label(test_logits, test_targets, best_thresholds)
    
    train_precision, train_recall = compute_precision_recall(train_logits, train_targets, best_thresholds)
    val_precision, val_recall = compute_precision_recall(val_logits, val_targets, best_thresholds)
    test_precision, test_recall = compute_precision_recall(test_logits, test_targets, best_thresholds)
    
    # Armazena métricas
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    train_f1s.append(train_f1_05)
    val_f1s.append(val_f1_05)
    test_f1s.append(test_f1)
    train_precisions.append(train_precision)
    val_precisions.append(val_precision)
    test_precisions.append(test_precision)
    train_recalls.append(train_recall)
    val_recalls.append(val_recall)
    test_recalls.append(test_recall)
    
    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f}, Train F1@0.5: {train_f1_05:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val F1@0.5: {val_f1_05:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
    
    # Plot das métricas
    plot_metric("Loss", train_losses, val_losses, test_losses, epoch, results_dir, "Loss")
    plot_metric("F1 Macro", train_f1s, val_f1s, test_f1s, epoch, results_dir, "F1 Macro")
    plot_metric("Precisão", train_precisions, val_precisions, test_precisions, epoch, results_dir, "Precisão")
    plot_metric("Revocação", train_recalls, val_recalls, test_recalls, epoch, results_dir, "Revocação (Recall)")
    
    val_preds = apply_thresholds(val_logits, best_thresholds)
    test_preds = apply_thresholds(test_logits, best_thresholds)
    
    # Salva matrizes de confusão
    save_confusion_matrix(epoch, val_preds, val_targets.numpy(), "val")
    save_confusion_matrix(epoch, test_preds, test_targets.numpy(), "test")
    
    print(f"\nMatriz de confusão salva em: {confusion_matrix_file}")

# Ajuste final de threshold na validação
val_loss, val_logits, val_targets = evaluate(val_loader)
best_thresholds = adjust_thresholds(val_logits, val_targets, num_outputs)

torch.save(model.state_dict(), f"{results_dir}/gat_gin_multilabel_model.pth")
print(f"Modelo salvo em: {results_dir}/gat_gin_multilabel_model.pth")

# ================================================
# 10. Plot das curvas de Loss e F1 (final)
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

# Exemplos de previsões do conjunto de teste
num_examples = 5
print(f"\nExemplos aleatórios de {num_examples} previsões do conjunto de teste:")
indices = np.random.choice(len(test_preds_np), size=min(num_examples, len(test_preds_np)), replace=False)
for i, idx in enumerate(indices, 1):
    print(f"Exemplo {i}:")
    print("  rótulo real:", test_targets_np[idx])
    print("  predição   :", test_preds_np[idx])
    print("  (logits)   :", test_logits[idx].detach().numpy())
    print("  (sigmoid)  :", test_probs[idx].detach().numpy())

# ================================================
# Salvar métricas em um arquivo JSON
# ================================================
metrics = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "test_losses": test_losses,
    "train_f1s": train_f1s,
    "val_f1s": val_f1s,
    "test_f1s": test_f1s,
    "best_thresholds": best_thresholds.tolist()
}

with open(f"{results_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Métricas salvas em: {results_dir}/metrics.json")
