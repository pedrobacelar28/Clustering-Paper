import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GINConv, global_mean_pool, GlobalAttention
from sklearn.model_selection import GroupShuffleSplit  # Usado para split baseado em grupos
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from torch.utils.data import Subset, ConcatDataset  
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import h5py

# ================================================
# 1. Dataset: Carregamento completo do arquivo de exames em HDF5
# ================================================
class ECGDatasetHDF5(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.data = []
        # Abre o arquivo HDF5 e itera sobre os exames no grupo "grafos"
        with h5py.File(file_path, "r") as f:
            if "grafos" not in f:
                raise ValueError(f"Formato inesperado no arquivo {file_path}: 'grafos' não encontrado.")
            grafo_grp = f["grafos"]
            for exam_id in grafo_grp.keys():
                exam_grp = grafo_grp[exam_id]
                # Carrega os datasets: x, edge_index e label
                # Estes datasets foram salvos mantendo os tipos originais (ex.: float32 para x, int64 para edge_index)
                x = exam_grp["x"][:]  # features do grafo
                edge_index = exam_grp["edge_index"][:]  # conectividade do grafo
                label = exam_grp["label"][:]  # label associada
                # Cria o objeto Data do PyTorch Geometric
                data_obj = Data(
                    x=torch.tensor(x, dtype=torch.float32),
                    edge_index=torch.tensor(edge_index, dtype=torch.int64)
                )
                # Adiciona a label como atributo do Data, com shape [1, num_classes]
                data_obj.y = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
                data_obj.exam_id = exam_id
                self.data.append(data_obj)
        print(f"Foram carregados {len(self.data)} exames a partir de {file_path}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

CLASS_NAMES = ["umdavb", "rbbb", "lbbb", "sb", "st", "af"]

# ================================================
# 2. Carregando os datasets de treino/validação e teste (HDF5)
# ================================================
gorgona = 8
if gorgona == 8:
    # Arquivos HDF5 para treino/validação e teste
    train_val_file = "/scratch2/guilherme.evangelista/Clustering-Paper/Grafo/dataset/datasethd.hdf5"
    #train_val_file2 = "/scratch2/guilherme.evangelista/Clustering-Paper/Grafo/dataset/normalized2.hdf5"
    test_file = "/scratch2/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetesthd.hdf5"
elif gorgona == 1:
    train_val_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/train_val/dataset.hdf5"
    train_val_file2 = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/train_val/dataset2.hdf5"
    test_file = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest/codetest.hdf5"

print("Carregando os datasets:")
dataset1 = ECGDatasetHDF5(train_val_file)
#dataset2 = ECGDatasetHDF5(train_val_file2)
dataset_train_val = ConcatDataset([dataset1])
dataset_test = ECGDatasetHDF5(test_file)

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
# ================================================
if len(dataset_train_val) == 0:
    print("Nenhum exame encontrado no dataset de treino/val.")
    train_data = []
    val_data = []
else:
    metadata = pd.read_csv("/scratch2/guilherme.evangelista/Clustering-Paper/Projeto/Database/exams.csv")
    exam_to_patient = dict(zip(metadata["exam_id"], metadata["patient_id"]))

    groups = []
    for data in dataset_train_val:
        patient_id = exam_to_patient.get(data.exam_id)
        if patient_id is None:
            raise ValueError(f"Exam_id {data.exam_id} não encontrado nos metadados.")
        groups.append(patient_id)

    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    indices = np.arange(len(dataset_train_val))
    train_idx, val_idx = next(gss.split(indices, groups=groups))
    from torch.utils.data import Subset
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
# 5. Preparando DataLoaders
# ================================================
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_loader  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

# ================================================
# 6. Definição do modelo GAT+GIN MultiLabel
# ================================================
class GAT_GINMultiLabel(nn.Module):
    def __init__(self, num_features, hidden_channels, num_outputs, dropout=0.3):
        super(GAT_GINMultiLabel, self).__init__()
        # Bloco 1: GAT seguido de GIN
        self.gat_conv = GATConv(num_features, hidden_channels, heads=4, concat=False)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(self.mlp1)
        self.ln1 = nn.LayerNorm(hidden_channels)
        
        # Bloco 2: Nova camada GAT e GIN
        self.gat_conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(self.mlp2)
        self.ln2 = nn.LayerNorm(hidden_channels)
        
        # Módulo de Global Attention Pooling
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        self.global_pool = GlobalAttention(gate_nn=self.gate_nn)
        
        # Camadas finais para classificação
        self.lin1 = nn.Linear(hidden_channels, hidden_channels * 2)
        self.ln_final = nn.LayerNorm(hidden_channels * 2)
        self.lin2 = nn.Linear(hidden_channels * 2, num_outputs)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Bloco 1
        x_gat1 = self.gat_conv(x, edge_index)
        x_gat1 = F.relu(x_gat1)
        x1_sub = self.conv1(x_gat1, edge_index)
        x1 = x_gat1 + F.dropout(x1_sub, p=self.dropout, training=self.training)
        x1 = self.ln1(x1)
        
        # Bloco 2
        x_gat2 = self.gat_conv2(x1, edge_index)
        x_gat2 = F.relu(x_gat2)
        x2_sub = self.conv2(x_gat2, edge_index)
        x2 = x_gat2 + F.dropout(x2_sub, p=self.dropout, training=self.training)
        x2 = self.ln2(x2)
        
        # Global Attention Pooling: agrega os nós de forma ponderada
        x_global = self.global_pool(x2, batch)
        
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
num_features = 28  # 28 features por nó
hidden_dim   = 64
num_outputs  = 6   # 6 rótulos
model = GAT_GINMultiLabel(num_features, hidden_dim, num_outputs, dropout=0.3).to(device)

criterion = nn.BCEWithLogitsLoss()
initial_lr = 0.001  # LR inicial
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)

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
    plt.title(f"Curva de {metric_name} (usando best thresholds)")
    plt.legend()
    plt.savefig(f"{results_dir}/{metric_name.lower().replace(' ', '_')}_curve_multilabel.png")

def apply_thresholds(logits, thresholds):
    probs = torch.sigmoid(logits)
    preds = torch.zeros_like(probs)
    for class_idx in range(logits.shape[1]):
        preds[:, class_idx] = (probs[:, class_idx] >= thresholds[class_idx]).int()
    return preds.numpy()

# ================================================
# 9. Loop de treinamento, ajuste de threshold e atualização iterativa das métricas
# ================================================
num_epochs = 50
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
metrics_history = []

current_lr = initial_lr

for epoch in tqdm(range(1, num_epochs+1), desc="Treinamento"):
    if epoch % 20 == 0:
        new_lr = current_lr / 2
        if new_lr < 0.00005:
            new_lr = 0.00005
        current_lr = new_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    train_loss = train_epoch(train_loader)
    
    train_eval_loss, train_logits, train_targets = evaluate(train_loader)
    val_loss, val_logits, val_targets = evaluate(val_loader)
    best_thresholds = adjust_thresholds(val_logits, val_targets, num_outputs)
    
    train_f1_best = compute_f1_macro_multi_label(train_logits, train_targets, best_thresholds)
    val_f1_best = compute_f1_macro_multi_label(val_logits, val_targets, best_thresholds)
    
    test_loss, test_logits, test_targets = evaluate(test_loader)
    test_f1 = compute_f1_macro_multi_label(test_logits, test_targets, best_thresholds)
    
    train_precision, train_recall = compute_precision_recall(train_logits, train_targets, best_thresholds)
    val_precision, val_recall = compute_precision_recall(val_logits, val_targets, best_thresholds)
    test_precision, test_recall = compute_precision_recall(test_logits, test_targets, best_thresholds)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    test_losses.append(test_loss)
    train_f1s.append(train_f1_best)
    val_f1s.append(val_f1_best)
    test_f1s.append(test_f1)
    train_precisions.append(train_precision)
    val_precisions.append(val_precision)
    test_precisions.append(test_precision)
    train_recalls.append(train_recall)
    val_recalls.append(val_recall)
    test_recalls.append(test_recall)
    
    print(f"Epoch {epoch:02d} | LR: {current_lr:.6f} | "
          f"Train Loss: {train_loss:.4f}, Train F1: {train_f1_best:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val F1: {val_f1_best:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
    plot_metric("Loss", train_losses, val_losses, test_losses, epoch, results_dir, "Loss")
    plot_metric("F1 Macro", train_f1s, val_f1s, test_f1s, epoch, results_dir, "F1 Macro")
    plot_metric("Precisão", train_precisions, val_precisions, test_precisions, epoch, results_dir, "Precisão")
    plot_metric("Revocação", train_recalls, val_recalls, test_recalls, epoch, results_dir, "Revocação (Recall)")
    
    val_preds = apply_thresholds(val_logits, best_thresholds)
    test_preds = apply_thresholds(test_logits, best_thresholds)
    
    save_confusion_matrix(epoch, val_preds, val_targets.numpy(), "val")
    save_confusion_matrix(epoch, test_preds, test_targets.numpy(), "test")
    
    epoch_metrics = {
        "epoch": epoch,
        "lr": current_lr,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_f1": train_f1_best,
        "val_f1": val_f1_best,
        "test_f1": test_f1,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "best_thresholds": best_thresholds.tolist()
    }
    metrics_history.append(epoch_metrics)
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=4)
    
    print(f"\nMatriz de confusão salva em: {confusion_matrix_file}")

val_loss, val_logits, val_targets = evaluate(val_loader)
best_thresholds = adjust_thresholds(val_logits, val_targets, num_outputs)
torch.save(model.state_dict(), f"{results_dir}/gat_gin_multilabel_model.pth")
print(f"Modelo salvo em: {results_dir}/gat_gin_multilabel_model.pth")

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
plt.ylabel("F1 Macro (best thresholds)")
plt.title("Curva de F1 Macro (Treino vs Validação)")
plt.legend()
plt.grid(True)
plt.savefig("f1_curve_multilabel.png")
plt.show()

test_loss, test_logits, test_targets = evaluate(test_loader)
test_f1 = compute_f1_macro_multi_label(test_logits, test_targets, best_thresholds)

print(f"\nLoss no Teste: {test_loss:.4f}")
print(f"F1 Macro no Teste: {test_f1:.4f}")

test_probs = torch.sigmoid(test_logits)
test_preds = torch.zeros_like(test_probs)
for class_idx in range(num_outputs):
    test_preds[:, class_idx] = (test_probs[:, class_idx] >= best_thresholds[class_idx]).int()
test_preds_np = test_preds.numpy()
test_targets_np = test_targets.numpy()

print("\nF1 por classe (conjunto de TESTE):")
for class_idx in range(num_outputs):
    f1_class = f1_score(test_targets_np[:, class_idx], test_preds_np[:, class_idx], zero_division=1)
    print(f"  Classe {class_idx} ({CLASS_NAMES[class_idx]}): {f1_class:.4f}")

num_examples = 5
print(f"\nExemplos aleatórios de {num_examples} previsões do conjunto de teste:")
indices = np.random.choice(len(test_preds_np), size=min(num_examples, len(test_preds_np)), replace=False)
for i, idx in enumerate(indices, 1):
    print(f"Exemplo {i}:")
    print("  Rótulo real:", test_targets_np[idx])
    print("  Predição   :", test_preds_np[idx])
    print("  (logits)   :", test_logits[idx].detach().numpy())
    print("  (sigmoid)  :", test_probs[idx].detach().numpy())

# ================================================
# Fim do script
# ================================================
