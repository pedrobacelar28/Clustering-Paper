import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

# ================================================
# 1. Função para carregar um arquivo .pt
# ================================================
def load_graphs_from_pt(path_pt):
    conteudo = torch.load(path_pt, map_location="cpu", weights_only=False)
    if "grafos" not in conteudo:
        print(f"Formato do arquivo inesperado em {path_pt}: não encontrei 'grafos'.")
        return []

    exames_dict = conteudo["grafos"]
    dataset = []
    
    for exam_id, exam_info in exames_dict.items():
        if "grafos" not in exam_info or "label" not in exam_info:
            print(f"Exame {exam_id} sem 'grafos' ou 'label'; ignorando.")
            continue
        
        # Ajuste: cria shape [1, 6] em vez de [6]
        label_tensor = torch.tensor(exam_info["label"], dtype=torch.float).unsqueeze(0)
        
        leads_dict = exam_info["grafos"]
        if "lead_0" not in leads_dict:
            print(f"Exame {exam_id} sem 'lead_0'; ignorando.")
            continue
        
        lead0_data = leads_dict["lead_0"]
        edge_index = lead0_data.edge_index
        num_nodes = lead0_data.x.shape[0]
        
        all_leads_features = []
        for i in range(12):
            lead_key = f"lead_{i}"
            if lead_key not in leads_dict:
                pad = torch.zeros((num_nodes, 4), dtype=torch.float)
                all_leads_features.append(pad)
                continue

            x_this_lead = leads_dict[lead_key].x
            if x_this_lead.shape[0] != num_nodes:
                tmp = torch.zeros((num_nodes, x_this_lead.shape[1]), dtype=torch.float)
                min_nodes = min(num_nodes, x_this_lead.shape[0])
                tmp[:min_nodes, :] = x_this_lead[:min_nodes, :]
                x_this_lead = tmp
            
            all_leads_features.append(x_this_lead)
        
        x_concat = torch.cat(all_leads_features, dim=1)  # shape: [num_nodes, 48]
        
        data_obj = Data(
            x=x_concat,
            edge_index=edge_index,
            y=label_tensor  # tensor de rótulo com shape [1, 6]
        )
        data_obj.exam_id = exam_id
        dataset.append(data_obj)
    
    return dataset

# ================================================
# 2. Função para carregar vários arquivos .pt de um diretório
# ================================================
def load_graphs_from_folder(folder_path):
    dataset = []
    pt_files = glob.glob(os.path.join(folder_path, "*.pt"))
    print(f"Foram encontrados {len(pt_files)} arquivos .pt em {folder_path}")
    for pt_file in pt_files:
        dataset += load_graphs_from_pt(pt_file)
    return dataset

# ================================================
# 3. Carregando os datasets de treino/validação e teste
# ================================================
train_val_folder = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/test/train_val"  # ajuste para o diretório correto
test_folder      = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/test/teste"       # ajuste para o diretório correto

dataset_train_val = load_graphs_from_folder(train_val_folder)
dataset_test = load_graphs_from_folder(test_folder)

print(f"\nTotal de amostras em TREINO/VAL: {len(dataset_train_val)}")
print(f"Total de amostras em TESTE: {len(dataset_test)}")

# ================================================
# 4. Função para calcular a distribuição de classes
# ================================================
def compute_class_distribution(dataset):
    labels = [data.y for data in dataset]
    if len(labels) == 0:
        return None, None
    labels_tensor = torch.cat(labels, dim=0)  # shape: [N, num_classes]
    positive_counts = labels_tensor.sum(dim=0)  # soma dos positivos por classe
    total_examples = labels_tensor.size(0)
    return positive_counts, total_examples

# Exibe distribuição de classes para o dataset de treino/validação (antes do split)
counts, total = compute_class_distribution(dataset_train_val)
if counts is not None:
    print(f"\nDistribuição de classes no dataset TREINO/VAL (Total: {total} exemplos):")
    for i, count in enumerate(counts):
        print(f"  Classe {i}: {int(count.item())} (proporção: {count.item()/total:.2f})")

# ================================================
# 5. Separando em treino e validação
# ================================================
if len(dataset_train_val) == 0:
    print("Nenhuma amostra encontrada no dataset de treino/val. Verifique os arquivos e o parsing.")
    train_data = []
    val_data = []
else:
    train_data, val_data = train_test_split(dataset_train_val, test_size=0.2, random_state=42)
    print(f"\nExames em TREINO: {len(train_data)}")
    print(f"Exames em VAL: {len(val_data)}")
    print(f"Exames em TESTE: {len(dataset_test)}")

    # Exibe a distribuição de classes para cada subconjunto
    datasets = {"TREINO": train_data, "VALIDAÇÃO": val_data, "TESTE": dataset_test}
    for name, dataset in datasets.items():
        counts, total = compute_class_distribution(dataset)
        if counts is not None:
            print(f"\n{name} - Total de exemplos: {total}")
            for i, count in enumerate(counts):
                print(f"  Classe {i}: {int(count.item())} (proporção: {count.item()/total:.2f})")
        else:
            print(f"\n{name}: Nenhum exemplo encontrado.")

    # ================================================
    # 6. Preparando DataLoaders
    # ================================================
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # ================================================
    # 7. Definição do modelo GIN com 4 blocos
    # ================================================
    class GINMultiLabel(nn.Module):
        def __init__(self, num_features, hidden_channels, num_outputs, dropout=0.3):
            super(GINMultiLabel, self).__init__()
            
            self.res_conv1 = nn.Linear(num_features, hidden_channels) if num_features != hidden_channels else nn.Identity()
            # Bloco 1
            self.mlp1 = nn.Sequential(
                nn.Linear(num_features, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.conv1 = GINConv(self.mlp1)
            self.bn1 = nn.BatchNorm1d(hidden_channels)

            # Bloco 2
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.conv2 = GINConv(self.mlp2)
            self.bn2 = nn.BatchNorm1d(hidden_channels)

            # Bloco 3
            self.mlp3 = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.conv3 = GINConv(self.mlp3)
            self.bn3 = nn.BatchNorm1d(hidden_channels)

            # Bloco 4
            self.mlp4 = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.conv4 = GINConv(self.mlp4)
            self.bn4 = nn.BatchNorm1d(hidden_channels)

            # Jumping Knowledge
            self.lin_jump = nn.Linear(hidden_channels * 4, hidden_channels)

            # Camadas finais
            self.lin1 = nn.Linear(hidden_channels, hidden_channels * 2)
            self.bn_final = nn.BatchNorm1d(hidden_channels * 2)
            self.lin2 = nn.Linear(hidden_channels * 2, num_outputs)

            self.dropout = dropout

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x1 = self.conv1(x, edge_index)
            x1 = F.relu(x1 + self.res_conv1(x))
            x1 = self.bn1(x1)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)

            x2 = self.conv2(x1, edge_index)
            x2 = F.relu(x2 + x1)
            x2 = self.bn2(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)

            x3 = self.conv3(x2, edge_index)
            x3 = F.relu(x3 + x2)
            x3 = self.bn3(x3)
            x3 = F.dropout(x3, p=self.dropout, training=self.training)

            x4 = self.conv4(x3, edge_index)
            x4 = F.relu(x4 + x3)
            x4 = self.bn4(x4)

            x_cat = torch.cat([x1, x2, x3, x4], dim=1)
            x_cat = F.relu(self.lin_jump(x_cat))
            x_global = global_mean_pool(x_cat, batch)

            x_global = self.lin1(x_global)
            x_global = self.bn_final(x_global)
            x_global = F.relu(x_global)
            x_global = F.dropout(x_global, p=self.dropout, training=self.training)

            out = self.lin2(x_global)
            return out

    # ================================================
    # 8. Instanciar modelo, critério e otimizador
    # ================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_features = 48  # 4 features * 12 leads
    hidden_dim   = 128
    num_outputs  = 6   # 6 rótulos
    model = GINMultiLabel(num_features, hidden_dim, num_outputs, dropout=0.3).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ================================================
    # 9. Funções de treinamento e avaliação
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
        all_logits = torch.cat(all_logits, dim=0)  # shape: [N, 6]
        all_targets = torch.cat(all_targets, dim=0)  # shape: [N, 6]
        return avg_loss, all_logits, all_targets

    def compute_f1_macro_multi_label(logits, targets, threshold=0.5):
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int()
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        f1_macro = f1_score(targets_np, preds_np, average='macro', zero_division=1)
        return f1_macro

    # ================================================
    # 10. Loop de treinamento e afinamento do threshold
    # ================================================
    num_epochs = 30
    train_losses = []
    val_losses   = []
    train_f1s    = []
    val_f1s      = []

    for epoch in tqdm(range(1, num_epochs+1), desc="Treinamento"):
        train_loss = train_epoch(train_loader)

        # Métricas (threshold=0.5)
        train_eval_loss, train_logits, train_targets = evaluate(train_loader)
        train_f1_05 = compute_f1_macro_multi_label(train_logits, train_targets, threshold=0.5)
        
        val_loss, val_logits, val_targets = evaluate(val_loader)
        val_f1_05 = compute_f1_macro_multi_label(val_logits, val_targets, threshold=0.5)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1_05)
        val_f1s.append(val_f1_05)
        
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f}, Train F1@0.5: {train_f1_05:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val F1@0.5: {val_f1_05:.4f}")

    # Ajuste de threshold na validação
    val_loss, val_logits, val_targets = evaluate(val_loader)
    thresholds = np.arange(0.0, 1.01, 0.05)
    best_thr = 0.5
    best_f1 = 0.0
    for thr in thresholds:
        f1_val = compute_f1_macro_multi_label(val_logits, val_targets, threshold=thr)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thr = thr

    print(f"\nMelhor threshold na validação: {best_thr:.2f} (F1 Macro = {best_f1:.4f})")

    # ================================================
    # 11. Plot das curvas de Loss e F1
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
    # 12. Avaliação final no conjunto de teste
    # ================================================
    test_loss, test_logits, test_targets = evaluate(test_loader)
    test_f1 = compute_f1_macro_multi_label(test_logits, test_targets, threshold=best_thr)

    print(f"\nLoss no Teste: {test_loss:.4f}")
    print(f"F1 Macro no Teste @threshold={best_thr:.2f}: {test_f1:.4f}")

    # F1 por classe
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= best_thr).int()
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
        print("  Rótulo real:", test_targets_np[idx])
        print("  Predição   :", test_preds_np[idx])
        print("  (logits)   :", test_logits[idx].detach().numpy())
        print("  (sigmoid)  :", test_probs[idx].detach().numpy())
