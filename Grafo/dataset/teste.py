import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
from torch.serialization import safe_globals

# Defina o caminho do arquivo .pt
file_path = "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest2.pt"

# Use safe_globals para permitir explicitamente as classes necessárias
with safe_globals([Data, DataEdgeAttr]):
    # Carrega o arquivo na GPU; weights_only=False para carregar o objeto completo
    data = torch.load(file_path, map_location="cuda:0", weights_only=False)

print("Arquivo carregado na GPU com sucesso!")
print("Chaves principais do dicionário:", list(data.keys()))

if "grafos" in data:
    graficos = data["grafos"]
    print("Número de exames no arquivo:", len(graficos))
    
    # Inspeciona um exemplo de exame
    exam_id = list(graficos.keys())[29554]
    exame = graficos[exam_id]
    print(f"\nFormato do exame {exam_id}:")
    print("Chaves disponíveis:", list(exame.keys()))
    
    if "grafo" in exame:
        print("Tipo do objeto 'grafo':", type(exame["grafo"]))
    if "label" in exame:
        print("Tipo do objeto 'label':", type(exame["label"]))
else:
    print("A chave 'grafos' não foi encontrada no arquivo!")
