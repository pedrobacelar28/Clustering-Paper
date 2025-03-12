import torch
import os
import json
from tqdm import tqdm

def create_index_for_dataset(file_path):
    """
    Lê o arquivo .pt e constrói um mapeamento que associa o índice interno (0, 1, 2, ...)
    ao exam_id correspondente.
    
    Se o arquivo tiver as chaves "exam_ids" e "grafos" (grafos armazenados como lista),
    o mapeamento será {0: exam_id0, 1: exam_id1, ...}.
    Se o arquivo não possuir "exam_ids", assume-se que data["grafos"] é um dicionário,
    e o mapeamento será feito a partir da ordem dos exam_ids (ordenados).
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Arquivo não encontrado: {file_path}")
    
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    
    mapping = {}
    if "exam_ids" in data and "grafos" in data:
        exam_ids_list = data["exam_ids"]
        total_ids = len(exam_ids_list)
        for idx, exam_id in enumerate(tqdm(exam_ids_list, desc=f"Indexando {os.path.basename(file_path)}", total=total_ids, leave=False)):
            mapping[str(idx)] = exam_id  # chave é o índice, valor é o exam_id
    else:
        # Se for um dicionário, obtemos as chaves, ordenamos e determinamos o total
        exam_ids = list(data.get("grafos", {}).keys())
        exam_ids.sort()
        total_ids = len(exam_ids)
        for idx, exam_id in enumerate(tqdm(exam_ids, desc=f"Indexando {os.path.basename(file_path)}", total=total_ids, leave=False)):
            mapping[str(idx)] = exam_id
    return mapping

def save_index_file(file_path, mapping):
    """
    Salva o mapeamento gerado em um arquivo JSON.
    O arquivo de índice será salvo com o nome <file_path>.index.json.
    """
    index_file = file_path + ".index.json"
    with open(index_file, "w") as f:
        json.dump(mapping, f)
    print(f"Arquivo de índice salvo: {index_file}")

if __name__ == "__main__":
    # Lista dos arquivos .pt que compõem o dataset (exemplo para treino/validação)
    file_paths = [
        "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/lead1.pt",
        "/scratch/guilherme.evangelista/Clustering-Paper/Grafo/dataset/codetest2.pt"
    ]
    
    for fp in tqdm(file_paths, desc="Processando arquivos"):
        print(f"\nProcessando {fp} ...")
        mapping = create_index_for_dataset(fp)
        save_index_file(fp, mapping)
