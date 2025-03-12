import torch
from torch.utils.data import Dataset
import os
import json

class LazyGraphPtDataset(Dataset):
    """
    Dataset que carrega sob demanda grafos salvos em arquivos .pt.
    
    Cada arquivo pode ter uma das duas estruturas:
      1. Um dicionário com a chave "grafos", onde cada exame (identificado por exam_id)
         contém um dicionário com:
           - "grafo": objeto do tipo torch_geometric.data.Data (o grafo com suas features)
           - "label": rótulo associado ao exame.
      2. Um dicionário com as chaves "exam_ids" e "grafos", onde:
           - "exam_ids": lista de exam_ids,
           - "grafos": lista de dicionários, cada um com "grafo" e "label".
    
    Essa classe constrói um índice com tuplas (file_path, exam_id) para todos os exames e,
    para cada arquivo, utiliza um mapeamento que relaciona exam_id ao índice interno.
    Se existir um arquivo de índice (formato JSON), o mapeamento é carregado dele,
    evitando carregar todo o dataset apenas para extrair os exam_ids.
    """
    def __init__(self, file_paths):
        """
        Parâmetros:
          - file_paths: lista com os caminhos dos arquivos .pt que compõem o dataset.
        """
        self.file_paths = file_paths
        self.index = []  # Lista de tuplas (file_path, exam_id)
        # Mapeamento: para cada arquivo, guarda um dicionário {exam_id: internal_index}
        self.file_index_mapping = {}
        
        # Constrói o índice e o mapeamento para cada arquivo.
        # Se existir um arquivo de índice (arquivo.pt.index.json), ele é carregado.
        # Caso contrário, carrega o arquivo completo para extrair os exam_ids.
        for fp in self.file_paths:
            if not os.path.exists(fp):
                raise ValueError(f"Arquivo não encontrado: {fp}")
            index_file = fp + ".index.json"
            if os.path.exists(index_file):
                # Carrega o mapeamento do arquivo de índice (formato: {"0": exam_id0, "1": exam_id1, ...})
                with open(index_file, "r") as f:
                    mapping_json = json.load(f)
                # Converte para um mapeamento {exam_id: internal_index}
                mapping = {int(exam_id): int(key) for key, exam_id in mapping_json.items()}
                self.file_index_mapping[fp] = mapping
                # Constrói o índice a partir dos valores do mapeamento
                for exam_id in mapping.keys():
                    self.index.append((fp, exam_id))
            else:
                # Fallback: constrói o mapeamento carregando o arquivo completo
                data = torch.load(fp, map_location="cpu", weights_only=False)
                if "exam_ids" in data and "grafos" in data:
                    exam_ids_list = data["exam_ids"]
                    mapping = {}
                    for idx, exam_id in enumerate(exam_ids_list):
                        mapping[exam_id] = idx
                        self.index.append((fp, exam_id))
                    self.file_index_mapping[fp] = mapping
                else:
                    # Caso o arquivo já seja um dicionário indexado por exam_id
                    graficos = data.get("grafos", {})
                    mapping = {int(exam_id): int(exam_id) for exam_id in graficos.keys()}
                    for exam_id in mapping.keys():
                        self.index.append((fp, exam_id))
                    self.file_index_mapping[fp] = mapping
        
        # Cache simples: guarda o último arquivo aberto e seus dados para evitar recarregamento
        self._cache_file = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)
    
    def _load_file(self, file_path):
        """
        Carrega e retorna os dados do arquivo dado, atualizando o cache.
        """
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        self._cache_file = file_path
        self._cache_data = data
        return data
    
    def __getitem__(self, idx):
        file_path, exam_id = self.index[idx]
        
        # Se o arquivo requisitado já está no cache, use-o; senão, carregue-o.
        if self._cache_file != file_path:
            data = self._load_file(file_path)
        else:
            data = self._cache_data
        
        # Verifica qual estrutura o arquivo possui e utiliza o mapeamento para acesso rápido.
        if "exam_ids" in data and "grafos" in data:
            # Arquivo com listas: usa o mapeamento para localizar o índice do exame.
            mapping = self.file_index_mapping[file_path]
            if exam_id not in mapping:
                raise ValueError(f"Exam ID {exam_id} não encontrado no arquivo {file_path}")
            internal_index = mapping[exam_id]
            graficos = data.get("grafos", [])
            if internal_index >= len(graficos):
                raise ValueError(f"Índice {internal_index} fora do alcance em {file_path}")
            sample = graficos[internal_index].copy()
        else:
            # Arquivo com dicionário: acesso direto pelo exam_id.
            graficos = data.get("grafos", {})
            if exam_id not in graficos:
                raise ValueError(f"Exam ID {exam_id} não encontrado no arquivo {file_path}")
            sample = graficos[exam_id].copy()
        
        # Adiciona o exam_id ao sample (opcional).
        sample["exam_id"] = exam_id
        return sample
