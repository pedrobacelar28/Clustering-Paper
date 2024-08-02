import csv
import tkinter as tk
from tkinter import messagebox
import os
import sys
import json
from PIL import Image

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_images_from_directory(directory):
    """ Load images from a directory and extract their IDs """
    exames = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_id = filename.split("(")[1].split(")")[0]  # Extract the ID from the filename
            img_path = resource_path(os.path.join(directory, filename))
            exames.append({"id": img_id, "img_path": img_path})
    return exames

class ExameClassifierApp(tk.Tk):
    def __init__(self, exames):
        super().__init__()
        self.title("Classificador de Exames")
        self.geometry("400x300")
        self.exames = exames
        self.resultados = []
        self.load_progress()

        self.entry_label = tk.Label(self, text="Escolha uma classe para o exame:")
        self.entry_label.pack(pady=10)

        self.entry = tk.Entry(self)
        self.entry.pack(pady=5)
        self.entry.bind("<Return>", self.process_entry)

        self.comment_label = tk.Label(self, text="Comentários (opcional):")
        self.comment_label.pack(pady=10)

        self.comment_entry = tk.Entry(self)
        self.comment_entry.pack(pady=5)

        self.submit_button = tk.Button(self, text="Submeter", command=self.process_entry)
        self.submit_button.pack(pady=10)

        self.save_button = tk.Button(self, text="Salvar", command=self.save_results)
        self.save_button.pack(pady=10)

        self.show_exame()

    def show_exame(self):
        """Abre a imagem do exame no visualizador padrão do sistema."""
        if self.current_exame_index < len(self.exames):
            exame = self.exames[self.current_exame_index]
            img_path = exame["img_path"]
            
            try:
                # Abre a imagem com o visualizador padrão do sistema
                Image.open(img_path).show()
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao abrir a imagem: {e}")
        else:
            messagebox.showinfo("Info", "Todos os exames foram classificados.")

    def process_entry(self, event=None):
        classificacao = self.entry.get().strip().upper()
        comentario = self.comment_entry.get().strip()
        classes = {
            "ST": "ST",
            "SB": "SB",
            "AF": "AF",
            "LBBB": "LBBB",
            "RBBB": "RBBB",
            "1DAVB": "1dAVb",
            "NORMAL": "normal"
        }

        if classificacao not in classes:
            messagebox.showerror("Erro", "Classe inválida. Tente novamente.")
            return

        exame = self.exames[self.current_exame_index]
        resultado = {
            "ID": exame["id"],
            "ST": classificacao == "ST",
            "SB": classificacao == "SB",
            "AF": classificacao == "AF",
            "LBBB": classificacao == "LBBB",
            "RBBB": classificacao == "RBBB",
            "1dAVb": classificacao == "1dAVb",
            "normal": classificacao == "normal",
            "comentarios": comentario
        }
        self.resultados.append(resultado)

        self.current_exame_index += 1
        self.entry.delete(0, tk.END)
        self.comment_entry.delete(0, tk.END)

        self.save_progress()

        if self.current_exame_index < len(self.exames):
            self.show_exame()
        else:
            self.save_results()
            messagebox.showinfo("Fim", "As classificações foram salvas no arquivo classificacoes_exames.csv")
            self.quit()

    def save_results(self):
        csv_file = "classificacoes_exames.csv"
        try:
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["ID", "ST", "SB", "AF", "LBBB", "RBBB", "1dAVb", "normal", "comentarios"])
                writer.writeheader()
                writer.writerows(self.resultados)
            messagebox.showinfo("Info", "As classificações foram salvas no arquivo classificacoes_exames.csv")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar o arquivo CSV: {e}")

    def save_progress(self):
        progress = {
            "current_exame_index": self.current_exame_index,
            "resultados": self.resultados
        }
        with open("progress.json", "w") as file:
            json.dump(progress, file)

    def load_progress(self):
        if os.path.exists("progress.json"):
            with open("progress.json", "r") as file:
                progress = json.load(file)
                self.current_exame_index = progress.get("current_exame_index", 0)
                self.resultados = progress.get("resultados", [])
        else:
            self.current_exame_index = 0
            self.resultados = []

if __name__ == "__main__":
    try:
        exames = load_images_from_directory(resource_path("imagens"))

        app = ExameClassifierApp(exames)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Erro", f"Erro durante a execução do aplicativo: {e}")
