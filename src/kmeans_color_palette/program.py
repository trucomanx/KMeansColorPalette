#!/usr/bin/python3

import sys
import json
import signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QPushButton, QFileDialog, QSpinBox, QCheckBox, QScrollArea, QGridLayout
)
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter
from PyQt5.QtCore import Qt

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

import colorsys

def rgb_to_hsl(r, g, b):
    """
    Converte RGB (0-255) para HSL.
    Retorna: H (0-360), S (0-1), L (0-1)
    """
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    h, l, s = colorsys.rgb_to_hls(r_, g_, b_)  # note: HLS no colorsys
    return h * 360, s, l


def hsl_to_rgb(h, s, l):
    """
    Converte HSL para RGB (0-255).
    Entrada: H (0-360), S (0-1), L (0-1)
    Retorna: r, g, b (0-255)
    """
    h_ = h / 360.0
    r_, g_, b_ = colorsys.hls_to_rgb(h_, l, s)  # note: HLS no colorsys
    return int(round(r_ * 255)), int(round(g_ * 255)), int(round(b_ * 255))


def rgb_to_lab(r, g, b):
    """
    Converte uma cor de RGB para CIELAB.
    Entrada: r, g, b (0-255)
    Saída: L, a, b (valores Lab)
    """
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)  # precisa de forma (1,1,3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[0, 0, 0], lab[0, 0, 1], lab[0, 0, 2]


def lab_to_rgb(L, a, b):
    """
    Converte uma cor de CIELAB para RGB.
    Entrada: L, a, b (valores Lab como retornados pela função anterior)
    Saída: r, g, b (0-255)
    """
    lab = np.array([[[L, a, b]]], dtype=np.uint8)  # precisa de forma (1,1,3)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb[0, 0, 0], rgb[0, 0, 1], rgb[0, 0, 2]


def convert_image(img, analysis_type="rgb"):
    """
    Converte uma imagem PIL ou numpy array para o espaço de cor desejado.
    Retorna uma matriz Nx3 para clustering.
    """
    img_np = np.array(img)

    if analysis_type == "rgb":
        return img_np.reshape(-1, 3)

    elif analysis_type == "lab":
        lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        return lab_img.reshape(-1, 3)

    elif analysis_type == "hsl":
        # converter pixel a pixel
        hsl_pixels = []
        for pixel in img_np.reshape(-1, 3):
            r, g, b = pixel
            h, s, l = rgb_to_hsl(r, g, b)
            hsl_pixels.append([h, s, l])
        return np.array(hsl_pixels)

    else:
        raise ValueError(f"Espaço de cor '{analysis_type}' não suportado.")

def create_color_data(centroid, w, d, score, analysis_type):
    """
    Converte o centróide para RGB se necessário e monta o dicionário de dados de cor.
    """
    if analysis_type == "rgb":
        rgb_centroid = tuple(map(int, centroid))
    elif analysis_type == "lab":
        rgb_centroid = tuple(map(int, lab_to_rgb(centroid[0], centroid[1], centroid[2])))
    elif analysis_type == "hsl":
        rgb_centroid = tuple(map(int, hsl_to_rgb(centroid[0], centroid[1], centroid[2])))
    else:
        rgb_centroid = tuple(map(int, centroid))  # fallback

    return {
        "centroid": rgb_centroid,
        "w": w,
        "d": d,
        "score": score,
        "checkbox": None
    }

class ColorPaletteGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gerador de Paleta de Cores")
        self.image_path = None
        self.colors_data = []  # Lista de dicts: {"centroid": (r,g,b), "w":..., "d":..., "score":...}
        
        self.init_ui()

    def init_ui(self):
        self.resize(800, 800)
    
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Seleção de arquivo e K ---
        file_layout = QHBoxLayout()
        
        # button
        self.btn_file = QPushButton("Selecionar Imagem")
        self.btn_file.clicked.connect(self.select_file)
        file_layout.addWidget(self.btn_file)
        
        # filepath
        self.file_label = QLabel("Nenhuma imagem selecionada")
        file_layout.addWidget(self.file_label)
        
        main_layout.addLayout(file_layout)
        
        
        # --- Label para mostrar imagem ---
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setFixedHeight(300)  # altura fixa para pré-visualização
        self.image_preview.setStyleSheet("border: 1px solid gray;")
        main_layout.addWidget(self.image_preview)
        
        
        # --- Seleção de arquivo e K ---
        kmeans_layout = QHBoxLayout()
        
        # Kmeans label
        self.lbl_k = QLabel("K clusters:")
        kmeans_layout.addWidget(self.lbl_k)
        
        # Kmeans spin
        self.spin_k = QSpinBox()
        self.spin_k.setMinimum(1)
        self.spin_k.setValue(5)
        kmeans_layout.addWidget(self.spin_k)
        
        # Novo combobox para escolher tipo de análise
        self.combo_analysis = QComboBox()
        self.combo_analysis.addItems(["RGB", "LAB", "HSL"])  # opções
        kmeans_layout.addWidget(self.combo_analysis)

        main_layout.addLayout(kmeans_layout)



        # --- Botão processar ---
        self.btn_process = QPushButton("Processar Imagem")
        self.btn_process.clicked.connect(self.process_image)
        main_layout.addWidget(self.btn_process)

        # --- Área de cores ---
        self.scroll_area = QScrollArea()
        self.color_widget = QWidget()
        self.color_layout = QHBoxLayout()
        self.color_widget.setLayout(self.color_layout)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.color_widget)
        main_layout.addWidget(self.scroll_area)

        # --- Botão gerar paleta ---
        btn_generate = QPushButton("Gerar Paleta")
        btn_generate.clicked.connect(self.generate_palette)
        main_layout.addWidget(btn_generate)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            self.file_label.setText(path.split("/")[-1])

            # Mostrar preview da imagem
            pixmap = QPixmap(self.image_path)
            pixmap = pixmap.scaled( self.image_preview.width(), 
                                    self.image_preview.height(), 
                                    Qt.KeepAspectRatio, 
                                    Qt.SmoothTransformation)
            self.image_preview.setPixmap(pixmap)

    def process_image(self):
        self.setEnabled(False)
        
        QApplication.processEvents()
        
        if not self.image_path:
            return

        K = self.spin_k.value()
        img = Image.open(self.image_path).convert("RGB")
        
        analysis_type = self.combo_analysis.currentText().lower()
        
        img_np = convert_image(img, analysis_type=analysis_type)

        
        # --- K-means ---
        kmeans = KMeans(n_clusters=K, random_state=42).fit(img_np)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # --- Calcular w, d, score ---
        w = []
        d = []
        score = []
        for i in range(K):
            mask = labels == i
            wi = mask.sum() / len(labels)
            di = np.linalg.norm(img_np[mask] - centroids[i], axis=1).mean() if mask.sum() > 0 else 0
            score_i = wi*255.0 / (1 + di)
            w.append(wi)
            d.append(di)
            score.append(score_i)
        
        # --- Salvar dados ---
        self.colors_data = []
        for i in range(K):
            self.colors_data.append(
                create_color_data(centroids[i], w[i], d[i], score[i], analysis_type)
            )
        
        # --- Atualizar GUI ---
        self.update_colors_gui()
        
        self.setEnabled(True)

    def update_colors_gui(self):
        self.colors_data.sort(key=lambda c: c['w'], reverse=True)
        
        # Limpar layout anterior
        while self.color_layout.count():
            item = self.color_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Adicionar cores
        for cdata in self.colors_data:
            color_box = QWidget()
            layout = QVBoxLayout()
            color_box.setLayout(layout)
            
            # Informações w, d, score
            color_hex = rgb_to_hex(cdata['centroid'])
            info_color = QLabel(f"{color_hex}")
            info_color.setAlignment(Qt.AlignCenter)
            info_color.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            layout.addWidget(info_color)
            
            # Quadrado de cor com checkbox
            chk = QCheckBox()
            chk.setStyleSheet(f"""
                QCheckBox {{
                    background-color: rgb{cdata['centroid']};
                    min-width: 50px;
                    min-height: 50px;
                }}
            QCheckBox::indicator {{
                background-color: white;
                border: 1px solid gray;
                width: 20px;
                height: 20px;
            }}
            QCheckBox::indicator:checked {{
                image: url(check.svg);
            }}
            QCheckBox::indicator:unchecked {{
                image: none;
            }}
            """)

            cdata["checkbox"] = chk
            layout.addWidget(chk)

            # Informações w, d, score
            info_label = QLabel(f"w: {100.0*cdata['w']:.2f}%\nd: {cdata['d']:.2f}\nscore: {cdata['score']:.4f}")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
            
            layout.addWidget(info_label)

            self.color_layout.addWidget(color_box)

    def generate_palette(self):
        if not self.colors_data:
            return

        selected_colors = [c['centroid'] for c in self.colors_data if c['checkbox'].isChecked()]
        if not selected_colors:
            return

        # Perguntar a pasta de destino
        save_dir = QFileDialog.getExistingDirectory(self, "Selecione a pasta para salvar a paleta")
        if not save_dir:
            return  # usuário cancelou

        # --- Salvar JSON ---
        json_path = f"{save_dir}/paleta.json"
        with open(json_path, "w") as f:
            json.dump([{"r": r, "g": g, "b": b} for r, g, b in selected_colors], f, indent=2)

        # --- Salvar PNG ---
        img = Image.open(self.image_path).convert("RGB")
        w, h = img.size
        bar_height = 50
        new_img = Image.new("RGB", (w, h + bar_height), color=(255, 255, 255))
        new_img.paste(img, (0, 0))

        # Desenhar barra de cores
        step = w / len(selected_colors)
        for i, c in enumerate(selected_colors):
            x0 = int(i * step)
            x1 = int((i + 1) * step)
            for xi in range(x0, x1):
                for yi in range(bar_height):
                    new_img.putpixel((xi, h + yi), c)

        png_path = f"{save_dir}/paleta.png"
        new_img.save(png_path)

        print(f"Paleta gerada:\n- {json_path}\n- {png_path}")


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    
    app = QApplication(sys.argv)
    window = ColorPaletteGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

