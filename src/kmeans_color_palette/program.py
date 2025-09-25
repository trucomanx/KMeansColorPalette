#!/usr/bin/python3

import os
import sys
import json
import signal
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSizePolicy, 
    QLabel, QPushButton, QFileDialog, QSpinBox, QCheckBox, QScrollArea, QGridLayout,
    QAction, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QDesktopServices, QIcon, QPixmap, QColor, QImage, QPainter
from PyQt5.QtCore import Qt, QUrl

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2

# import kmeans_color_palette.modules.configure as configure 
from kmeans_color_palette.modules.color import rgb_to_hex
from kmeans_color_palette.modules.color import rgb_to_hsl
from kmeans_color_palette.modules.color import hsl_to_rgb
from kmeans_color_palette.modules.color import rgb_to_lab
from kmeans_color_palette.modules.color import lab_to_rgb

from kmeans_color_palette.desktop import create_desktop_file, create_desktop_directory, create_desktop_menu
from kmeans_color_palette.modules.wabout  import show_about_window

import kmeans_color_palette.about as about
import kmeans_color_palette.modules.configure as configure 

CONFIG_PATH = os.path.join(os.path.expanduser("~"),".config",about.__package__,"config.json")


DEFAULT_CONTENT = { "toolbar_configure": "Configure",
                    "toolbar_configure_tooltip": "Open the configure Json file",
                    "toolbar_about": "About",
                    "toolbar_about_tooltip": "About the program",
                    "toolbar_coffee": "Coffee",
                    "toolbar_coffee_tooltip": "Buy me a coffee (TrucomanX)",
                    "window_width":800,
                    "window_height":700,
                    "preview_height": 300,
                    "select_image": "1. Select Image",
                    "no_selected_image": "No selected image",
                    "k_clusters": "K clusters:",
                    "process_image": "2. Process Image",
                    "generate_palette": "3. Generate palette",
                    "error": "Error",
                    "please_upload_image": "No image selected.\nPlease upload an image before initiating the process.",
                    "please_process_image": "No colors were processed.\nPlease upload and process an image before generating the palette.",
                    "please_select_colors": "No colors have been checked.\nPlease select some colors before generating the palette.",
                    "select_the_folder": "Select the folder to save the palette.",
                    "color_palette_generated": "Color palette generated"
                    }

configure.verify_default_config(CONFIG_PATH, default_content = DEFAULT_CONTENT)



CONFIG=configure.load_config(CONFIG_PATH)

#CONFIG = merge_defaults(CONFIG, DEFAULT_CONTENT)



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
        self.image_path = None
        self.colors_data = []  # Lista de dicts: {"centroid": (r,g,b), "w":..., "d":..., "score":...}
        
        self.init_ui()
        self.create_toolbar()
        self.init_progress_ui()

    def init_ui(self):
        self.setWindowTitle(about.__program_name__)
        self.resize(CONFIG["window_width"], CONFIG["window_height"])

        ## Icon
        # Get base directory for icons
        self.base_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.icon_path = os.path.join(self.base_dir_path, 'icons', 'logo.png')
        self.setWindowIcon(QIcon(self.icon_path)) 

    
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Seleção de arquivo e K ---
        file_layout = QHBoxLayout()
        
        # button
        self.btn_file = QPushButton(CONFIG["select_image"])
        self.btn_file.clicked.connect(self.select_file)
        file_layout.addWidget(self.btn_file)
        
        # filepath
        self.file_label = QLabel(CONFIG["no_selected_image"])
        file_layout.addWidget(self.file_label)
        
        main_layout.addLayout(file_layout)
        
        
        # --- Label para mostrar imagem ---
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setFixedHeight(CONFIG["preview_height"])  # altura fixa para pré-visualização
        self.image_preview.setStyleSheet("border: 1px solid gray;")
        main_layout.addWidget(self.image_preview)
        
        
        # --- Seleção de arquivo e K ---
        kmeans_layout = QHBoxLayout()
        
        # Kmeans label
        self.lbl_k = QLabel(CONFIG["k_clusters"])
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
        self.btn_process = QPushButton(CONFIG["process_image"])
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
        btn_generate = QPushButton(CONFIG["generate_palette"])
        btn_generate.clicked.connect(self.generate_palette)
        main_layout.addWidget(btn_generate)

    def init_progress_ui(self):
        # Criar barra de progresso
        self.progress = QProgressBar()
        self.progress.setMinimum(0)   # valor inicial
        self.progress.setValue(0)  # exemplo: 40%
        self.progress.setMaximum(100) 
        
        # Adicionar na status bar
        self.statusBar().addPermanentWidget(self.progress)
        
    def create_toolbar(self):
        # Toolbar exemplo (você pode adicionar actions depois)
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # Adicionar o espaçador
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        #
        self.configure_action = QAction(QIcon.fromTheme("document-properties"), CONFIG["toolbar_configure"], self)
        self.configure_action.setToolTip(CONFIG["toolbar_configure_tooltip"])
        self.configure_action.triggered.connect(self.open_configure_editor)
        
        #
        self.about_action = QAction(QIcon.fromTheme("help-about"), CONFIG["toolbar_about"], self)
        self.about_action.setToolTip(CONFIG["toolbar_about_tooltip"])
        self.about_action.triggered.connect(self.open_about)
        
        # Coffee
        self.coffee_action = QAction(QIcon.fromTheme("emblem-favorite"), CONFIG["toolbar_coffee"], self)
        self.coffee_action.setToolTip(CONFIG["toolbar_coffee_tooltip"])
        self.coffee_action.triggered.connect(self.on_coffee_action_click)
    
        self.toolbar.addWidget(spacer)
        self.toolbar.addAction(self.configure_action)
        self.toolbar.addAction(self.about_action)
        self.toolbar.addAction(self.coffee_action)

    def open_configure_editor(self):
        if os.name == 'nt':  # Windows
            os.startfile(CONFIG_PATH)
        elif os.name == 'posix':  # Linux/macOS
            subprocess.run(['xdg-open', CONFIG_PATH])

    def on_coffee_action_click(self):
        QDesktopServices.openUrl(QUrl("https://ko-fi.com/trucomanx"))
    
    def open_about(self):
        data={
            "version": about.__version__,
            "package": about.__package__,
            "program_name": about.__program_name__,
            "author": about.__author__,
            "email": about.__email__,
            "description": about.__description__,
            "url_source": about.__url_source__,
            "url_doc": about.__url_doc__,
            "url_funding": about.__url_funding__,
            "url_bugs": about.__url_bugs__
        }
        show_about_window(data,self.icon_path)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, CONFIG["select_image"], "", "Images (*.png *.jpg *.jpeg)")
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

    def convert_image(self, img, analysis_type="rgb"):
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
            
            total_pixels = img_np.shape[0] * img_np.shape[1]   # largura * altura
            self.progress.setMaximum(total_pixels)
            
            for i,pixel in enumerate(img_np.reshape(-1, 3)):
                r, g, b = pixel
                h, s, l = rgb_to_hsl(r, g, b)
                hsl_pixels.append([h, s, l])
                self.progress.setValue(i+1)
            return np.array(hsl_pixels)

        else:
            raise ValueError(f"Espaço de cor '{analysis_type}' não suportado.")

    def process_image(self):
        self.setEnabled(False)
        self.progress.setValue(0)
        
        QApplication.processEvents()
        
        if not self.image_path:
            self.setEnabled(True)
            QMessageBox.warning(
                self,
                CONFIG["error"],
                CONFIG["please_upload_image"]
            )
            return

        K = self.spin_k.value()
        img = Image.open(self.image_path).convert("RGB")
        
        analysis_type = self.combo_analysis.currentText().lower()
        
        img_np = self.convert_image(img, analysis_type=analysis_type)

        
        # --- K-means ---
        self.progress.setMaximum(K)
        kmeans = KMeans(n_clusters=K, random_state=42).fit(img_np)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        self.progress.setValue(K)

        # --- Calcular w, d, score ---
        w = []
        d = []
        score = []
        self.progress.setMaximum(K)
        for i in range(K):
            mask = labels == i
            wi = mask.sum() / len(labels)
            di = np.linalg.norm(img_np[mask] - centroids[i], axis=1).mean() if mask.sum() > 0 else 0
            score_i = wi*255.0 / (1 + di)
            w.append(wi)
            d.append(di)
            score.append(score_i)
            self.progress.setValue(i+1)
        
        # --- Salvar dados ---
        self.colors_data = []
        for i in range(K):
            self.colors_data.append(
                create_color_data(centroids[i], w[i], d[i], score[i], analysis_type)
            )
            self.progress.setValue(i+1)
        
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
            img_path = os.path.join(self.base_dir_path, 'icons', 'check.svg')
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
                image: url({img_path});
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
            QMessageBox.warning(
                self,
                CONFIG["error"],
                CONFIG["please_process_image"]
            )
            return

        selected_colors = [c['centroid'] for c in self.colors_data if c['checkbox'].isChecked()]
        if not selected_colors:
            QMessageBox.warning(
                self,
                CONFIG["error"],
                CONFIG["please_select_colors"]
            )
            return

        # Perguntar a pasta de destino
        save_dir = QFileDialog.getExistingDirectory(self, CONFIG["select_the_folder"])
        
        if not save_dir:
            return  # usuário cancelou

        # --- Salvar JSON ---
        json_path = f"{save_dir}/color_palette.json"
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

        png_path = f"{save_dir}/color_palette.png"
        new_img.save(png_path)

        QMessageBox.information(
            self,
            CONFIG["color_palette_generated"],
            f"{json_path}\n{png_path}"
        )
        

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    
    app = QApplication(sys.argv)
    window = ColorPaletteGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

