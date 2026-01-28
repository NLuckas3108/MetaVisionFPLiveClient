import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QFileDialog, QColorDialog, QFrame)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint

# --- 3D Visualization Library ---
import pyvista as pv
from pyvistaqt import QtInteractor

# --- 1. Eigene Label-Klasse für Mausklicks ---
class ClickableVideoLabel(QLabel):
    on_click = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)

# --- 2. Der Kamera-Worker (Unverändert) ---
class RealSenseThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.pipeline = None

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            self.pipeline.start(config)
            while self._run_flag:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue

                cv_img = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.IgnoreAspectRatio)
                self.change_pixmap_signal.emit(p)

        except Exception as e:
            print(f"[ERROR] Kamera Fehler: {e}")
        finally:
            self.stop_pipeline()

    def stop(self):
        self._run_flag = False
        self.wait()

    def stop_pipeline(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass

# --- 3. Das CAD-Vorschau Widget (Angepasst für Farb-Updates) ---
class CADPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        
        # --- ÄNDERUNG: Hintergrund auf Schwarz setzen ---
        self.plotter.set_background("#000000") 
        
        # Optional: Achsen weglassen für einen cleaner Look
        # self.plotter.add_axes() 
        
        self.plotter.view_isometric()
        self.mesh_actor = None

    def load_mesh(self, file_path, initial_qcolor=None):
        try:
            self.plotter.clear()
            # Falls du die Achsen doch willst, hier wieder einkommentieren:
            # self.plotter.add_axes()
            
            mesh = pv.read(file_path)
            c = "lightgrey"
            if initial_qcolor and initial_qcolor.isValid():
                 c = initial_qcolor.name()

            self.mesh_actor = self.plotter.add_mesh(mesh, color=c, pbr=True, metallic=0.5)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as e:
            print(f"[ERROR] Konnte CAD nicht laden: {e}")

    def update_color(self, qcolor):
        if self.mesh_actor and qcolor.isValid():
            self.mesh_actor.prop.color = qcolor.name()
            self.plotter.render()


# --- 4. Die GUI ---
class ClientApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoundationPose Client")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.drawing_mode = False
        self.mask_points = []
        # Initialfarbe (Grün, volle Deckkraft für den Start)
        self.mask_color = QColor(0, 255, 0, 255) 
        self.cad_path = None

        # 1. LINKER BEREICH: VIDEO
        self.image_label = ClickableVideoLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("Kamera wird gestartet...")
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: #000;")
        self.image_label.setFixedSize(640, 480) 
        self.image_label.on_click.connect(self.handle_image_click)
        
        # 2. RECHTER BEREICH: SIDEBAR
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        btn_style = """
            QPushButton {
                background-color: #444; border-radius: 5px; 
                padding: 10px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
        """
        self.btn_style_active = """
            QPushButton {
                background-color: #d32f2f; border-radius: 5px; 
                padding: 10px; font-weight: bold; font-size: 14px; color: white;
            }
        """

        # Buttons
        self.btn_cad = QPushButton("📂 Upload CAD Model")
        self.btn_cad.setStyleSheet(btn_style)
        self.btn_cad.clicked.connect(self.upload_cad)
        
        self.btn_color = QPushButton("🎨 Pick Color")
        self.btn_color.setStyleSheet(btn_style)
        self.btn_color.clicked.connect(self.pick_color)
        # Initialfarbe auf den Button anwenden
        self.btn_color.setStyleSheet(f"background-color: {self.mask_color.name()}; color: black; padding: 10px; border-radius: 5px; font-weight: bold;")


        self.btn_mask = QPushButton("✏️ Draw Mask")
        self.btn_mask.setStyleSheet(btn_style)
        self.btn_mask.clicked.connect(self.start_drawing_mode)

        self.sidebar_layout.addWidget(self.btn_cad)
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(self.btn_color)
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(self.btn_mask)
        
        self.sidebar_layout.addSpacing(20)
        self.preview_label = QLabel("Preview:")
        self.preview_label.setStyleSheet("font-size: 12px; color: #aaa;")
        self.sidebar_layout.addWidget(self.preview_label)

        # Das 3D Widget
        self.cad_preview = CADPreviewWidget()
        self.cad_preview.setMinimumSize(250, 250)
        self.cad_preview.setStyleSheet("border: 1px solid #555;")
        self.sidebar_layout.addWidget(self.cad_preview)
        
        self.sidebar_layout.addStretch()

        self.main_layout.addWidget(self.image_label)
        self.main_layout.addSpacing(20)
        self.main_layout.addLayout(self.sidebar_layout)

        self.thread = RealSenseThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    # --- Logik Methoden ---
    def start_drawing_mode(self):
        self.drawing_mode = True
        self.mask_points = []
        self.btn_mask.setText("Click Point 1...")
        self.btn_mask.setStyleSheet(self.btn_style_active)

    def handle_image_click(self, x, y):
        if not self.drawing_mode: return

        self.mask_points.append((x, y))

        if len(self.mask_points) == 1:
            self.btn_mask.setText("Click Point 2...")
        elif len(self.mask_points) == 2:
            self.drawing_mode = False
            self.btn_mask.setText("✅ Mask Ready\n(Click for new)")
            self.btn_mask.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px; font-weight: bold;")

    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        painter = QPainter(pixmap)
        pen = QPen(self.mask_color, 3) # Randfarbe auch anpassen
        painter.setPen(pen)

        if len(self.mask_points) == 1:
            p1 = self.mask_points[0]
            painter.setBrush(self.mask_color)
            painter.drawEllipse(QPoint(p1[0], p1[1]), 4, 4)

        if len(self.mask_points) == 2:
            p1 = self.mask_points[0]
            p2 = self.mask_points[1]
            x, y = min(p1[0], p2[0]), min(p1[1], p2[1])
            w, h = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
            
            # Temporäre Farbe mit Transparenz für die Maske erstellen
            transparent_color = QColor(self.mask_color)
            transparent_color.setAlpha(100)
            
            painter.setBrush(QBrush(transparent_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(x, y, w, h)
            
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(Qt.GlobalColor.white, 2, Qt.PenStyle.DashLine))
            painter.drawRect(x, y, w, h)

        painter.end()
        self.image_label.setPixmap(pixmap)

    def upload_cad(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Wähle CAD Modell", "", "OBJ Files (*.obj);;All Files (*)")
        if file_name:
            self.cad_path = file_name
            self.btn_cad.setText("✅ CAD Geladen")
            self.btn_cad.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px; font-weight: bold;")
            
            # --- ÄNDERUNG: Aktuelle Farbe beim Laden übergeben ---
            self.cad_preview.load_mesh(file_name, initial_qcolor=self.mask_color)

    def pick_color(self):
            # Dialog öffnen mit der aktuellen Farbe als Startwert
            color = QColorDialog.getColor(initial=self.mask_color)
            if color.isValid():
                self.mask_color = color
                
                # --- ÄNDERUNG: Text mit Häkchen und Hintergrundfarbe ---
                self.btn_color.setText("✅ Color Selected")
                self.btn_color.setStyleSheet(f"""
                    background-color: {color.name()}; 
                    color: black; 
                    padding: 10px; 
                    border-radius: 5px; 
                    font-weight: bold;
                """)
                
                # Preview sofort updaten
                self.cad_preview.update_color(self.mask_color)

    def closeEvent(self, event):
        if hasattr(self, 'cad_preview'):
            self.cad_preview.plotter.close()
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClientApp()
    window.show()
    sys.exit(app.exec())