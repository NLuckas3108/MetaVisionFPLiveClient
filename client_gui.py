import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QFileDialog, QColorDialog)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint

# --- 1. Eigene Label-Klasse für Mausklicks ---
class ClickableVideoLabel(QLabel):
    # Signal definiert: sendet x und y Koordinate beim Klick
    on_click = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        # Wir wollen nur linke Mausklicks
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click.emit(event.pos().x(), event.pos().y())
        # Event weiterreichen (Good Practice)
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
        # 640x480 ist wichtig für die Klick-Koordinaten
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
                
                # Wir skalieren hier direkt auf 640x480, damit Bild- und Label-Koordinaten 1:1 passen
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

# --- 3. Die GUI ---
class ClientApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoundationPose Client")
        self.setGeometry(100, 100, 900, 550)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Variablen für Maske ---
        self.drawing_mode = False  # Sind wir im Zeichen-Modus?
        self.mask_points = []      # Liste für die 2 Punkte [(x1,y1), (x2,y2)]
        self.mask_color = QColor(0, 255, 0, 100) # Standard: Grün, semi-transparent (Alpha=100)

        # 1. LINKER BEREICH: VIDEO (Jetzt mit ClickableVideoLabel)
        self.image_label = ClickableVideoLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("Kamera wird gestartet...")
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: #000;")
        self.image_label.setFixedSize(640, 480) 
        # Klick-Signal verbinden
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
        btn_style_active = """
            QPushButton {
                background-color: #d32f2f; border-radius: 5px; 
                padding: 10px; font-weight: bold; font-size: 14px; color: white;
            }
        """
        self.btn_style_normal = btn_style
        self.btn_style_active = btn_style_active

        self.btn_cad = QPushButton("📂 Upload CAD Model")
        self.btn_cad.setStyleSheet(btn_style)
        self.btn_cad.clicked.connect(self.upload_cad)
        
        self.btn_color = QPushButton("🎨 Pick Color")
        self.btn_color.setStyleSheet(btn_style)
        self.btn_color.clicked.connect(self.pick_color)

        # Masken Button
        self.btn_mask = QPushButton("✏️ Draw Mask")
        self.btn_mask.setStyleSheet(btn_style)
        self.btn_mask.clicked.connect(self.start_drawing_mode)

        self.sidebar_layout.addWidget(self.btn_cad)
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(self.btn_color)
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(self.btn_mask)
        self.sidebar_layout.addStretch() 

        self.main_layout.addWidget(self.image_label)
        self.main_layout.addLayout(self.sidebar_layout)

        # Thread starten
        self.thread = RealSenseThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.cad_path = None

    def start_drawing_mode(self):
        """Startet den Masken-Zeichnen Modus"""
        self.drawing_mode = True
        self.mask_points = [] # Alte Punkte löschen
        self.btn_mask.setText("Click Point 1...")
        self.btn_mask.setStyleSheet(self.btn_style_active) # Rot färben als Indikator

    def handle_image_click(self, x, y):
        """Wird aufgerufen, wenn ins Bild geklickt wird"""
        if not self.drawing_mode:
            return

        self.mask_points.append((x, y))

        if len(self.mask_points) == 1:
            print(f"Punkt 1 gesetzt: {x}, {y}")
            self.btn_mask.setText("Click Point 2...")
        
        elif len(self.mask_points) == 2:
            print(f"Punkt 2 gesetzt: {x}, {y}")
            self.drawing_mode = False # Fertig mit Zeichnen
            self.btn_mask.setText("✅ Mask Ready")
            self.btn_mask.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px; font-weight: bold;")
            # Falls man neu zeichnen will, klickt man wieder den Button

    def update_image(self, qt_img):
        """Wird jeden Frame aufgerufen. Hier malen wir das Overlay drauf."""
        
        # 1. Bild in Pixmap umwandeln, damit wir malen können
        pixmap = QPixmap.fromImage(qt_img)
        
        # 2. Painter starten
        painter = QPainter(pixmap)
        
        # Einstellungen für den Stift (Rand) und Pinsel (Füllung)
        pen = QPen(Qt.GlobalColor.green, 3)
        painter.setPen(pen)

        # --- Logik zum Zeichnen der Maske ---
        if len(self.mask_points) >= 1:
            # Ersten Punkt als kleinen Kreis malen
            p1 = self.mask_points[0]
            painter.setBrush(Qt.GlobalColor.green)
            painter.drawEllipse(QPoint(p1[0], p1[1]), 4, 4)

        if len(self.mask_points) == 2:
            # Wenn beide Punkte da sind, Rechteck malen
            p1 = self.mask_points[0]
            p2 = self.mask_points[1]
            
            # Koordinaten berechnen (oben links, breite, höhe)
            x = min(p1[0], p2[0])
            y = min(p1[1], p2[1])
            w = abs(p1[0] - p2[0])
            h = abs(p1[1] - p2[1])
            
            # Rechteck füllen (Transparent)
            painter.setBrush(QBrush(self.mask_color))
            painter.setPen(Qt.PenStyle.NoPen) # Kein Rand für das Füll-Rechteck
            painter.drawRect(x, y, w, h)
            
            # Optional: Rand drumherum
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(Qt.GlobalColor.white, 2, Qt.PenStyle.DashLine))
            painter.drawRect(x, y, w, h)

        painter.end()

        # 3. Das fertig bemalte Bild anzeigen
        self.image_label.setPixmap(pixmap)

    def upload_cad(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Wähle CAD Modell", "", "OBJ Files (*.obj);;All Files (*)")
        if file_name:
            self.cad_path = file_name
            self.btn_cad.setText("✅ CAD Geladen")
            self.btn_cad.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px; font-weight: bold;")

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            # Farbe für die Maske aktualisieren (Transparenz wieder auf 100 setzen)
            self.mask_color = color
            self.mask_color.setAlpha(100) 
            
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: black; padding: 10px; border-radius: 5px; font-weight: bold;")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClientApp()
    window.show()
    sys.exit(app.exec())