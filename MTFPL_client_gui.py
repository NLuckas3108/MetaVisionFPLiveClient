import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import zmq
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QFileDialog, QColorDialog)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint

# --- 3D Visualization ---
import pyvista as pv
from pyvistaqt import QtInteractor

# --- 1. Eigene Label-Klasse ---
class ClickableVideoLabel(QLabel):
    on_click = pyqtSignal(int, int)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)

# --- 2. Result Receiver Thread (PULL) ---
# Muss VOR ClientApp definiert sein!
class ResultReceiver(QThread):
    new_result = pyqtSignal(QImage)
    
    def __init__(self, ip):
        super().__init__()
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect(f"tcp://{ip}:5557") # Neuer Port OUT vom Proxy
        self.socket.setsockopt(zmq.RCVTIMEO, 1000) # 1s Timeout check

    def run(self):
        while self.running:
            try:
                packet = self.socket.recv_pyobj()
                img_data = packet["image"]
                
                # Konvertierung für QT
                rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                self.new_result.emit(qt_img)
            except zmq.Again:
                continue
            except Exception:
                pass
    
    def stop(self):
        self.running = False
        self.wait()

# --- 3. Kamera Thread (PUSH) ---
class RealSenseThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.tracking_active = False 
        self.pipeline = None
        
        # ZMQ Video Connection (PUSH Socket!)
        self.server_ip = "192.168.10.52" # <--- IP PRÜFEN!
        self.context = zmq.Context()
        self.video_socket = self.context.socket(zmq.PUSH) # PUSH für Fire & Forget
        self.video_socket.connect(f"tcp://{self.server_ip}:5556")
        
        # WICHTIG: Queue extrem klein halten (High Water Mark).
        self.video_socket.setsockopt(zmq.SNDHWM, 1) 

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        try:
            self.pipeline.start(config)
            print("[CLIENT] Kamera läuft (Async Mode).")
            
            while self._run_flag:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame: continue

                cv_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())

                # --- 1. NETZWERK SENDEN (Asynchron) ---
                if self.tracking_active:
                    try:
                        # flags=zmq.NOBLOCK ist entscheidend!
                        payload = {"rgb": cv_img, "depth": depth_img}
                        self.video_socket.send_pyobj(payload, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass # Puffer voll, Frame droppen
                    except Exception as e:
                        print(f"[ERROR] ZMQ Send: {e}")

                # --- 2. LOKALE ANZEIGE ---
                # Nur anzeigen, wenn wir NICHT tracken. Sonst kommt Bild vom Server.
                if not self.tracking_active:
                    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    
                    qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    p = qt_img.scaled(640, 480, Qt.AspectRatioMode.IgnoreAspectRatio)
                    
                    self.change_pixmap_signal.emit(p)

        except Exception as e:
            print(f"[ERROR] Pipeline: {e}")
        finally:
            self.stop_pipeline()
            self.video_socket.close()
            self.context.term()

    def stop(self):
        self._run_flag = False
        self.wait()

    def stop_pipeline(self):
        if self.pipeline:
            try: self.pipeline.stop()
            except: pass

# --- 4. CAD Preview Widget ---
class CADPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#000000") 
        self.plotter.view_isometric()
        self.mesh_actor = None

    def load_mesh(self, file_path, initial_qcolor=None):
        try:
            self.plotter.clear()
            mesh = pv.read(file_path)
            c = initial_qcolor.name() if initial_qcolor else "lightgrey"
            self.mesh_actor = self.plotter.add_mesh(mesh, color=c, pbr=True, metallic=0.5)
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as e:
            print(f"[ERROR] CAD Preview: {e}")

    def update_color(self, qcolor):
        if self.mesh_actor:
            self.mesh_actor.prop.color = qcolor.name()
            self.plotter.render()

# --- 5. Die GUI App ---
class ClientApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoundationPose Client")
        self.setGeometry(100, 100, 1000, 650)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # ZMQ Command Setup (Bleibt Synchron REQ-REP)
        self.server_ip = "192.168.10.52"
        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.REQ)
        self.cmd_socket.connect(f"tcp://{self.server_ip}:5555")
        self.cmd_socket.setsockopt(zmq.RCVTIMEO, 15000)

        # Status Flags
        self.status_cad = False
        self.status_color = False
        self.status_mask = False
        self.drawing_mode = False
        self.mask_points = []
        self.mask_color = QColor(0, 255, 0, 255) 

        # UI Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.image_label = ClickableVideoLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: #000;")
        self.image_label.on_click.connect(self.handle_image_click)

        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        btn_style = "QPushButton { background-color: #444; border-radius: 5px; padding: 10px; font-weight: bold; }"
        
        self.btn_cad = QPushButton("1. 📂 Upload CAD Model")
        self.btn_cad.setStyleSheet(btn_style)
        self.btn_cad.clicked.connect(self.upload_cad)
        
        self.btn_color = QPushButton("2. 🎨 Pick Color")
        self.btn_color.setStyleSheet(f"background-color: {self.mask_color.name()}; color: black; padding: 10px; border-radius: 5px; font-weight: bold;")
        self.btn_color.clicked.connect(self.pick_color)

        self.btn_mask = QPushButton("3. ✏️ Draw Mask")
        self.btn_mask.setStyleSheet(btn_style)
        self.btn_mask.clicked.connect(self.start_drawing_mode)

        self.sidebar_layout.addSpacing(20)
        self.btn_start = QPushButton("🚀 Start Tracking")
        self.style_disabled = "background-color: #555; color: #888; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"
        self.style_start = "background-color: #2196F3; color: white; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"
        self.style_stop = "background-color: #d32f2f; color: white; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"

        self.btn_start.setStyleSheet(self.style_disabled)
        self.btn_start.setEnabled(False) 
        self.btn_start.clicked.connect(self.toggle_tracking)

        self.cad_preview = CADPreviewWidget()
        self.cad_preview.setMinimumSize(200, 200)

        self.sidebar_layout.addWidget(self.btn_cad)
        self.sidebar_layout.addWidget(self.btn_color)
        self.sidebar_layout.addWidget(self.btn_mask)
        self.sidebar_layout.addWidget(self.btn_start)
        self.sidebar_layout.addWidget(QLabel("Preview:"))
        self.sidebar_layout.addWidget(self.cad_preview)
        self.sidebar_layout.addStretch()

        self.main_layout.addWidget(self.image_label)
        self.main_layout.addLayout(self.sidebar_layout)

        # 1. Thread: Kamera (Producer)
        self.thread = RealSenseThread()
        self.thread.change_pixmap_signal.connect(self.update_image) # Lokales Bild
        self.thread.start()

        # 2. Thread: Ergebnis (Consumer)
        self.result_receiver = ResultReceiver(self.thread.server_ip)
        self.result_receiver.new_result.connect(self.update_image) # Server Bild
        self.result_receiver.start()

    def check_ready_status(self):
        if self.thread.tracking_active: return 
        if self.status_cad and self.status_color and self.status_mask:
            self.btn_start.setEnabled(True)
            self.btn_start.setText("🚀 Start Tracking")
            self.btn_start.setStyleSheet(self.style_start)
        else:
            self.btn_start.setEnabled(False)
            self.btn_start.setText("🚀 Start Tracking")
            self.btn_start.setStyleSheet(self.style_disabled)

    def toggle_tracking(self):
        """Schaltet Tracking AN oder AUS und resettet den Server"""
        if not self.thread.tracking_active:
            # --- STARTEN ---
            self.thread.tracking_active = True
            self.btn_start.setText("🛑 Stop Tracking")
            self.btn_start.setStyleSheet(self.style_stop)
        else:
            # --- STOPPEN ---
            self.thread.tracking_active = False
            self.btn_start.setText("🚀 Start Tracking")
            self.btn_start.setStyleSheet(self.style_start)
            
            # NEU: Server Reset befehlen
            print("[CLIENT] Sende STOP an Server...")
            try:
                # Wir nutzen den synchronen CMD Socket
                self.cmd_socket.send_pyobj({"cmd": "STOP"})
                resp = self.cmd_socket.recv_string()
                
                if resp == "OK":
                    print("[CLIENT] Server erfolgreich resettet.")
                    # Optional: Status Flags zurücksetzen, falls man
                    # gezwungen werden soll, neu zu maskieren:
                    # self.status_mask = False
                    # self.check_ready_status()
                else:
                    print(f"[CLIENT] Server Warnung: {resp}")
                    
            except Exception as e:
                print(f"[CLIENT] Fehler beim Stoppen: {e}")

    def upload_cad(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "CAD Modell wählen", "", "OBJ Files (*.obj)")
        if file_path:
            self.btn_cad.setText("⏳ Uploading...")
            self.cad_preview.load_mesh(file_path, self.mask_color)
            
            try:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                payload = {"cmd": "UPLOAD_CAD", "data": file_data, "filename": file_path.split("/")[-1]}
                self.cmd_socket.send_pyobj(payload)
                resp = self.cmd_socket.recv_string()
                
                self.btn_cad.setText("✅ CAD Uploaded")
                self.btn_cad.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px;")
                self.status_cad = True
                self.check_ready_status()
            except Exception as e:
                print(f"ZMQ Error: {e}")
                self.btn_cad.setText("❌ Upload Failed")

    def handle_image_click(self, x, y):
        if not self.drawing_mode: return
        self.mask_points.append((x, y))
        
        if len(self.mask_points) == 1:
            self.btn_mask.setText("Click Point 2...")
        elif len(self.mask_points) == 2:
            self.drawing_mode = False
            self.btn_mask.setText("✅ Mask Ready")
            self.btn_mask.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px;")
            
            profile = self.thread.pipeline.get_active_profile()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
            
            try:
                self.cmd_socket.send_pyobj({"cmd": "SET_MASK", "points": self.mask_points, "K": K})
                self.cmd_socket.recv_string()
                self.status_mask = True
                self.check_ready_status()
            except: print("ZMQ Error sending mask")

    def start_drawing_mode(self):
        if self.thread.tracking_active:
            self.toggle_tracking()
        self.drawing_mode = True
        self.mask_points = []
        self.btn_mask.setText("Click Point 1...")
        self.btn_mask.setStyleSheet("background-color: #d32f2f; padding: 10px; border-radius: 5px;")
        self.status_mask = False 
        self.check_ready_status()

    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        # Zeichne Maske nur, wenn wir gerade zeichnen oder nicht tracken
        if not self.thread.tracking_active or self.drawing_mode:
            painter = QPainter(pixmap)
            if len(self.mask_points) == 1:
                painter.setBrush(self.mask_color)
                painter.drawEllipse(QPoint(self.mask_points[0][0], self.mask_points[0][1]), 4, 4)
            elif len(self.mask_points) == 2:
                p1, p2 = self.mask_points
                x, y = min(p1[0], p2[0]), min(p1[1], p2[1])
                w, h = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
                m_color = QColor(self.mask_color); m_color.setAlpha(100)
                painter.setBrush(m_color)
                painter.drawRect(x, y, w, h)
            painter.end()
        self.image_label.setPixmap(pixmap)

    def pick_color(self):
        color = QColorDialog.getColor(initial=self.mask_color)
        if color.isValid():
            self.mask_color = color
            self.btn_color.setText("✅ Color Selected")
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: black; padding: 10px; border-radius: 5px; font-weight: bold;")
            self.cad_preview.update_color(self.mask_color)
            self.status_color = True
            self.check_ready_status()

    def closeEvent(self, event):
        self.thread.stop()
        self.result_receiver.stop() 
        self.cad_preview.plotter.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClientApp()
    window.show()
    sys.exit(app.exec())