import sys
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import zmq
import zlib
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QFileDialog, 
                             QColorDialog, QDialog, QLineEdit, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint

# --- 3D Visualization ---
import pyvista as pv
from pyvistaqt import QtInteractor

# Klasse zum Aufbauen der Verbindung zum Proxy
class ManualConnectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Server Verbindung")
        self.setFixedSize(350, 150)
        self.layout = QVBoxLayout(self)
        
        self.layout.addWidget(QLabel("Bitte Server-IP eingeben:", self))
        
        self.ip_input = QLineEdit("") 
        self.layout.addWidget(self.ip_input)
        
        # Status Label für Feedback
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.layout.addWidget(self.status_label)
        
        self.btn_connect = QPushButton("Verbindung prüfen")
        self.btn_connect.clicked.connect(self.verify_connection)
        self.layout.addWidget(self.btn_connect)
        
        self.entered_ip = None

    def verify_connection(self):
        ip = self.ip_input.text().strip()
        if not ip:
            return

        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Prüfe Verbindung...")
        self.status_label.setText(f"Sende Ping an {ip}:5555 ...")
        QApplication.processEvents() 

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        socket.setsockopt(zmq.LINGER, 0)
        
        try:
            # Verbindungsversuch
            socket.connect(f"tcp://{ip}:5555")
            socket.send_pyobj({"cmd": "PING"})
            reply = socket.recv_string()
            
            if reply == "PONG" or reply == "UNKNOWN": 
                self.status_label.setText("✅ Server gefunden!")
                self.status_label.setStyleSheet("color: green;")
                self.entered_ip = ip
                socket.close()
                context.term()
                super().accept()
                
            else:
                raise Exception(f"Unerwartete Antwort: {reply}")

        except zmq.Again:
            self.show_error("Timeout", f"Der Server unter {ip} antwortet nicht.\n\n- Läuft der Proxy?\n- Stimmt die IP?\n- Blockiert eine Firewall Port 5555?")
        except Exception as e:
            self.show_error("Fehler", f"Verbindungsfehler: {e}")
        finally:
            if not socket.closed:
                socket.close()
                context.term()
            
            self.btn_connect.setEnabled(True)
            self.btn_connect.setText("Verbinden & Prüfen")

    def show_error(self, title, msg):
        self.status_label.setText("❌ Verbindung fehlgeschlagen")
        self.status_label.setStyleSheet("color: red;")
        QMessageBox.warning(self, title, msg)

# Klasse ClickableVideo (Maskenerstellung)
class ClickableVideoLabel(QLabel):
    on_click = pyqtSignal(int, int)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)

# Klasse die die Ergebnisse des Backends empfängt
class ResultReceiver(QThread):
    new_result = pyqtSignal(list, np.ndarray, float)
    
    def __init__(self, ip):
        super().__init__()
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.connect(f"tcp://{ip}:5557") 
        self.socket.setsockopt(zmq.RCVTIMEO, 1000) 

    def run(self):
        while self.running:
            try:
                packet = self.socket.recv_pyobj()
                
                if "box_points" in packet and "pose" in packet:
                    points = packet["box_points"]
                    pose = packet["pose"]
                    timestamp = packet.get("timestamp", 0)
                    self.new_result.emit(points, pose, timestamp)
                
            except zmq.Again:
                continue
            except Exception:
                pass
    
    def stop(self):
        self.running = False
        self.wait()

# Klasse die die Kamerabilder aufnimmt und verarbeitet
class RealSenseThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, server_ip):
        super().__init__()
        self._run_flag = True
        self.tracking_active = False 
        self.pipeline = None
        self.align = rs.align(rs.stream.color)
        
        self.server_ip = server_ip
        self.context = zmq.Context()
        self.video_socket = self.context.socket(zmq.PUSH) 
        self.video_socket.connect(f"tcp://{self.server_ip}:5556")
        self.video_socket.setsockopt(zmq.SNDHWM, 1) 

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

        try:
            self.pipeline.start(config)
            print(f"[CLIENT] Kamera läuft. Sende an {self.server_ip}")
            
            while self._run_flag:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame: continue

                cv_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())

                if self.tracking_active:
                    try:
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                        _, rgb_encoded = cv2.imencode('.jpg', cv_img, encode_param)

                        depth_bytes = depth_img.tobytes()
                        depth_compressed = zlib.compress(depth_bytes, level=1)
                        
                        payload = {
                            "rgb_compressed": rgb_encoded,
                            "depth_compressed": depth_compressed,
                            "shape": depth_img.shape,
                            "dtype": str(depth_img.dtype)
                        }
                        self.video_socket.send_pyobj(payload, flags=zmq.NOBLOCK)
                    except zmq.Again:
                        pass 

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

# Klasse für Preview des geladenene Bauteils
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

# Klasse für die GUI an sich
class ClientApp(QMainWindow):
    def __init__(self, server_ip):
        super().__init__()
        self.server_ip = server_ip 
        self.setWindowTitle(f"FoundationPose Client (Connected to {server_ip})")
        self.setGeometry(100, 100, 1000, 650)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        self.current_box_points = None
        self.tracking_fps_buffer = deque()
        self.tracking_fps = 0
        self.pose_log = []      
        self.image_counter = 0

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

        self.btn_start = QPushButton("🚀 Start Tracking")
        self.style_disabled = "background-color: #555; color: #888; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"
        self.style_start = "background-color: #2196F3; color: white; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"
        self.style_stop = "background-color: #d32f2f; color: white; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"

        self.btn_start.setStyleSheet(self.style_disabled)
        self.btn_start.setEnabled(False) 
        self.btn_start.clicked.connect(self.toggle_tracking)

        self.btn_log = QPushButton("💾 Download Log")
        self.btn_log.setStyleSheet(self.style_disabled)
        self.btn_log.setEnabled(False)
        self.btn_log.clicked.connect(self.save_log_file)
        
        self.cad_preview = CADPreviewWidget()
        self.cad_preview.setMinimumSize(200, 200)

        self.sidebar_layout.addWidget(self.btn_cad)
        self.sidebar_layout.addWidget(self.btn_color)
        self.sidebar_layout.addWidget(self.btn_mask)
        self.sidebar_layout.addSpacing(20)
        self.sidebar_layout.addWidget(self.btn_start)
        self.sidebar_layout.addSpacing(10)
        self.sidebar_layout.addWidget(self.btn_log)
        self.sidebar_layout.addStretch()
        self.sidebar_layout.addWidget(QLabel("Preview:"))
        self.sidebar_layout.addWidget(self.cad_preview)

        self.main_layout.addWidget(self.image_label)
        self.main_layout.addLayout(self.sidebar_layout)

        # Threads
        self.thread = RealSenseThread(self.server_ip)
        self.thread.change_pixmap_signal.connect(self.update_image) 
        self.thread.start()

        self.result_receiver = ResultReceiver(self.server_ip)
        self.result_receiver.new_result.connect(self.update_box_points) 
        self.result_receiver.start()

        self.pose_log = []     
        self.image_counter = 0
    
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
        if not self.thread.tracking_active:
            self.pose_log = []      
            self.image_counter = 0 
            self.tracking_fps_buffer.clear()
            self.tracking_fps = 0
            self.btn_log.setEnabled(False) 
            self.btn_log.setStyleSheet(self.style_disabled)
            self.thread.tracking_active = True
            self.btn_start.setText("🛑 Stop Tracking")
            self.btn_start.setStyleSheet(self.style_stop)
            self.btn_cad.setEnabled(False)
            self.btn_mask.setEnabled(False)
            self.btn_color.setEnabled(False)
        else:
            self.thread.tracking_active = False
            self.current_box_points = None
            self.tracking_fps = 0
            self.btn_start.setText("🚀 Start Tracking")
            self.btn_cad.setEnabled(True)
            self.btn_mask.setEnabled(True)
            self.btn_color.setEnabled(True)
            self.btn_log.setEnabled(True)
            self.btn_log.setStyleSheet("background-color: #f57c00; color: white; border-radius: 5px; padding: 15px; font-weight: bold;")
            print("[CLIENT] Resetting Mask State...")
            self.status_mask = False
            self.mask_points = []
            self.btn_mask.setText("3. ✏️ Draw Mask")
            self.btn_mask.setStyleSheet("QPushButton { background-color: #444; border-radius: 5px; padding: 10px; font-weight: bold; }")
            self.check_ready_status()
            try:
                self.cmd_socket.send_pyobj({"cmd": "STOP"})
                self.cmd_socket.recv_string()
            except: pass

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
                self.cmd_socket.recv_string()
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
            except: pass

    def start_drawing_mode(self):
        if self.thread.tracking_active: self.toggle_tracking()
        self.current_box_points = None 
        self.drawing_mode = True
        self.mask_points = []
        self.btn_mask.setText("Click Point 1...")
        self.btn_mask.setStyleSheet("background-color: #d32f2f; padding: 10px; border-radius: 5px;")
        self.status_mask = False 
        self.check_ready_status()

    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        painter = QPainter(pixmap)
        if self.thread.tracking_active and self.current_box_points:
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            pts = self.current_box_points
            if len(pts) == 8:
                lines = [(0,1), (1,3), (3,2), (2,0), (4,5), (5,7), (7,6), (6,4), (0,4), (1,5), (2,6), (3,7)]
                for p1, p2 in lines:
                    painter.drawLine(pts[p1][0], pts[p1][1], pts[p2][0], pts[p2][1])
        if not self.thread.tracking_active or self.drawing_mode:
            if len(self.mask_points) == 1:
                painter.setBrush(self.mask_color); painter.drawEllipse(QPoint(self.mask_points[0][0], self.mask_points[0][1]), 4, 4)
            elif len(self.mask_points) == 2:
                p1, p2 = self.mask_points
                x, y = min(p1[0], p2[0]), min(p1[1], p2[1])
                w, h = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
                m_color = QColor(self.mask_color); m_color.setAlpha(100)
                painter.setBrush(m_color); painter.drawRect(x, y, w, h)
        if self.thread.tracking_active:
            painter.setPen(QColor("yellow")); painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            painter.drawText(10, 30, f"FPS: {self.tracking_fps}")
        painter.end()
        self.image_label.setPixmap(pixmap)

    def update_box_points(self, points, pose, timestamp):
        self.current_box_points = points
        now = time.time()
        self.tracking_fps_buffer.append(now)
        while self.tracking_fps_buffer and self.tracking_fps_buffer[0] < now - 1.0:
            self.tracking_fps_buffer.popleft()
        self.tracking_fps = len(self.tracking_fps_buffer)
        if self.thread.tracking_active:
            self.image_counter += 1
            self.pose_log.append({"id": self.image_counter, "ts": timestamp, "pose": pose})

    def pick_color(self):
        color = QColorDialog.getColor(initial=self.mask_color)
        if color.isValid():
            self.mask_color = color
            self.btn_color.setText("✅ Color Selected")
            self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: black; padding: 10px; border-radius: 5px; font-weight: bold;")
            self.cad_preview.update_color(self.mask_color)
            self.status_color = True
            self.check_ready_status()

    def save_log_file(self):
        if not self.pose_log: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Log speichern", "tracking.log", "Log Files (*.log)")
        if not file_path: return
        try:
            with open(file_path, "w") as f:
                for entry in self.pose_log:
                    f.write(f"Image: {entry['id']}\n")
                    ts = entry['ts']; seconds = int(ts); nanos = int((ts - seconds) * 1_000_000_000)
                    f.write(f"Timestamp: {seconds}_{nanos:09d}\n")
                    for row in entry['pose']:
                        f.write("".join([f"[{x: .15f}] " for x in row]).strip() + "\n")
                    f.write("\n")
            print(f"Log gespeichert: {file_path}")
            self.btn_log.setText("✅ Saved")
        except Exception as e: print(e)

    def closeEvent(self, event):
        self.thread.stop()
        self.result_receiver.stop() 
        self.cad_preview.plotter.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ManualConnectDialog()
    
    if dialog.exec(): 
        server_ip = dialog.entered_ip
        
        # Main App
        window = ClientApp(server_ip)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)