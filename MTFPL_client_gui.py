import sys
import cv2
import time
import numpy as np
import pyrealsense2 as rs
import zmq
import zlib
import os
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QFileDialog, 
                             QColorDialog, QDialog, QLineEdit, QMessageBox,
                             QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont, QIcon
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint, QSize

import pyvista as pv
from pyvistaqt import QtInteractor

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class TextureSelectorDialog(QDialog):
    def __init__(self, texture_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Textur wählen")
        self.setFixedSize(600, 400)
        self.setStyleSheet("background-color: #333; color: white;")
        self.selected_texture_name = None
        self.selected_texture_bytes = None 
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Verfügbare Texturen (aus Cache oder Server):"))
        
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(100, 100))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setStyleSheet("background-color: #444; border: 1px solid #666;")
        
        self.tex_data_map = {}

        for tex in texture_list:
            name = tex["name"]
            thumb_bytes = tex["thumbnail"]
            self.tex_data_map[name] = thumb_bytes 
            
            pixmap = QPixmap()
            pixmap.loadFromData(thumb_bytes)
            icon = QIcon(pixmap)
            
            item = QListWidgetItem(icon, name)
            item.setSizeHint(QSize(120, 140))
            self.list_widget.addItem(item)
            
        self.list_widget.itemDoubleClicked.connect(self.accept_selection)
        layout.addWidget(self.list_widget)
        
        btn_select = QPushButton("Auswählen")
        btn_select.setStyleSheet("background-color: #00549F; color: white; padding: 10px; font-weight: bold;")
        btn_select.clicked.connect(self.accept_selection)
        layout.addWidget(btn_select)

    def accept_selection(self, arg=None):
        item = None
        if isinstance(arg, QListWidgetItem):
            item = arg
        else:
            items = self.list_widget.selectedItems()
            if items:
                item = items[0]
        
        if item:
            self.selected_texture_name = item.text()
            self.selected_texture_bytes = self.tex_data_map.get(self.selected_texture_name)
            self.accept()
        else:
            QMessageBox.warning(self, "Info", "Bitte wähle eine Textur aus.")

# --- Verbindung zum Proxy ---
class ManualConnectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Server Verbindung")
        self.setFixedSize(350, 150)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Bitte Server-IP eingeben:", self))
        self.ip_input = QLineEdit("") 
        self.layout.addWidget(self.ip_input)
        
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.layout.addWidget(self.status_label)
        
        self.btn_connect = QPushButton("Verbindung prüfen")
        self.btn_connect.clicked.connect(self.verify_connection)
        self.layout.addWidget(self.btn_connect)
        self.entered_ip = None

    def verify_connection(self):
        ip = self.ip_input.text().strip()
        if not ip: return
        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Prüfe Verbindung...")
        self.status_label.setText(f"Sende Ping an {ip}:5555 ...")
        QApplication.processEvents() 

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 2000)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.connect(f"tcp://{ip}:5555")
            socket.send_pyobj({"cmd": "PING"})
            reply = socket.recv_string()
            self.status_label.setText("✅ Server gefunden!")
            self.status_label.setStyleSheet("color: green;")
            self.entered_ip = ip
            socket.close()
            context.term()
            super().accept()
        except zmq.Again:
            self.show_error("Timeout", f"Der Server unter {ip} antwortet nicht.")
        except Exception as e:
            self.show_error("Fehler", f"Verbindungsfehler: {e}")
        finally:
            if not socket.closed: socket.close(); context.term()
            self.btn_connect.setEnabled(True); self.btn_connect.setText("Verbinden & Prüfen")

    def show_error(self, title, msg):
        self.status_label.setText("❌ Verbindung fehlgeschlagen")
        self.status_label.setStyleSheet("color: red;")
        QMessageBox.warning(self, title, msg)

class ClickableVideoLabel(QLabel):
    on_click = pyqtSignal(int, int)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)

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
            except zmq.Again: continue
            except Exception: pass
    def stop(self):
        self.running = False; self.wait()

class RealSenseThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    connection_error_signal = pyqtSignal(str)
    intrinsics_signal = pyqtSignal(object)
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
            profile = self.pipeline.start(config)
            print(f"[CLIENT] Kamera läuft. Sende an {self.server_ip}")
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            self.intrinsics_signal.emit(K)
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
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] 
                        _, rgb_encoded = cv2.imencode('.jpg', cv_img, encode_param)
                        depth_bytes = depth_img.tobytes()
                        depth_compressed = zlib.compress(depth_bytes, level=1)
                        payload = {"rgb_compressed": rgb_encoded, "depth_compressed": depth_compressed, "shape": depth_img.shape, "dtype": str(depth_img.dtype)}
                        self.video_socket.send_pyobj(payload, flags=zmq.NOBLOCK)
                    except zmq.Again: pass 
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = qt_img.scaled(640, 480, Qt.AspectRatioMode.IgnoreAspectRatio)
                self.change_pixmap_signal.emit(p)
        except Exception as e:
            print(f"[ERROR] Pipeline: {e}")
            self.connection_error_signal.emit(f"Kamera Fehler: {e}")
        finally:
            self.stop_pipeline(); self.video_socket.close(); self.context.term()
    def stop(self):
        self._run_flag = False; self.wait()
    def stop_pipeline(self):
        if self.pipeline:
            try: self.pipeline.stop()
            except: pass

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
        self.current_mesh = None

    def load_mesh(self, file_path, initial_qcolor=None):
        try:
            self.plotter.clear()
            self.current_mesh = pv.read(file_path)
            
            c = initial_qcolor.name() if initial_qcolor else "lightgrey"
            
            self.mesh_actor = self.plotter.add_mesh(
                self.current_mesh, 
                color=c, 
                smooth_shading=True,
                specular=0.5
            )
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as e:
            print(f"[ERROR] CAD Preview: {e}")

    def update_color(self, qcolor):
        if self.mesh_actor:
            self.mesh_actor.texture = None
            self.mesh_actor.prop.color = qcolor.name()
            self.plotter.render()

    def update_texture(self, texture_bytes):
        if self.mesh_actor and texture_bytes and self.current_mesh:
            try:
                nparr = np.frombuffer(texture_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tex = pv.numpy_to_texture(img)
                
                if hasattr(self.current_mesh, "texture_map_to_box"):
                    mapped_mesh = self.current_mesh.texture_map_to_box()
                else:
                    mapped_mesh = self.current_mesh.texture_map_to_plane()
                
                self.mesh_actor.mapper.dataset = mapped_mesh
                self.mesh_actor.texture = tex
                self.mesh_actor.prop.color = "white"
                self.plotter.render()
            except Exception as e:
                print(f"[ERROR] Preview Texture Error: {e}")

class ClientApp(QMainWindow):
    def __init__(self, server_ip):
        super().__init__()
        self.server_ip = server_ip 
        
        self.setWindowTitle(f"MetaVision Client (Connected to {server_ip})")
        logo_path = resource_path(os.path.join("logo", "logo_weiss.png"))
        self.setWindowIcon(QIcon(logo_path))
        self.setGeometry(100, 100, 1000, 650)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: rgb(0, 84, 159);
            }
            QWidget {
                background-color: rgb(0, 84, 159);
                color: white;
            }
            QLabel {
                background-color: transparent;
                color: white;
            }
        """)
        
        self.btn_style_unified = """
            QPushButton { 
                background-color: rgb(100, 130, 160); 
                color: white;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 5px; 
                padding: 10px; 
                font-weight: bold; 
            }
            QPushButton:hover {
                background-color: rgb(120, 150, 180);
            }
            QPushButton:pressed {
                background-color: rgb(0, 70, 130);
            }
            /* Damit man sieht, wenn etwas noch nicht klickbar ist */
            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 0.1);
                color: #888;
                border: 1px solid transparent;
            }
        """

        self.current_box_points = None
        self.current_pose = None
        self.K = None
        self.tracking_fps_buffer = deque()
        self.tracking_fps = 0
        self.pose_log = []      
        self.image_counter = 0
        self.texture_cache = None

        self.context = zmq.Context()
        self.cmd_socket = self.context.socket(zmq.REQ)
        self.cmd_socket.connect(f"tcp://{self.server_ip}:5555")
        self.cmd_socket.setsockopt(zmq.RCVTIMEO, 15000)

        self.status_cad = False
        self.status_appearance = False 
        self.status_mask = False
        self.drawing_mode = False
        self.mask_points = []
        self.mask_color = QColor(255, 255, 255, 255)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.left_container = QWidget()
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = ClickableVideoLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid white; background-color: #000;")
        self.image_label.on_click.connect(self.handle_image_click)
        self.left_layout.addWidget(self.image_label)

        self.logo_label = QLabel()
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path)
            self.logo_label.setPixmap(pix.scaledToHeight(60, Qt.TransformationMode.SmoothTransformation))
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.left_layout.addWidget(self.logo_label)
        
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.btn_cad = QPushButton("1. 📂 Upload CAD Model")
        self.btn_cad.setStyleSheet(self.btn_style_unified)
        self.btn_cad.clicked.connect(self.upload_cad)
        self.sidebar_layout.addWidget(self.btn_cad)
        
        self.appearance_layout = QHBoxLayout()
        
        self.btn_color = QPushButton("2a. 🎨 Color")
        self.btn_color.setStyleSheet(self.btn_style_unified)
        self.btn_color.clicked.connect(self.pick_color)
        
        self.btn_texture = QPushButton("2b. 🖼️ Texture")
        self.btn_texture.setStyleSheet(self.btn_style_unified)
        self.btn_texture.clicked.connect(self.open_texture_dialog)
        
        self.appearance_layout.addWidget(self.btn_color)
        self.appearance_layout.addWidget(self.btn_texture)
        self.sidebar_layout.addLayout(self.appearance_layout)

        self.btn_mask = QPushButton("3. ✏️ Draw Mask")
        self.btn_mask.setStyleSheet(self.btn_style_unified)
        self.btn_mask.clicked.connect(self.start_drawing_mode)
        self.sidebar_layout.addWidget(self.btn_mask)

        self.sidebar_layout.addSpacing(20)

        self.btn_start = QPushButton("🚀 Start Tracking")
        self.style_disabled = "background-color: #555; color: #888; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"
        self.style_start = "background-color: #0098A1; color: white; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;" 
        self.style_stop = "background-color: #CC071E; color: white; border-radius: 5px; padding: 15px; font-weight: bold; font-size: 16px;"

        self.btn_start.setStyleSheet(self.btn_style_unified)
        self.btn_start.setEnabled(False) 
        self.btn_start.clicked.connect(self.toggle_tracking)
        self.sidebar_layout.addWidget(self.btn_start)

        self.sidebar_layout.addSpacing(10)
        
        self.btn_log = QPushButton("💾 Download Log")
        self.btn_log.setStyleSheet(self.btn_style_unified)
        self.btn_log.setEnabled(False)
        self.btn_log.clicked.connect(self.save_log_file)
        self.sidebar_layout.addWidget(self.btn_log)
        
        self.sidebar_layout.addStretch()
        self.sidebar_layout.addWidget(QLabel("Preview:"))
        self.cad_preview = CADPreviewWidget()
        self.cad_preview.setMinimumSize(200, 200)
        self.sidebar_layout.addWidget(self.cad_preview)

        self.main_layout.addWidget(self.left_container)
        self.main_layout.addLayout(self.sidebar_layout)

        self.thread = RealSenseThread(self.server_ip)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.connection_error_signal.connect(self.show_camera_error)
        self.thread.intrinsics_signal.connect(self.set_intrinsics)
        self.thread.start()

        self.result_receiver = ResultReceiver(self.server_ip)
        self.result_receiver.new_result.connect(self.update_box_points) 
        self.result_receiver.start()

    def show_camera_error(self, msg):
        QMessageBox.critical(self, "Kamera Fehler", msg)
        self.close()

    def set_intrinsics(self, K): self.K = K
    
    def check_ready_status(self):
        if self.thread.tracking_active: return 
        if self.status_cad and self.status_appearance and self.status_mask:
            self.btn_start.setEnabled(True)
            self.btn_start.setText("🚀 Start Tracking")
            #self.btn_start.setStyleSheet(self.style_start)
        else:
            self.btn_start.setEnabled(False)
            self.btn_start.setText("🚀 Start Tracking")
            #self.btn_start.setStyleSheet(self.style_disabled)

    def toggle_tracking(self):
        self.current_box_points = None
        self.current_pose = None

        if not self.thread.tracking_active:
            self.pose_log = []      
            self.image_counter = 0 
            self.tracking_fps_buffer.clear()
            self.tracking_fps = 0
            
            self.btn_log.setEnabled(False) 
            
            self.thread.tracking_active = True
            self.btn_start.setText("🛑 Stop Tracking")
            
            self.btn_cad.setEnabled(False); self.btn_mask.setEnabled(False)
            self.btn_color.setEnabled(False); self.btn_texture.setEnabled(False)
        else:
            self.thread.tracking_active = False
            self.tracking_fps = 0
            self.btn_start.setText("🚀 Start Tracking") # Nur Text ändert sich
            
            self.btn_cad.setEnabled(True); self.btn_mask.setEnabled(True)
            self.btn_color.setEnabled(True); self.btn_texture.setEnabled(True)
            self.btn_log.setEnabled(True)
            
            print("[CLIENT] Resetting Mask State...")
            self.status_mask = False
            self.mask_points = []
            self.btn_mask.setText("3. ✏️ Draw Mask")
            
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
                with open(file_path, "rb") as f: file_data = f.read()
                filename = os.path.basename(file_path)
                payload = {"cmd": "UPLOAD_CAD", "data": file_data, "filename": filename}
                self.cmd_socket.send_pyobj(payload)
                self.cmd_socket.recv_string()
                self.btn_cad.setText("✅ CAD Uploaded")
                #self.btn_cad.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px;")
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
            #self.btn_mask.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px;")
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
        #self.btn_mask.setStyleSheet("background-color: #d32f2f; padding: 10px; border-radius: 5px;")
        self.status_mask = False 
        self.check_ready_status()

    def open_texture_dialog(self):
        if self.texture_cache is None:
            self.btn_texture.setText("⏳ Loading List...")
            QApplication.processEvents()
        
        textures = []
        
        try:
            if self.texture_cache is not None:
                textures = self.texture_cache
            else:
                self.cmd_socket.send_pyobj({"cmd": "GET_TEXTURES"})
                response = self.cmd_socket.recv_pyobj()
                if response.get("status") == "OK":
                    textures = response.get("textures", [])
                    textures = sorted(textures, key=lambda x: x['name'].lower())
                    self.texture_cache = textures
                else:
                    self.btn_texture.setText("❌ Error")
                    return

            if not textures:
                QMessageBox.information(self, "Info", "Keine Texturen gefunden.")
                self.btn_texture.setText("2b. 🖼️ Texture")
                self.btn_texture.setStyleSheet(self.btn_style_unified)
                return

            dlg = TextureSelectorDialog(textures, self)
            
            if dlg.exec():
                selected_name = dlg.selected_texture_name
                
                self.btn_texture.setText("⏳ Downloading HD...")
                QApplication.processEvents()
                
                print(f"[CLIENT] Frage High-Res Textur an: {selected_name}")
                self.cmd_socket.send_pyobj({"cmd": "GET_TEXTURE_FULL", "name": selected_name})
                high_res_resp = self.cmd_socket.recv_pyobj()
                
                if high_res_resp.get("status") == "OK":
                    full_bytes = high_res_resp["data"]
                    self.cad_preview.update_texture(full_bytes)
                else:
                    print("[WARN] Konnte High-Res Textur nicht laden, nutze Thumbnail.")
                    if dlg.selected_texture_bytes:
                         self.cad_preview.update_texture(dlg.selected_texture_bytes)

                self.cmd_socket.send_pyobj({"cmd": "SET_TEXTURE", "name": selected_name})
                resp = self.cmd_socket.recv_string()
                
                if resp == "OK" or "NO MESH" in resp:
                    self.btn_texture.setText(f"✅ {selected_name}")
                    #.btn_texture.setStyleSheet("background-color: #2e7d32; padding: 10px; border-radius: 5px;")
                    self.status_appearance = True
                    self.check_ready_status()
                else:
                    QMessageBox.warning(self, "Fehler", f"Server Fehler: {resp}")
            else:
                if not self.status_appearance:
                    self.btn_texture.setText("2b. 🖼️ Texture")
                    self.btn_texture.setStyleSheet(self.btn_style_unified)
                else:
                    pass

        except Exception as e:
            print(f"Texture Error: {e}")
            self.btn_texture.setText("❌ Timeout")

    def pick_color(self):
        color = QColorDialog.getColor(initial=self.mask_color)
        if color.isValid():
            self.mask_color = color
            self.btn_color.setText("✅ Color")
            #self.btn_color.setStyleSheet(f"background-color: {color.name()}; color: black; padding: 10px; border-radius: 5px; font-weight: bold;")
            
            self.btn_texture.setText("2b. 🖼️ Texture")
            self.btn_texture.setStyleSheet(self.btn_style_unified)
            
            self.cad_preview.update_color(self.mask_color)
            self.status_appearance = True
            self.check_ready_status()

    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        painter = QPainter(pixmap)
        def project(p_3d, pose, K):
            if pose is None: return None
            R = pose[:3, :3]; t = pose[:3, 3]
            p_cam = (R @ p_3d) + t
            if p_cam[2] <= 0.001: return None
            p_img = K @ p_cam
            return (int(p_img[0]/p_img[2]), int(p_img[1]/p_img[2]))

        if self.thread.tracking_active and self.current_box_points:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            pts = self.current_box_points
            if len(pts) == 8:
                lines = [(0,1), (1,3), (3,2), (2,0), (4,5), (5,7), (7,6), (6,4), (0,4), (1,5), (2,6), (3,7)]
                for p1, p2 in lines:
                    painter.drawLine(pts[p1][0], pts[p1][1], pts[p2][0], pts[p2][1])
        
        if self.thread.tracking_active and self.current_pose is not None and self.K is not None:
            origin = np.array([0.,0.,0.])
            p_org = project(origin, self.current_pose, self.K)
            if p_org:
                cols = [QColor(255,0,0), QColor(0,255,0), QColor(0,0,255)]
                axes = [np.array([0.1,0,0]), np.array([0,0.1,0]), np.array([0,0,0.1])]
                for ax, col in zip(axes, cols):
                    p_end = project(ax, self.current_pose, self.K)
                    if p_end:
                        painter.setPen(QPen(col, 3))
                        painter.drawLine(p_org[0], p_org[1], p_end[0], p_end[1])

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
        self.current_pose = pose
        
        now = time.time()
        self.tracking_fps_buffer.append(now)
        while self.tracking_fps_buffer and self.tracking_fps_buffer[0] < now - 1.0:
            self.tracking_fps_buffer.popleft()
        self.tracking_fps = len(self.tracking_fps_buffer)
        
        if self.thread.tracking_active:
            self.image_counter += 1
            self.pose_log.append({"id": self.image_counter, "ts": timestamp, "pose": pose})

    def save_log_file(self):
        if not self.pose_log: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Log speichern", "tracking.log", "Log Files (*.log)")
        if not file_path: return
        try:
            with open(file_path, "w") as f:
                for entry in self.pose_log:
                    f.write(f"Image: {entry['id']}\n")
                    f.write(f"Timestamp: {entry['ts']:.6f}\n")
                    for row in entry['pose']:
                        f.write("".join([f"[{x: .15f}] " for x in row]).strip() + "\n")
                    f.write("\n")
            self.btn_log.setText("✅ Saved")
        except Exception as e: print(e)

    def closeEvent(self, event):
        self.thread.stop()
        self.result_receiver.stop() 
        self.cad_preview.plotter.close()
        event.accept()

if __name__ == "__main__":
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    dialog = ManualConnectDialog()
    if dialog.exec(): 
        window = ClientApp(dialog.entered_ip)
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)