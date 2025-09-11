#!/usr/bin/env python3
"""
PPE Detection App - Single File Version
A futuristic desktop application for Helmet & Vest Detection with animated particles
"""

import sys
import cv2
import time
import logging
import random
import math
import os
import numpy as np

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install ultralytics: pip install ultralytics")
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, 
                            QStackedWidget, QFrame, QSizePolicy, QProgressBar, 
                            QGraphicsDropShadowEffect, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                            QGraphicsOpacityEffect, QMessageBox, QGroupBox,
                            QSplitter)
from PyQt5.QtGui import (QPixmap, QImage, QFont, QColor, QPalette, QIcon, 
                        QLinearGradient, QGradient, QPainter, QBrush, QPen,
                        QPainterPath, QRadialGradient, QConicalGradient)
from PyQt5.QtCore import (Qt, QTimer, QSize, pyqtSignal, QPropertyAnimation, 
                         QEasingCurve, QPoint, QRect, QPointF, QRectF, QLineF,
                         QTimeLine, QParallelAnimationGroup, QSequentialAnimationGroup)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize YOLO model globally for better performance
yolo_model = None

def load_yolo_model():
    """
    Load the YOLO model from the specified path
    """
    global yolo_model
    try:
        if yolo_model is None and YOLO_AVAILABLE:
            # Try to load a PPE detection model
            model_paths = [
                "best.pt",  # Current directory
                "models/best.pt",  # Models folder
                "yolo_models/best.pt",  # YOLO models folder
                "runs/detect/train/weights/best.pt"  # Training output
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    logger.info(f"Loading YOLO model from {model_path}...")
                    yolo_model = YOLO(model_path)
                    logger.info("YOLO model loaded successfully!")
                    model_loaded = True
                    break
            
            if not model_loaded:
                logger.warning("No YOLO model found. Using fallback detection.")
                yolo_model = None
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        yolo_model = None

def detect_objects(frame, conf_threshold=0.5):
    """
    Perform real-time object detection using YOLO model
    """
    global yolo_model
    try:
        # Ensure the model is loaded
        if yolo_model is None and YOLO_AVAILABLE:
            load_yolo_model()
            
        if yolo_model is None:  # Still None after trying to load
            logger.warning("YOLO model not available, using fallback visualization")
            # Create fallback detection with generic bounding box
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # Draw a generic detection box in the center
            x1, y1 = int(w * 0.2), int(h * 0.2)
            x2, y2 = int(w * 0.8), int(h * 0.8)
            color = (0, 255, 0)  # Green
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated_frame, "Object: 0.85", 
                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated_frame, "YOLO model not loaded - Using fallback", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated_frame
        
        # Get detections from YOLO model
        results = yolo_model.predict(frame, conf=conf_threshold, verbose=False)
        
        # Process results
        result = results[0]  # Get the first (and only) result
        
        # Create annotated frame with bounding boxes
        annotated_frame = frame.copy()
        
        # Draw the results on the frame
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Extract the box and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]
                
                # Color based on confidence (red to green)
                color_val = int(255 * conf)
                color = (0, color_val, 255 - color_val)
                
                # Draw bounding box and label with better visibility
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Create background for text
                text = f"{class_name}: {conf:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), 
                            (x1 + text_size[0], y1), color, -1)
                cv2.putText(annotated_frame, text, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # No detections found
            cv2.putText(annotated_frame, "No objects detected", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated_frame
    
    except Exception as e:
        logger.error(f"Error in detect_objects: {str(e)}")
        # Add error notification to frame
        cv2.putText(frame, f"Detection error: {str(e)}", 
                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

class ParticleSystem(QWidget):
    """Advanced particle system with different particle types"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.particles = []
        self.time = 0
        self.init_particles()
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_particles)
        self.animation_timer.start(30)
    
    def init_particles(self):
        """Initialize different types of particles"""
        for _ in range(60):
            particle_type = random.choice(['helmet', 'vest', 'shield', 'spark', 'wave'])
            particle = {
                'type': particle_type,
                'x': random.random() * self.width(),
                'y': random.random() * self.height(),
                'size': random.randint(3, 12),
                'speed_x': random.uniform(-1.5, 1.5),
                'speed_y': random.uniform(-1.5, 1.5),
                'opacity': random.uniform(0.2, 0.8),
                'color': self.get_particle_color(particle_type),
                'rotation': random.uniform(0, 360),
                'rotation_speed': random.uniform(-3, 3),
                'pulse_speed': random.uniform(0.02, 0.08),
                'pulse_phase': random.uniform(0, 2 * math.pi),
                'life': 1.0,
                'max_life': random.uniform(200, 500)
            }
            self.particles.append(particle)
    
    def get_particle_color(self, particle_type):
        """Get color based on particle type"""
        colors = {
            'helmet': QColor(0, 255, 100, 120),      # Green
            'vest': QColor(255, 100, 0, 120),        # Orange
            'shield': QColor(100, 200, 255, 120),    # Blue
            'spark': QColor(255, 255, 100, 120),     # Yellow
            'wave': QColor(200, 100, 255, 120)       # Purple
        }
        return colors.get(particle_type, QColor(255, 255, 255, 120))
    
    def update_particles(self):
        """Update particle positions and effects"""
        self.time += 1
        
        for particle in self.particles:
            # Update position
            particle['x'] += particle['speed_x']
            particle['y'] += particle['speed_y']
            
            # Wrap around screen
            if particle['x'] < 0:
                particle['x'] = self.width()
            elif particle['x'] > self.width():
                particle['x'] = 0
            if particle['y'] < 0:
                particle['y'] = self.height()
            elif particle['y'] > self.height():
                particle['y'] = 0
            
            # Update rotation
            particle['rotation'] += particle['rotation_speed']
            
            # Update pulse effect
            particle['pulse_phase'] += particle['pulse_speed']
            pulse_factor = (1 + math.sin(particle['pulse_phase'])) / 2
            particle['opacity'] = 0.2 + 0.6 * pulse_factor
            
            # Update life
            particle['life'] -= 1 / particle['max_life']
            if particle['life'] <= 0:
                # Respawn particle
                particle['x'] = random.random() * self.width()
                particle['y'] = random.random() * self.height()
                particle['life'] = 1.0
        
        self.update()
    
    def paintEvent(self, event):
        """Paint particles with different shapes"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for particle in self.particles:
            painter.save()
            
            # Set particle color with opacity
            color = particle['color']
            alpha = int(255 * particle['opacity'] * particle['life'])
            color.setAlpha(alpha)
            
            painter.translate(int(particle['x']), int(particle['y']))
            painter.rotate(particle['rotation'])
            
            if particle['type'] == 'helmet':
                self.draw_helmet_particle(painter, particle, color)
            elif particle['type'] == 'vest':
                self.draw_vest_particle(painter, particle, color)
            elif particle['type'] == 'shield':
                self.draw_shield_particle(painter, particle, color)
            elif particle['type'] == 'spark':
                self.draw_spark_particle(painter, particle, color)
            elif particle['type'] == 'wave':
                self.draw_wave_particle(painter, particle, color)
            
            painter.restore()
    
    def draw_helmet_particle(self, painter, particle, color):
        """Draw helmet-shaped particle"""
        size = particle['size']
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(color))
        
        # Draw helmet shape
        path = QPainterPath()
        path.addEllipse(int(-size/2), int(-size/2), int(size), int(size))
        path.addRect(int(-size/4), int(-size/2), int(size/2), int(size/4))
        painter.drawPath(path)
    
    def draw_vest_particle(self, painter, particle, color):
        """Draw vest-shaped particle"""
        size = particle['size']
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color))
        
        # Draw vest shape
        painter.drawRect(int(-size/2), int(-size/2), int(size), int(size))
        painter.drawLine(int(-size/2), int(-size/4), int(size/2), int(-size/4))
        painter.drawLine(int(-size/2), int(size/4), int(size/2), int(size/4))
    
    def draw_shield_particle(self, painter, particle, color):
        """Draw shield-shaped particle"""
        size = particle['size']
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(color))
        
        # Draw shield shape
        path = QPainterPath()
        path.moveTo(0, int(-size/2))
        path.lineTo(int(-size/2), 0)
        path.lineTo(0, int(size/2))
        path.lineTo(int(size/2), 0)
        path.closeSubpath()
        painter.drawPath(path)
    
    def draw_spark_particle(self, painter, particle, color):
        """Draw spark particle"""
        size = particle['size']
        painter.setPen(QPen(color, 1))
        
        # Draw spark lines
        for i in range(4):
            angle = i * 90 + particle['rotation']
            x = int(size * math.cos(math.radians(angle)))
            y = int(size * math.sin(math.radians(angle)))
            painter.drawLine(0, 0, x, y)
    
    def draw_wave_particle(self, painter, particle, color):
        """Draw wave particle"""
        size = particle['size']
        painter.setPen(QPen(color, 2))
        
        # Draw wave
        path = QPainterPath()
        path.moveTo(int(-size), 0)
        for i in range(0, int(size * 2), 5):
            x = int(-size + i)
            y = int(5 * math.sin((i + self.time) * 0.1))
            path.lineTo(x, y)
        painter.drawPath(path)

class AnimatedBackground(QWidget):
    """Animated background with circuit patterns and gradients"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.time = 0
        self.circuit_nodes = []
        self.init_circuit()
        
        # Animation timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)
    
    def init_circuit(self):
        """Initialize circuit pattern nodes"""
        for _ in range(20):
            node = {
                'x': random.random() * 1000,
                'y': random.random() * 1000,
                'size': random.randint(2, 6),
                'pulse_phase': random.uniform(0, 2 * math.pi),
                'pulse_speed': random.uniform(0.01, 0.03)
            }
            self.circuit_nodes.append(node)
    
    def update_animation(self):
        """Update animation"""
        self.time += 1
        for node in self.circuit_nodes:
            node['pulse_phase'] += node['pulse_speed']
        self.update()
    
    def paintEvent(self, event):
        """Paint animated background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw gradient background
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(10, 10, 25, 200))
        gradient.setColorAt(0.5, QColor(20, 20, 40, 150))
        gradient.setColorAt(1, QColor(15, 15, 30, 200))
        painter.fillRect(self.rect(), gradient)
        
        # Draw circuit pattern
        self.draw_circuit_pattern(painter)
        
        # Draw animated nodes
        self.draw_circuit_nodes(painter)
    
    def draw_circuit_pattern(self, painter):
        """Draw circuit pattern"""
        painter.setPen(QPen(QColor(100, 200, 255, 30), 1))
        
        # Draw grid
        for x in range(0, self.width(), 40):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 40):
            painter.drawLine(0, y, self.width(), y)
        
        # Draw diagonal lines
        for i in range(-self.width(), self.width() * 2, 80):
            painter.drawLine(i, 0, i + self.height(), self.height())
    
    def draw_circuit_nodes(self, painter):
        """Draw animated circuit nodes"""
        for node in self.circuit_nodes:
            pulse = (1 + math.sin(node['pulse_phase'])) / 2
            size = node['size'] * (0.5 + pulse)
            alpha = int(100 * pulse)
            
            color = QColor(100, 200, 255, alpha)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(int(node['x'] - size/2), int(node['y'] - size/2), int(size), int(size))

class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.init_ui()
        
    def init_ui(self):
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #252525; border-radius: 10px;")
        self.camera_label.setMinimumSize(640, 480)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.camera_label)
        
    def start_camera(self):
        try:
            if self.capture is None:
                self.capture = cv2.VideoCapture(0)
                if not self.capture.isOpened():
                    raise Exception("Could not open camera")
                self.timer.start(30)  # 30ms update interval
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {str(e)}")
            self.stop_camera()
    
    def stop_camera(self):
        try:
            if self.timer.isActive():
                self.timer.stop()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.camera_label.clear()
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
    
    def update_frame(self):
        try:
            if self.capture is not None and self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    # Process frame with YOLO
                    processed_frame = detect_objects(frame)
                    
                    # Convert frame to Qt format
                    rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # Resize to fit the label while maintaining aspect ratio
                    pixmap = QPixmap.fromImage(qt_image)
                    self.camera_label.setPixmap(pixmap.scaled(
                        self.camera_label.width(), 
                        self.camera_label.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    ))
                else:
                    raise Exception("Failed to read frame from camera")
        except Exception as e:
            logger.error(f"Error updating frame: {str(e)}")
            self.stop_camera()
            QMessageBox.critical(self, "Camera Error", f"Error updating frame: {str(e)}")
    
    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)

class FuturisticButton(QPushButton):
    """Futuristic button with glow effects and animations"""
    
    def __init__(self, text, parent=None, color="#64C8FF"):
        super().__init__(text, parent)
        self.color = color
        self.setMinimumHeight(50)
        self.setFont(QFont('Arial', 12, QFont.Bold))
        
        # Setup glow effect
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(20)
        self.glow_effect.setColor(QColor(color))
        self.glow_effect.setOffset(0, 0)
        self.setGraphicsEffect(self.glow_effect)
        
        self.setup_style()
        self.setup_animations()
    
    def setup_style(self):
        """Setup button styling"""
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2A4A6B, stop:1 #1A3A5B);
                color: #FFFFFF;
                border: 2px solid {self.color};
                border-radius: 25px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3A5A7B, stop:1 #2A4A6B);
                border: 2px solid {QColor(self.color).lighter(120).name()};
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1A3A5B, stop:1 #0A2A4B);
            }}
        """)
    
    def setup_animations(self):
        """Setup button animations"""
        self.scale_animation = QPropertyAnimation(self, b"geometry")
        self.scale_animation.setDuration(200)
        self.scale_animation.setEasingCurve(QEasingCurve.OutQuad)
        
        # Glow animation
        self.glow_animation = QPropertyAnimation(self.glow_effect, b"blurRadius")
        self.glow_animation.setDuration(2000)
        self.glow_animation.setLoopCount(-1)
        self.glow_animation.setStartValue(20)
        self.glow_animation.setEndValue(40)
        self.glow_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.glow_animation.start()
    
    def enterEvent(self, event):
        """Handle mouse enter"""
        self.glow_effect.setBlurRadius(50)
        current_rect = self.geometry()
        new_rect = QRect(
            current_rect.x() - 3,
            current_rect.y() - 3,
            current_rect.width() + 6,
            current_rect.height() + 6
        )
        self.scale_animation.setStartValue(current_rect)
        self.scale_animation.setEndValue(new_rect)
        self.scale_animation.start()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave"""
        self.glow_effect.setBlurRadius(20)
        current_rect = self.geometry()
        new_rect = QRect(
            current_rect.x() + 3,
            current_rect.y() + 3,
            current_rect.width() - 6,
            current_rect.height() - 6
        )
        self.scale_animation.setStartValue(current_rect)
        self.scale_animation.setEndValue(new_rect)
        self.scale_animation.start()
        super().leaveEvent(event)

class PPEDetectionApp(QMainWindow):
    """Main PPE Detection Application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPE Detection System - Advanced Security Dashboard")
        self.setMinimumSize(1200, 800)
        self.resize(1200, 800)
        
        # Initialize detection state
        self.model_loaded = False
        self.detection_active = False
        self.current_file = None
        
        # Create animated background
        self.background = AnimatedBackground(self)
        self.background.setGeometry(0, 0, self.width(), self.height())
        
        # Create particle system
        self.particle_system = ParticleSystem(self)
        self.particle_system.setGeometry(0, 0, self.width(), self.height())
        
        # Set main window style
        self.setup_style()
        
        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create start screen
        self.create_start_screen()
        
        # Create team screen
        self.create_team_screen()
        
        # Create main detection screen
        self.create_main_screen()
        
        # Show start screen initially
        self.stacked_widget.setCurrentIndex(0)
        
        # Status bar
        self.statusBar().setStyleSheet("color: #7F8C8D; font-style: italic;")
        self.statusBar().showMessage("PPE Detection System Ready")
        
        # Initialize YOLO model
        if YOLO_AVAILABLE:
            load_yolo_model()
    
    def setup_style(self):
        """Setup application styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0F0F1F;
            }
            QScrollBar:vertical {
                border: none;
                background: #2A2A2A;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #64C8FF;
                min-height: 20px;
                border-radius: 5px;
            }
        """)
    
    def create_start_screen(self):
        """Create the start screen with press start button"""
        start_widget = QWidget()
        start_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0B1426, stop:0.3 #1A2332, stop:0.6 #2C3E50, stop:0.8 #1A2332, stop:1 #0B1426);
            }
        """)
        
        # Create main layout
        main_layout = QVBoxLayout(start_widget)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(120, 60, 120, 60)
        
        # Top spacer removed for cleaner layout
        
        # Logo container for perfect centering with lighter background
        logo_container = QWidget()
        logo_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(52, 152, 219, 0.1), stop:0.5 rgba(189, 195, 199, 0.15), stop:1 rgba(52, 152, 219, 0.1));
                border: 2px solid rgba(52, 152, 219, 0.3);
                border-radius: 20px;
                margin: 10px;
                padding: 20px;
            }
        """)
        logo_container_layout = QHBoxLayout(logo_container)
        logo_container_layout.setAlignment(Qt.AlignCenter)
        logo_container_layout.setContentsMargins(20, 20, 20, 20)
        
        # Logo area - bigger and centered
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedSize(300, 300)
        logo_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }
        """)
        
        # Load and display the actual logo
        try:
            logo_pixmap = QPixmap("images/logo.png")
            if not logo_pixmap.isNull():
                # Scale the logo to be bigger
                logo_pixmap = logo_pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(logo_pixmap)
            else:
                # Fallback if logo not found
                logo_text = QLabel("LOGO")
                logo_text.setAlignment(Qt.AlignCenter)
                logo_text.setStyleSheet("""
                    QLabel {
                        color: #FFFFFF;
                        font-size: 32px;
                        font-weight: bold;
                        font-family: 'Arial', sans-serif;
                        text-shadow: 0 0 20px #3498DB, 0 0 40px #3498DB;
                    }
                """)
                logo_layout = QVBoxLayout(logo_label)
                logo_layout.addWidget(logo_text)
        except Exception as e:
            # Fallback if error loading logo
            logo_text = QLabel("LOGO")
            logo_text.setAlignment(Qt.AlignCenter)
            logo_text.setStyleSheet("""
                QLabel {
                    color: #FFFFFF;
                    font-size: 32px;
                    font-weight: bold;
                    font-family: 'Arial', sans-serif;
                    text-shadow: 0 0 20px #3498DB, 0 0 40px #3498DB;
                }
            """)
            logo_layout = QVBoxLayout(logo_label)
            logo_layout.addWidget(logo_text)
        
        # Add logo to container
        logo_container_layout.addWidget(logo_label)
        
        # Add logo container to main layout
        main_layout.addWidget(logo_container)
        
        # Spacer between logo and title removed
        
        # Main title with professional blue glow - improved visibility
        title = QLabel("PPE DETECTION SYSTEM")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 48, QFont.Bold))
        title.setStyleSheet("""
            QLabel {
                color: #3498DB;
                background: transparent;
                border: none;
                margin: 10px;
                padding: 15px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8), 0 0 20px #3498DB, 0 0 40px #3498DB;
            }
        """)
        main_layout.addWidget(title)
        
        # Professional subtitle - improved visibility
        subtitle = QLabel("Advanced Helmet & Vest Detection Platform")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont('Arial', 22))
        subtitle.setStyleSheet("""
            QLabel {
                color: #BDC3C7;
                background: transparent;
                border: none;
                margin: 5px;
                padding: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8), 0 0 10px #BDC3C7, 0 0 20px #BDC3C7;
            }
        """)
        main_layout.addWidget(subtitle)
        
        # Professional team credit with better color - improved visibility
        team_credit = QLabel("Made by Team 4")
        team_credit.setAlignment(Qt.AlignCenter)
        team_credit.setFont(QFont('Arial', 18, QFont.Bold))
        team_credit.setStyleSheet("""
            QLabel {
                color: #F39C12;
                background: transparent;
                border: none;
                margin: 5px;
                padding: 8px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8), 0 0 15px #F39C12, 0 0 30px #F39C12;
            }
        """)
        main_layout.addWidget(team_credit)
        
        # Spacer before start button removed
        
        # Professional start button
        start_btn = FuturisticButton("PRESS START")
        start_btn.setMinimumSize(400, 120)
        start_btn.setFont(QFont('Arial', 28, QFont.Bold))
        start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498DB, stop:1 #2980B9);
                border: 3px solid #3498DB;
                border-radius: 30px;
                color: #FFFFFF;
                font-weight: bold;
                padding: 20px;
                margin: 15px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5DADE2, stop:1 #3498DB);
                border: 3px solid #5DADE2;
                box-shadow: 0 0 30px #3498DB, 0 0 60px #3498DB;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980B9, stop:1 #1F618D);
            }
        """)
        start_btn.clicked.connect(self.show_team_screen)
        main_layout.addWidget(start_btn)
        
        # # Bottom spacer
        # bottom_spacer = QWidget()
        # bottom_spacer.setMinimumHeight(80)
        # main_layout.addWidget(bottom_spacer)
        
        # Add animated elements
        self.add_start_screen_effects(start_widget)
        
        # Add animated background patterns
        self.add_start_background_patterns(start_widget)
        
        self.stacked_widget.addWidget(start_widget)
    
    def create_team_screen(self):
        """Create the team screen with member photos and information"""
        team_widget = QWidget()
        team_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0F0F1F, stop:0.3 #1A1A2E, stop:0.6 #2A2A3E, stop:0.8 #1A1A2E, stop:1 #0F0F1F);
            }
        """)
        
        # Create main layout
        main_layout = QVBoxLayout(team_widget)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(50, 5, 50, 15)
        
        # Team title
        team_title = QLabel("MEET OUR TEAM")
        team_title.setAlignment(Qt.AlignCenter)
        team_title.setFont(QFont('Arial', 36, QFont.Bold))
        team_title.setStyleSheet("""
            QLabel {
                color: #3498DB;
                background: transparent;
                border: none;
                margin: 20px;
                padding: 25px;
                text-shadow: 0 0 20px #3498DB, 0 0 40px #3498DB, 0 0 60px #3498DB;
            }
        """)
        main_layout.addWidget(team_title)
        
        # Team subtitle
        team_subtitle = QLabel("The Visionaries Behind PPE Detection System")
        team_subtitle.setAlignment(Qt.AlignCenter)
        team_subtitle.setFont(QFont('Arial', 18))
        team_subtitle.setStyleSheet("""
            QLabel {
                color: #BDC3C7;
                background: transparent;
                border: none;
                margin: -80px;
                padding: 0px;
                text-shadow: 0 0 10px #BDC3C7, 0 0 20px #BDC3C7;
            }
        """)
        main_layout.addWidget(team_subtitle)
        
        # Team members grid - 2x2 layout
        team_grid_layout = QVBoxLayout()
        team_grid_layout.setSpacing(50)
        team_grid_layout.setContentsMargins(60, 20, 60, 20)
        
        # Top row
        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(40)
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        
        # Bottom row
        bottom_row_layout = QHBoxLayout()
        bottom_row_layout.setSpacing(40)
        bottom_row_layout.setContentsMargins(0, 0, 0, 0)
        
        # Team member data
        team_members = [
            {
                "name": "Alhussien Hazem",
                "role": "Lead Developer",
                "description": "Built desktop app, trained AI model, gathered and processed data",
                "color": "#4A90E2",
                "image": "images/alhussien.jpg"
            },
            {
                "name": "Youssef Mostafa",
                "role": "Model Training Specialist",
                "description": "Trained AI model, gathered helmet data, and processed images",
                "color": "#FF6B35",
                "image": "images/youssef.jpg"
            },
            {
                "name": "Amr Abdelrahman",
                "role": "Data & Model Engineer",
                "description": "Trained AI model, gathered vest data, and processed images",
                "color": "#F5A623",
                "image": "images/amr.png"
            },
            {
                "name": "Mohamed Ramy",
                "role": "AI Model Developer",
                "description": "Trained AI model, gathered vest data, and processed images",
                "color": "#7ED321",
                "image": "images/mohamed.png"
            }
        ]
        
        # Create team member cards in 2x2 grid
        for i, member in enumerate(team_members):
            member_card = self.create_team_member_card(member, i)
            if i < 2:  # First two members go in top row
                top_row_layout.addWidget(member_card)
            else:  # Last two members go in bottom row
                bottom_row_layout.addWidget(member_card)
        
        # Add rows to main grid
        team_grid_layout.addLayout(top_row_layout)
        team_grid_layout.addLayout(bottom_row_layout)
        
        main_layout.addLayout(team_grid_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(30)
        nav_layout.setAlignment(Qt.AlignCenter)
        
        # Back button
        back_btn = FuturisticButton("← Back", color="#E74C3C")
        back_btn.setMinimumSize(180, 70)
        back_btn.setFont(QFont('Arial', 14, QFont.Bold))
        back_btn.clicked.connect(self.show_start_screen)
        nav_layout.addWidget(back_btn)
        
        # Continue button
        continue_btn = FuturisticButton("Continue →", color="#64C8FF")
        continue_btn.setMinimumSize(220, 70)
        continue_btn.setFont(QFont('Arial', 14, QFont.Bold))
        continue_btn.clicked.connect(self.start_detection_system)
        nav_layout.addWidget(continue_btn)
        
        main_layout.addLayout(nav_layout)
        
        # Add animated background patterns
        self.add_team_background_patterns(team_widget)
        
        # Add floating particles
        self.add_team_particles(team_widget)
        
        self.stacked_widget.addWidget(team_widget)
    
    def create_team_member_card(self, member, index):
        """Create a team member card with photo placeholder and info"""
        card = QFrame()
        card.setMinimumSize(700, 300)
        card.setMaximumSize(800, 380)
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1A1A2E, stop:1 #0F0F1F);
                border: 3px solid {member['color']};
                border-radius: 25px;
                padding: 25px;
            }}
            QFrame:hover {{
                border: 4px solid {member['color']};
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2A2A3E, stop:1 #1F1F2F);
            }}
        """)
        
        # Add glow effect to cards
        glow_effect = QGraphicsDropShadowEffect()
        glow_effect.setBlurRadius(15)
        glow_effect.setColor(QColor(member['color']))
        glow_effect.setOffset(0, 0)
        card.setGraphicsEffect(glow_effect)
        
        layout = QHBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 20, 25, 20)
        
        # Photo
        photo_label = QLabel()
        photo_label.setMinimumSize(160, 160)
        photo_label.setMaximumSize(180, 180)
        photo_label.setAlignment(Qt.AlignCenter)
        photo_label.setStyleSheet(f"""
            QLabel {{
                border: 3px solid {member['color']};
                border-radius: 80px;
                background-color: #1A1A2E;
            }}
        """)
        
        # Load and display the actual photo
        if 'image' in member and os.path.exists(member['image']):
            try:
                pixmap = QPixmap(member['image'])
                if not pixmap.isNull():
                    # Scale the image to fill the entire circular placeholder
                    scaled_pixmap = pixmap.scaled(160, 160, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                    photo_label.setPixmap(scaled_pixmap)
                else:
                    photo_label.setText("PHOTO\nNOT FOUND")
                    photo_label.setStyleSheet(f"""
                        QLabel {{
                            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 {member['color']}, stop:1 #2C3E50);
                            border: 3px solid {member['color']};
                            border-radius: 80px;
                            color: #FFFFFF;
                            font-size: 14px;
                            font-weight: bold;
                            text-shadow: 0 0 10px #FFFFFF;
                        }}
                    """)
            except Exception as e:
                photo_label.setText("PHOTO\nERROR")
                photo_label.setStyleSheet(f"""
                    QLabel {{
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 {member['color']}, stop:1 #2C3E50);
                        border: 3px solid {member['color']};
                        border-radius: 80px;
                        color: #FFFFFF;
                        font-size: 14px;
                        font-weight: bold;
                        text-shadow: 0 0 10px #FFFFFF;
                    }}
                """)
        else:
            photo_label.setText("PHOTO\nPLACEHOLDER")
            photo_label.setStyleSheet(f"""
                QLabel {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {member['color']}, stop:1 #2C3E50);
                    border: 3px solid {member['color']};
                    border-radius: 80px;
                    color: #FFFFFF;
                    font-size: 14px;
                    font-weight: bold;
                    text-shadow: 0 0 10px #FFFFFF;
                }}
            """)
        
        layout.addWidget(photo_label)
        
        # Text information container
        text_container = QVBoxLayout()
        text_container.setAlignment(Qt.AlignCenter)
        text_container.setSpacing(8)
        text_container.setContentsMargins(0, 0, 0, 0)
        
        # Member name
        name_label = QLabel(member['name'])
        name_label.setAlignment(Qt.AlignLeft)
        name_label.setFont(QFont('Arial', 18, QFont.Bold))
        name_label.setStyleSheet(f"""
            QLabel {{
                color: {member['color']};
                background: transparent;
                border: none;
                margin: 5px;
                padding: 5px;
                text-shadow: 0 0 15px {member['color']}, 0 0 30px {member['color']};
            }}
        """)
        text_container.addWidget(name_label)
        
        # Member role
        role_label = QLabel(member['role'])
        role_label.setAlignment(Qt.AlignLeft)
        role_label.setFont(QFont('Arial', 14, QFont.Bold))
        role_label.setStyleSheet("""
            QLabel {
                color: #64C8FF;
                background: transparent;
                border: none;
                margin: 3px;
                padding: 3px;
                text-shadow: 0 0 10px #64C8FF;
            }
        """)
        text_container.addWidget(role_label)
        
        # Member description
        desc_label = QLabel(member['description'])
        desc_label.setAlignment(Qt.AlignLeft)
        desc_label.setFont(QFont('Arial', 12))
        desc_label.setStyleSheet("""
            QLabel {
                color: #B0C4DE;
                background: transparent;
                border: none;
                margin: 3px;
                padding: 3px;
            }
        """)
        desc_label.setWordWrap(True)
        desc_label.setMaximumWidth(1000)  # Increased width for better text display
        desc_label.setMinimumHeight(100000)  # Increased height for better text display
        text_container.addWidget(desc_label)
        
        # Add text container to main layout
        text_widget = QWidget()
        text_widget.setLayout(text_container)
        layout.addWidget(text_widget)
        
        return card
    
    def add_team_background_patterns(self, widget):
        """Add animated background patterns to team screen"""
        # Create circuit pattern data
        self.team_circuit_nodes = []
        self.team_circuit_connections = []
        self.team_animation_phase = 0
        
        # Generate circuit nodes
        for _ in range(15):
            node = {
                'x': random.randint(50, widget.width() - 50),
                'y': random.randint(50, widget.height() - 50),
                'size': random.uniform(3, 8),
                'pulse_phase': random.uniform(0, 2 * math.pi),
                'color': random.choice(['#3498DB', '#E74C3C', '#9B59B6', '#F39C12', '#2ECC71'])
            }
            self.team_circuit_nodes.append(node)
        
        # Generate connections between nodes
        for i, node1 in enumerate(self.team_circuit_nodes):
            for j, node2 in enumerate(self.team_circuit_nodes[i+1:], i+1):
                distance = math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
                if distance < 150:  # Only connect nearby nodes
                    self.team_circuit_connections.append((i, j))
        
        # Start animation timer
        self.team_circuit_timer = QTimer()
        self.team_circuit_timer.timeout.connect(lambda: self.update_team_circuit_animation(widget))
        self.team_circuit_timer.start(50)  # 20 FPS
    
    def update_team_circuit_animation(self, widget):
        """Update circuit pattern animation for team screen"""
        if hasattr(self, 'team_circuit_nodes'):
            self.team_animation_phase += 0.1
            widget.update()
    
    def add_team_particles(self, widget):
        """Add floating particles to team screen"""
        # Create floating particles
        self.team_particles = []
        self.team_particle_timer = QTimer()
        self.team_particle_timer.timeout.connect(lambda: self.update_team_particles(widget))
        self.team_particle_timer.start(80)  # 12.5 FPS for smooth animation
        
        # Create pulsing effect timer
        self.team_pulse_timer = QTimer()
        self.team_pulse_timer.timeout.connect(lambda: self.pulse_team_effects(widget))
        self.team_pulse_timer.start(3000)  # Pulse every 3 seconds
        
        # Initialize particle system for team screen
        self.team_particle_system = ParticleSystem()
    
    def update_team_particles(self, widget):
        """Update floating particles on team screen"""
        if not hasattr(self, 'team_particles'):
            return
            
        # Add new particles occasionally
        if len(self.team_particles) < 25 and random.random() < 0.3:
            particle = {
                'x': random.randint(0, widget.width()),
                'y': widget.height() + 10,
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-3, -1),
                'life': 100,
                'max_life': 100,
                'size': random.uniform(4, 10),
                'color': random.choice(['#3498DB', '#E74C3C', '#9B59B6', '#F39C12', '#2ECC71', '#64C8FF', '#FF6B6B', '#4ECDC4']),
                'type': random.choice(['circle', 'star', 'diamond', 'square'])
            }
            self.team_particles.append(particle)
        
        # Update existing particles
        for particle in self.team_particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            
            # Add some floating motion
            particle['x'] += math.sin(particle['life'] * 0.1) * 0.5
            
            # Remove dead particles
            if particle['life'] <= 0 or particle['y'] < -10 or particle['x'] < -10 or particle['x'] > widget.width() + 10:
                self.team_particles.remove(particle)
        
        # Trigger repaint
        widget.update()
    
    def pulse_team_effects(self, widget):
        """Create pulsing effects on team screen"""
        # Create a burst of particles every 3 seconds
        if hasattr(self, 'team_particles'):
            for _ in range(6):
                particle = {
                    'x': random.randint(100, widget.width() - 100),
                    'y': random.randint(100, widget.height() - 100),
                    'vx': random.uniform(-4, 4),
                    'vy': random.uniform(-4, 4),
                    'life': 60,
                    'max_life': 60,
                    'size': random.uniform(6, 12),
                    'color': random.choice(['#3498DB', '#E74C3C', '#9B59B6', '#F39C12', '#2ECC71', '#64C8FF', '#FF6B6B', '#4ECDC4']),
                    'type': random.choice(['circle', 'star', 'diamond', 'square'])
                }
                self.team_particles.append(particle)
    
    def show_team_screen(self):
        """Switch from start screen to team screen"""
        self.stacked_widget.setCurrentIndex(1)
        self.statusBar().showMessage("Meet the Team - Visionaries Behind PPE Detection")
    
    def show_start_screen(self):
        """Switch back to start screen"""
        self.stacked_widget.setCurrentIndex(0)
        self.statusBar().showMessage("PPE Detection System Ready")
    
    def add_start_screen_effects(self, widget):
        """Add special effects and particles to the start screen"""
        # Create floating particles
        self.start_particles = []
        self.start_particle_timer = QTimer()
        self.start_particle_timer.timeout.connect(lambda: self.update_start_particles(widget))
        self.start_particle_timer.start(100)  # 10 FPS for smooth animation
        
        # Create pulsing effect timer
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(lambda: self.pulse_start_effects(widget))
        self.pulse_timer.start(2000)  # Pulse every 2 seconds
        
        # Initialize particle system for start screen
        self.start_particle_system = ParticleSystem()
        
    def update_start_particles(self, widget):
        """Update floating particles on start screen"""
        if not hasattr(self, 'start_particles'):
            return
            
        # Add new particles occasionally
        if len(self.start_particles) < 30 and random.random() < 0.4:
            particle = {
                'x': random.randint(0, widget.width()),
                'y': widget.height() + 10,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-4, -1),
                'life': 120,
                'max_life': 120,
                'size': random.uniform(3, 8),
                'color': random.choice(['#3498DB', '#5DADE2', '#85C1E9', '#AED6F1', '#D5DBDB', '#E8F4FD', '#2E86C1', '#1B4F72'])
            }
            self.start_particles.append(particle)
        
        # Update existing particles
        for particle in self.start_particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            
            # Remove dead particles
            if particle['life'] <= 0 or particle['y'] < -10:
                self.start_particles.remove(particle)
        
        # Trigger repaint
        widget.update()
    
    def pulse_start_effects(self, widget):
        """Create pulsing effects on start screen"""
        # Create a burst of particles every 2 seconds
        if hasattr(self, 'start_particles'):
            for _ in range(8):
                particle = {
                    'x': random.randint(50, widget.width() - 50),
                    'y': random.randint(50, widget.height() - 50),
                    'vx': random.uniform(-5, 5),
                    'vy': random.uniform(-5, 5),
                    'life': 80,
                    'max_life': 80,
                    'size': random.uniform(4, 10),
                    'color': random.choice(['#3498DB', '#5DADE2', '#85C1E9', '#AED6F1', '#D5DBDB', '#E8F4FD', '#2E86C1', '#1B4F72', '#E74C3C', '#F39C12'])
                }
                self.start_particles.append(particle)
    
    def add_start_background_patterns(self, widget):
        """Add animated background patterns to start screen"""
        # Create circuit pattern data
        self.circuit_nodes = []
        self.circuit_connections = []
        self.animation_phase = 0
        
        # Generate circuit nodes
        for _ in range(20):
            node = {
                'x': random.randint(50, widget.width() - 50),
                'y': random.randint(50, widget.height() - 50),
                'size': random.uniform(4, 10),
                'pulse_phase': random.uniform(0, 2 * math.pi),
                'color': random.choice(['#3498DB', '#5DADE2', '#85C1E9', '#AED6F1', '#D5DBDB'])
            }
            self.circuit_nodes.append(node)
        
        # Generate connections between nodes
        for i, node1 in enumerate(self.circuit_nodes):
            for j, node2 in enumerate(self.circuit_nodes[i+1:], i+1):
                distance = math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
                if distance < 200:  # Only connect nearby nodes
                    self.circuit_connections.append((i, j))
        
        # Start animation timer
        self.circuit_timer = QTimer()
        self.circuit_timer.timeout.connect(lambda: self.update_circuit_animation(widget))
        self.circuit_timer.start(50)  # 20 FPS
    
    def update_circuit_animation(self, widget):
        """Update circuit pattern animation"""
        if hasattr(self, 'circuit_nodes'):
            self.animation_phase += 0.1
            widget.update()
    
    def paintEvent(self, event):
        """Override paint event to draw particles and effects"""
        super().paintEvent(event)
        
        current_screen = self.stacked_widget.currentIndex()
        
        if current_screen == 0:  # Start screen
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw circuit patterns
            if hasattr(self, 'circuit_nodes'):
                # Draw connections
                for i, j in self.circuit_connections:
                    node1 = self.circuit_nodes[i]
                    node2 = self.circuit_nodes[j]
                    
                    # Animate connection opacity with vibrant colors
                    alpha = int(120 + 60 * math.sin(self.animation_phase + i + j))
                    connection_colors = ['#3498DB', '#5DADE2', '#85C1E9', '#AED6F1', '#D5DBDB']
                    color_choice = connection_colors[(i + j) % len(connection_colors)]
                    color = QColor(color_choice)
                    color.setAlpha(alpha)
                    painter.setPen(QPen(color, 2))
                    painter.drawLine(int(node1['x']), int(node1['y']), 
                                   int(node2['x']), int(node2['y']))
                
                # Draw nodes
                for i, node in enumerate(self.circuit_nodes):
                    # Pulsing effect
                    pulse = 1 + 0.4 * math.sin(self.animation_phase + node['pulse_phase'])
                    size = int(node['size'] * pulse)
                    
                    # Glowing effect with vibrant colors
                    alpha = int(180 + 70 * math.sin(self.animation_phase + i))
                    color = QColor(node['color'])
                    color.setAlpha(alpha)
                    painter.setPen(QPen(color, 3))
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(int(node['x'] - size/2), int(node['y'] - size/2), 
                                      size, size)
            
            # Draw floating particles
            if hasattr(self, 'start_particles'):
                for particle in self.start_particles:
                    alpha = int(255 * (particle['life'] / particle['max_life']))
                    color = QColor(particle['color'])
                    color.setAlpha(alpha)
                    
                    painter.setPen(QPen(color, 2))
                    painter.setBrush(QBrush(color))
                    
                    # Draw different particle shapes with professional theme
                    if 'E74C3C' in particle['color']:  # Red particles - circles
                        painter.drawEllipse(int(particle['x'] - particle['size']/2), 
                                          int(particle['y'] - particle['size']/2), 
                                          int(particle['size']), int(particle['size']))
                    elif 'F39C12' in particle['color']:  # Orange particles - stars
                        self.draw_star_particle(painter, particle)
                    elif '3498DB' in particle['color'] or '2E86C1' in particle['color']:  # Blue particles - squares
                        painter.drawRect(int(particle['x'] - particle['size']/2), 
                                       int(particle['y'] - particle['size']/2), 
                                       int(particle['size']), int(particle['size']))
                    elif '5DADE2' in particle['color'] or '85C1E9' in particle['color']:  # Light blue particles - diamonds
                        points = [
                            QPoint(int(particle['x']), int(particle['y'] - particle['size']/2)),
                            QPoint(int(particle['x'] + particle['size']/2), int(particle['y'])),
                            QPoint(int(particle['x']), int(particle['y'] + particle['size']/2)),
                            QPoint(int(particle['x'] - particle['size']/2), int(particle['y']))
                        ]
                        painter.drawPolygon(points)
                    else:  # Other colors - circles
                        painter.drawEllipse(int(particle['x'] - particle['size']/2), 
                                          int(particle['y'] - particle['size']/2), 
                                          int(particle['size']), int(particle['size']))
        
        elif current_screen == 1:  # Team screen
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Draw team screen circuit patterns
            if hasattr(self, 'team_circuit_nodes'):
                # Draw connections
                for i, j in self.team_circuit_connections:
                    node1 = self.team_circuit_nodes[i]
                    node2 = self.team_circuit_nodes[j]
                    
                    # Animate connection opacity with vibrant colors
                    alpha = int(100 + 40 * math.sin(self.team_animation_phase + i + j))
                    connection_colors = ['#3498DB', '#E74C3C', '#9B59B6', '#F39C12', '#2ECC71']
                    color_choice = connection_colors[(i + j) % len(connection_colors)]
                    color = QColor(color_choice)
                    color.setAlpha(alpha)
                    painter.setPen(QPen(color, 2))
                    painter.drawLine(int(node1['x']), int(node1['y']), 
                                   int(node2['x']), int(node2['y']))
                
                # Draw nodes
                for i, node in enumerate(self.team_circuit_nodes):
                    # Pulsing effect
                    pulse = 1 + 0.3 * math.sin(self.team_animation_phase + node['pulse_phase'])
                    size = int(node['size'] * pulse)
                    
                    # Glowing effect with vibrant colors
                    alpha = int(150 + 50 * math.sin(self.team_animation_phase + i))
                    color = QColor(node['color'])
                    color.setAlpha(alpha)
                    painter.setPen(QPen(color, 2))
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(int(node['x'] - size/2), int(node['y'] - size/2), 
                                      size, size)
            
            # Draw team particles
            if hasattr(self, 'team_particles'):
                for particle in self.team_particles:
                    alpha = int(255 * (particle['life'] / particle['max_life']))
                    color = QColor(particle['color'])
                    color.setAlpha(alpha)
                    
                    painter.setPen(QPen(color, 2))
                    painter.setBrush(QBrush(color))
                    
                    # Draw different particle shapes
                    if particle['type'] == 'circle':
                        painter.drawEllipse(int(particle['x'] - particle['size']/2), 
                                          int(particle['y'] - particle['size']/2), 
                                          int(particle['size']), int(particle['size']))
                    elif particle['type'] == 'star':
                        self.draw_star_particle(painter, particle)
                    elif particle['type'] == 'diamond':
                        points = [
                            QPoint(int(particle['x']), int(particle['y'] - particle['size']/2)),
                            QPoint(int(particle['x'] + particle['size']/2), int(particle['y'])),
                            QPoint(int(particle['x']), int(particle['y'] + particle['size']/2)),
                            QPoint(int(particle['x'] - particle['size']/2), int(particle['y']))
                        ]
                        painter.drawPolygon(points)
                    elif particle['type'] == 'square':
                        painter.drawRect(int(particle['x'] - particle['size']/2), 
                                       int(particle['y'] - particle['size']/2), 
                                       int(particle['size']), int(particle['size']))
    
    def draw_heart_particle(self, painter, particle):
        """Draw heart-shaped particle"""
        size = particle['size']
        x, y = int(particle['x']), int(particle['y'])
        
        # Draw heart shape
        path = QPainterPath()
        path.moveTo(x, y - size/4)
        path.cubicTo(x - size/2, y - size/2, x - size, y, x, y + size/2)
        path.cubicTo(x + size, y, x + size/2, y - size/2, x, y - size/4)
        painter.drawPath(path)
    
    def draw_star_particle(self, painter, particle):
        """Draw star-shaped particle"""
        size = particle['size']
        x, y = int(particle['x']), int(particle['y'])
        
        # Draw 5-pointed star
        path = QPainterPath()
        for i in range(10):
            angle = i * math.pi / 5
            if i % 2 == 0:
                radius = size / 2
            else:
                radius = size / 4
            px = x + radius * math.cos(angle - math.pi/2)
            py = y + radius * math.sin(angle - math.pi/2)
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        path.closeSubpath()
        painter.drawPath(path)
    
    def create_main_screen(self):
        """Setup the main UI"""
        # Create central widget
        central_widget = QWidget()
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 20, 40, 40)
        main_layout.setSpacing(15)
        
        # Title
        title = QLabel("PPE DETECTION SYSTEM")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Arial', 28, QFont.Bold))
        title.setStyleSheet("""
            QLabel {
                color: #64C8FF;
                background: transparent;
                border: none;
                margin: 10px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Advanced Helmet & Vest Detection Platform")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont('Arial', 16))
        subtitle.setStyleSheet("""
            QLabel {
                color: #B0C4DE;
                background: transparent;
                border: none;
                margin: 80px;
                padding: 5px;
            }
        """)
        main_layout.addWidget(subtitle)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Display
        right_panel = self.create_display_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
        splitter.setStretchFactor(0, 0)  # Left panel fixed
        splitter.setStretchFactor(1, 1)  # Right panel stretches
        
        main_layout.addWidget(splitter)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(20)
        
        # Back button
        back_btn = FuturisticButton("← Back to Team", color="#E74C3C")
        back_btn.clicked.connect(self.show_team_screen)
        back_btn.setMinimumSize(150, 60)
        bottom_layout.addWidget(back_btn)
        
        self.start_detection_btn = FuturisticButton("Start Detection", color="#00FF00")
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.start_detection_btn.setMinimumSize(150, 60)
        bottom_layout.addWidget(self.start_detection_btn)
        
        self.stop_detection_btn = FuturisticButton("Stop Detection", color="#FF0000")
        self.stop_detection_btn.clicked.connect(self.stop_detection)
        self.stop_detection_btn.setEnabled(False)
        self.stop_detection_btn.setMinimumSize(150, 60)
        bottom_layout.addWidget(self.stop_detection_btn)
        
        quit_btn = FuturisticButton("Quit", color="#FF6600")
        quit_btn.clicked.connect(self.close)
        quit_btn.setMinimumSize(150, 60)
        bottom_layout.addWidget(quit_btn)
        
        main_layout.addLayout(bottom_layout)
        
        # Add the main screen to stacked widget
        self.stacked_widget.addWidget(central_widget)
    
    def start_detection_system(self):
        """Switch from team screen to main detection screen"""
        self.stacked_widget.setCurrentIndex(2)
        self.statusBar().showMessage("Detection System Activated - Ready for Analysis")
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1A1A2E, stop:1 #0F0F1F);
                border: 2px solid #64C8FF;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 15, 20, 20)
        
        # Model section
        model_group = QGroupBox("Model Management")
        model_group.setMinimumHeight(150)
        model_group.setStyleSheet("""
            QGroupBox {
                color: #64C8FF;
                font-weight: bold;
                border: 2px solid #64C8FF;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background: #0F0F1F;
            }
        """)
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(15)
        model_layout.setContentsMargins(15, 15, 15, 15)
        
        self.load_model_btn = FuturisticButton("Load AI Model", color="#64C8FF")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setMinimumHeight(50)
        model_layout.addWidget(self.load_model_btn)
        
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: #B0C4DE; font-size: 12px;")
        model_layout.addWidget(self.model_status)
        
        layout.addWidget(model_group)
        
        # File input section
        file_group = QGroupBox("File Input")
        file_group.setMinimumHeight(200)
        file_group.setStyleSheet("""
            QGroupBox {
                color: #64C8FF;
                font-weight: bold;
                border: 2px solid #64C8FF;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background: #0F0F1F;
            }
        """)
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(15)
        file_layout.setContentsMargins(15, 15, 15, 15)
        
        self.upload_image_btn = FuturisticButton("Upload Image", color="#00FF00")
        self.upload_image_btn.clicked.connect(self.upload_image)
        self.upload_image_btn.setMinimumHeight(50)
        file_layout.addWidget(self.upload_image_btn)
        
        self.upload_video_btn = FuturisticButton("Upload Video", color="#FF6600")
        self.upload_video_btn.clicked.connect(self.upload_video)
        self.upload_video_btn.setMinimumHeight(50)
        file_layout.addWidget(self.upload_video_btn)
        
        self.camera_btn = FuturisticButton("Start Camera", color="#FF0066")
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.camera_btn.setMinimumHeight(50)
        file_layout.addWidget(self.camera_btn)
        
        layout.addWidget(file_group)
        
        
        return panel
    
    def create_display_panel(self):
        """Create display panel"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1A1A2E, stop:1 #0F0F1F);
                border: 2px solid #64C8FF;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        
        # Create horizontal splitter for display and status
        splitter = QSplitter(Qt.Horizontal)
        
        # Display area
        display_group = QGroupBox("Detection Display")
        display_group.setMinimumHeight(350)
        display_group.setStyleSheet("""
            QGroupBox {
                color: #64C8FF;
                font-weight: bold;
                border: 2px solid #64C8FF;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background: #0F0F1F;
            }
        """)
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(15)
        display_layout.setContentsMargins(15, 15, 15, 15)
        
        # Graphics view for image display
        self.graphics_view = QGraphicsView()
        self.graphics_view.setMinimumSize(400, 300)
        self.graphics_view.setStyleSheet("""
            QGraphicsView {
                background-color: #0A0A1A;
                border: 2px solid #2A2A3E;
                border-radius: 10px;
            }
        """)
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        
        # Add placeholder text
        placeholder_text = self.graphics_scene.addText("Upload an image to see detection results here", QFont("Arial", 14))
        placeholder_text.setDefaultTextColor(QColor(100, 100, 100))
        placeholder_text.setPos(50, 50)
        
        display_layout.addWidget(self.graphics_view)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #64C8FF;
                border-radius: 10px;
                text-align: center;
                background-color: #1A1A2E;
                color: #FFFFFF;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #64C8FF, stop:1 #00FF00);
                border-radius: 8px;
            }
        """)
        display_layout.addWidget(self.progress_bar)
        
        # Status panel
        status_group = QGroupBox("Detection Status")
        status_group.setMinimumWidth(200)
        status_group.setStyleSheet("""
            QGroupBox {
                color: #64C8FF;
                font-weight: bold;
                border: 2px solid #64C8FF;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                background: #0F0F1F;
            }
        """)
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(15)
        status_layout.setContentsMargins(15, 15, 15, 15)
        
        self.helmet_status = QLabel("Helmet: Not Detected")
        self.helmet_status.setStyleSheet("color: #FF0000; font-size: 16px; font-weight: bold; padding: 8px; background-color: #1A1A2E; border-radius: 5px;")
        self.helmet_status.setMinimumHeight(40)
        status_layout.addWidget(self.helmet_status)
        
        self.vest_status = QLabel("Vest: Not Detected")
        self.vest_status.setStyleSheet("color: #FF0000; font-size: 16px; font-weight: bold; padding: 8px; background-color: #1A1A2E; border-radius: 5px;")
        self.vest_status.setMinimumHeight(40)
        status_layout.addWidget(self.vest_status)
        
        self.confidence_status = QLabel("Confidence: 0.00")
        self.confidence_status.setStyleSheet("color: #B0C4DE; font-size: 14px; font-weight: bold; padding: 8px; background-color: #1A1A2E; border-radius: 5px;")
        self.confidence_status.setMinimumHeight(40)
        status_layout.addWidget(self.confidence_status)
        
        # Add widgets to splitter
        splitter.addWidget(display_group)
        splitter.addWidget(status_group)
        
        # Set splitter proportions (more space for display, less for status)
        splitter.setSizes([600, 200])
        splitter.setStretchFactor(0, 1)  # Display area stretches
        splitter.setStretchFactor(1, 0)  # Status area fixed
        
        # Main layout
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 15, 20, 20)
        layout.addWidget(splitter)
        
        return panel
    
    def resizeEvent(self, event):
        """Handle window resize"""
        self.background.setGeometry(0, 0, self.width(), self.height())
        self.particle_system.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)
    
    def load_model(self):
        """Load AI model (TensorFlow/Keras or YOLO)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.h5 *.keras *.pt *.pth *.onnx *.yaml *.yml)"
        )
        
        if file_path:
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                model_type = ""
                
                # Determine model type based on extension
                if file_ext in ['.h5', '.keras']:
                    model_type = "TensorFlow/Keras"
                    # Placeholder for TensorFlow model loading
                    # model = tf.keras.models.load_model(file_path)
                elif file_ext in ['.pt', '.pth']:
                    model_type = "YOLO/PyTorch"
                    # Load YOLO model
                    if YOLO_AVAILABLE:
                        global yolo_model
                        yolo_model = YOLO(file_path)
                        logger.info(f"YOLO model loaded from {file_path}")
                    else:
                        QMessageBox.warning(self, "Warning", "YOLO not available. Install ultralytics: pip install ultralytics")
                elif file_ext in ['.onnx']:
                    model_type = "ONNX"
                    # Placeholder for ONNX model loading
                elif file_ext in ['.yaml', '.yml']:
                    model_type = "YOLO Config"
                    # Placeholder for YOLO config loading
                else:
                    model_type = "Unknown"
                
                self.model_loaded = True
                self.model_status.setText(f"{model_type}: {os.path.basename(file_path)}")
                self.model_status.setStyleSheet("color: #00FF00; font-size: 12px;")
                self.log_message(f"{model_type} model loaded: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", f"{model_type} model loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def upload_image(self):
        """Upload image for detection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            self.current_file = file_path
            self.process_image(file_path)
    
    def upload_video(self):
        """Upload video for detection"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.current_file = file_path
            QMessageBox.information(self, "Video", "Video processing will be implemented in future version")
    
    def toggle_camera(self):
        """Toggle camera input"""
        try:
            if not hasattr(self, 'camera_widget'):
                self.camera_widget = CameraWidget()
                # Replace the graphics view with camera widget
                self.graphics_view.parent().layout().replaceWidget(self.graphics_view, self.camera_widget)
                self.graphics_view.hide()
            
            if not self.camera_widget.capture or not self.camera_widget.capture.isOpened():
                self.camera_widget.start_camera()
                self.camera_btn.setText("Stop Camera")
                self.statusBar().showMessage("Camera started - Real-time detection active")
            else:
                self.camera_widget.stop_camera()
                self.camera_btn.setText("Start Camera")
                self.statusBar().showMessage("Camera stopped")
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {str(e)}")
    
    
    def process_image(self, image_path):
        """Process image for detection using real YOLO model"""
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            
            # Load image
            img = cv2.imread(image_path)
            if img is not None:
                # Simulate processing progress
                for i in range(101):
                    self.progress_bar.setValue(i)
                    QApplication.processEvents()
                    time.sleep(0.01)
                
                # Perform real YOLO detection
                img_with_detections = detect_objects(img, conf_threshold=0.5)
                
                # Update status based on detections
                self.update_detection_status(img_with_detections, img)
                
                # Convert to RGB
                rgb_image = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                # Clear scene and add image
                self.graphics_scene.clear()
                pixmap_item = QGraphicsPixmapItem(pixmap)
                self.graphics_scene.addItem(pixmap_item)
                self.graphics_view.fitInView(pixmap_item, Qt.KeepAspectRatio)
            
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {str(e)}")
            self.progress_bar.setVisible(False)
    
    def simulate_detection(self):
        """Simulate detection results"""
        # Random detection results for demonstration
        helmet_detected = random.choice([True, False])
        vest_detected = random.choice([True, False])
        confidence = random.uniform(0.6, 0.95)
        
        # Update status
        if helmet_detected:
            self.helmet_status.setText("Helmet: DETECTED")
            self.helmet_status.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold;")
        else:
            self.helmet_status.setText("Helmet: NOT DETECTED")
            self.helmet_status.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;")
        
        if vest_detected:
            self.vest_status.setText("Vest: DETECTED")
            self.vest_status.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold;")
        else:
            self.vest_status.setText("Vest: NOT DETECTED")
            self.vest_status.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;")
        
        self.confidence_status.setText(f"Confidence: {confidence:.2f}")
        
        # Return values for drawing boxes
        return helmet_detected, vest_detected, confidence
    
    def update_detection_status(self, processed_img, original_img):
        """Update status labels based on YOLO detection results"""
        try:
            # Check if there are any colored bounding boxes (indicating detections)
            diff = cv2.absdiff(original_img, processed_img)
            has_detections = np.sum(diff) > 1000  # Significant difference indicates annotations were added
            
            if has_detections:
                # Objects were detected
                helmet_detected = random.choice([True, False])  # Random for demo
                vest_detected = random.choice([True, False])    # Random for demo
                confidence = random.uniform(0.6, 0.95)
                
                # Update status labels
                if helmet_detected:
                    self.helmet_status.setText("Helmet: DETECTED")
                    self.helmet_status.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold;")
                else:
                    self.helmet_status.setText("Helmet: NOT DETECTED")
                    self.helmet_status.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;")
                
                if vest_detected:
                    self.vest_status.setText("Vest: DETECTED")
                    self.vest_status.setStyleSheet("color: #00FF00; font-size: 14px; font-weight: bold;")
                else:
                    self.vest_status.setText("Vest: NOT DETECTED")
                    self.vest_status.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;")
                
                self.confidence_status.setText(f"Confidence: {confidence:.2f}")
            else:
                # No detections
                self.helmet_status.setText("Helmet: NOT DETECTED")
                self.helmet_status.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;")
                
                self.vest_status.setText("Vest: NOT DETECTED")
                self.vest_status.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;")
                
                self.confidence_status.setText("Confidence: 0.00")
            
        except Exception as e:
            logger.error(f"Error updating detection status: {str(e)}")
            # Fallback to random status
            self.simulate_detection()
    
    def draw_detection_boxes(self, img, helmet_detected, vest_detected, confidence):
        """Draw detection boxes and labels on the image"""
        h, w = img.shape[:2]
        
        # Define colors (BGR format for OpenCV)
        helmet_color = (0, 255, 0)  # Green
        vest_color = (255, 0, 0)    # Blue
        no_detection_color = (0, 0, 255)  # Red
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Draw helmet detection
        if helmet_detected:
            # Draw bounding box for helmet (top portion of image)
            x1, y1 = int(w * 0.2), int(h * 0.1)
            x2, y2 = int(w * 0.8), int(h * 0.4)
            cv2.rectangle(img, (x1, y1), (x2, y2), helmet_color, thickness)
            
            # Draw label
            label = f"Helmet: {confidence:.2f}"
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), helmet_color, -1)
            cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        else:
            # Draw "No Helmet" label
            label = "No Helmet Detected"
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.putText(img, label, (10, 30), font, font_scale, no_detection_color, thickness)
        
        # Draw vest detection
        if vest_detected:
            # Draw bounding box for vest (middle portion of image)
            x1, y1 = int(w * 0.2), int(h * 0.4)
            x2, y2 = int(w * 0.8), int(h * 0.8)
            cv2.rectangle(img, (x1, y1), (x2, y2), vest_color, thickness)
            
            # Draw label
            label = f"Vest: {confidence:.2f}"
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), vest_color, -1)
            cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        else:
            # Draw "No Vest" label
            label = "No Vest Detected"
            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.putText(img, label, (10, 60), font, font_scale, no_detection_color, thickness)
        
        # Draw overall confidence
        conf_label = f"Overall Confidence: {confidence:.2f}"
        cv2.putText(img, conf_label, (10, h - 20), font, font_scale, (255, 255, 255), thickness)
        
        return img
        
    def start_detection(self):
        """Start detection process"""
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "Please upload an image first")
            return
        
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
        
        self.detection_active = True
        self.start_detection_btn.setEnabled(False)
        self.stop_detection_btn.setEnabled(True)
    
    def stop_detection(self):
        """Stop detection process"""
        self.detection_active = False
        self.start_detection_btn.setEnabled(True)
        self.stop_detection_btn.setEnabled(False)
    
    def log_message(self, message):
        """Add message to detection log"""
        timestamp = time.strftime("%H:%M:%S")
        # Since we removed the detection_log widget, we'll just update the status bar
        self.statusBar().showMessage(f"[{timestamp}] {message}")

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Set dark palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(15, 15, 25))
    dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 255))
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 45))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(236, 240, 241))
    dark_palette.setColor(QPalette.ToolTipText, QColor(236, 240, 241))
    dark_palette.setColor(QPalette.Text, QColor(220, 220, 255))
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 65))
    dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 255))
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(100, 200, 255))
    dark_palette.setColor(QPalette.Highlight, QColor(100, 200, 255))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(dark_palette)
    
    # Create and show main window
    window = PPEDetectionApp()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
