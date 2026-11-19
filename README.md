# 🛡️ PPE Detection System

<div align="center">

![PPE Detection Logo](images/logo.png)

**Advanced Helmet & Vest Detection Platform**

*Real-time Computer Vision Application with AI-Powered Safety Monitoring*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-orange.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🚀 Overview

The **PPE Detection System** is a cutting-edge desktop application that leverages advanced computer vision and deep learning to automatically detect Personal Protective Equipment (PPE) in real-time. Built with YOLOv8 and featuring a futuristic PyQt5 interface, this application ensures workplace safety compliance through intelligent monitoring.

### ✨ Key Highlights

- 🎯 **Real-time Detection**: Live camera feed processing with instant results
- 🤖 **AI-Powered**: YOLOv8 deep learning model for accurate detection
- 🎨 **Futuristic UI**: Modern interface with animated particles and effects
- 📱 **Multiple Input Sources**: Camera, images, and video support
- 📊 **Live Analytics**: Real-time status monitoring and confidence scoring
- 🔧 **Easy Setup**: Simple installation and configuration process

---

## 🎮 Features

### 🖥️ **User Interface**
- **Welcome Screen**: Professional startup with animated background effects
- **Team Showcase**: Meet the development team with interactive cards
- **Main Dashboard**: Comprehensive detection controls and live display
- **Status Panel**: Real-time helmet and vest detection monitoring
- **Animated Effects**: Particle systems and smooth transitions

### 🤖 **AI Detection Engine**
- **YOLOv8 Integration**: State-of-the-art object detection
- **Multi-Model Support**: YOLOv8n and YOLOv8m variants tested
- **High Accuracy**: Optimized for helmet and vest detection
- **Confidence Scoring**: Adjustable detection sensitivity
- **Real-time Processing**: 30 FPS target performance

### 📸 **Input Sources**
- **Live Camera**: Real-time webcam detection
- **Image Upload**: Single image analysis and processing
- **Video Support**: Batch video processing (planned)
- **Multiple Formats**: Support for various image and video formats

### 📊 **Analytics & Monitoring**
- **Live Status**: Real-time helmet and vest detection status
- **Confidence Metrics**: Detection reliability scoring
- **Visual Feedback**: Bounding boxes and labels
- **Performance Stats**: Processing speed and accuracy metrics

---

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, Linux, or macOS
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: CUDA-compatible (optional, for faster processing)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ppe-detection-system.git
   cd ppe-detection-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python ppe_detection_app.py
   ```

### Alternative Installation Methods

#### Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv ppe_env

# Activate virtual environment
# Windows:
ppe_env\Scripts\activate
# Linux/macOS:
source ppe_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python ppe_detection_app.py
```

#### Using Batch Files (Windows)
```bash
# Run the application
run.bat

# Or run with virtual environment
run_venv.bat
```

---

## 🎯 Usage Guide

### 1. **Getting Started**
1. Launch the application
2. Navigate through the welcome screen
3. Meet the team on the team screen
4. Proceed to the main detection interface

### 2. **Model Setup**
1. Click **"Load AI Model"** button
2. Select your trained YOLO model (`.pt` format)
3. Wait for model loading confirmation
4. Model is ready for detection

### 3. **Detection Process**
1. **For Live Camera**:
   - Click **"Start Camera"**
   - Position yourself in front of the camera
   - View real-time detection results

2. **For Image Upload**:
   - Click **"Upload Image"**
   - Select an image file
   - View detection results with bounding boxes

3. **For Video Processing**:
   - Click **"Upload Video"** (coming soon)
   - Select a video file
   - Process video frame by frame

### 4. **Understanding Results**
- **Green Boxes**: Detected helmets
- **Blue Boxes**: Detected vests
- **Confidence Score**: Detection reliability (0-1)
- **Status Panel**: Real-time compliance status

---

## 🏗️ Project Structure

```
PPE_Detection_App/
├── 📁 images/                    # Application images and logos
│   ├── logo.png                  # Main application logo
│   ├── alhussien.jpg            # Team member photos
│   ├── amr.png
│   ├── mohamed.png
│   └── youssef.jpg
├── 🐍 ppe_detection_app.py       # Main application file
└── 📄 README.md                  # This file
```

---

## 🧠 AI Model Information

### **Model Architecture**
- **Base Model**: YOLOv8 (Ultralytics)
- **Variants Tested**: YOLOv8n (nano) and YOLOv8m (medium)
- **Selected Model**: YOLOv8m for optimal accuracy-performance balance
- **Classes**: Helmet, Vest, No-PPE
- **Input Resolution**: 640x640 pixels

### **Performance Metrics**
- **mAP50**: [Your best mAP50 score]
- **mAP50-95**: [Your best mAP50-95 score]
- **Inference Speed**: ~30 FPS (with GPU)
- **Model Size**: ~50MB (YOLOv8m)

### **Training Process**
1. **Data Collection**: Helmet and vest image datasets
2. **Data Preprocessing**: Image augmentation and labeling
3. **Model Training**: Transfer learning with YOLOv8
4. **Validation**: Cross-validation and testing
5. **Optimization**: Hyperparameter tuning

---

## 👥 Team

<div align="center">

| **Alhussien Hazem** | **Youssef Mostafa** | **Amr Abdelrahman** | **Mohamed Ramy** |
|:---:|:---:|:---:|:---:|
| **Lead Developer** | **Model Training Specialist** | **Data & Model Engineer** | **AI Model Developer** |
| Desktop App, AI Model Training, Data Processing | AI Model Training, Helmet Data Collection | AI Model Training, Vest Data Processing | AI Model Training, Vest Data Collection |

</div>

---

## 🔧 Configuration

### **Model Configuration**
```python
# Model settings in ppe_detection_app.py
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = (640, 640)
```

### **Camera Settings**
```python
# Camera configuration
CAMERA_INDEX = 0  # Default camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
```

### **UI Customization**
- Modify colors in `utils/styles.py`
- Adjust animations in `utils/animations.py`
- Update team information in `ppe_detection_app.py`

---

## 🚨 Troubleshooting

### **Common Issues**

#### **Application Won't Start**
```bash
# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt

# Run test script
python test_app.py
```

#### **Camera Not Detected**
- Ensure camera is connected and not used by other applications
- Check camera permissions
- Try different camera index in code

#### **Model Loading Fails**
- Verify model file format (`.pt` for YOLO)
- Check file path and permissions
- Ensure model is compatible with YOLOv8

#### **Performance Issues**
- Reduce image/video resolution
- Lower confidence threshold
- Close other applications
- Use GPU acceleration if available

### **Getting Help**
1. Check console output for error messages
2. Run `python test_app.py` to verify installation
3. Review the troubleshooting section
4. Check GitHub issues for similar problems

---

## 🚀 Future Enhancements

### **Planned Features**
- [ ] **Video Processing**: Full video analysis capability
- [ ] **Multi-class Detection**: Additional PPE items (gloves, safety glasses)
- [ ] **Cloud Integration**: Remote monitoring and analytics
- [ ] **Mobile App**: Cross-platform mobile support
- [ ] **API Development**: REST API for third-party integration
- [ ] **Advanced Analytics**: Detailed reporting and insights
- [ ] **Real-time Alerts**: SMS/Email notifications for violations

### **Technical Improvements**
- [ ] **Model Optimization**: Quantization and pruning
- [ ] **Performance Tuning**: Multi-threading and GPU optimization
- [ ] **UI Enhancements**: Dark/light theme toggle
- [ ] **Database Integration**: SQLite/PostgreSQL support
- [ ] **Configuration Management**: YAML/JSON config files

---

## 📊 Performance Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| **Detection Speed** | 30 FPS | With GPU acceleration |
| **Accuracy (mAP50)** | [Your Score] | On validation dataset |
| **Memory Usage** | ~2GB | During real-time detection |
| **CPU Utilization** | 60-80% | On modern processors |
| **Model Size** | 50MB | YOLOv8m variant |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/ppe-detection-system/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ppe-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ppe-detection-system/discussions)
- **Email**: support@ppedetection.com

---

## 🙏 Acknowledgments

- **Ultralytics** for the amazing YOLOv8 framework
- **PyQt5** for the powerful GUI framework
- **OpenCV** for computer vision capabilities
- **Team 4** for the incredible development effort

---

<div align="center">

**Made with ❤️ by Team 4**

*Ensuring workplace safety through intelligent technology*

[![GitHub stars](https://img.shields.io/github/stars/yourusername/ppe-detection-system?style=social)](https://github.com/yourusername/ppe-detection-system)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/ppe-detection-system?style=social)](https://github.com/yourusername/ppe-detection-system)
[![GitHub watchers](https://img.shields.io/github/watchers/yourusername/ppe-detection-system?style=social)](https://github.com/yourusername/ppe-detection-system)

</div>
