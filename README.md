# AI-Based Smart Traffic Violation Detector

An intelligent system that automatically detects traffic violations using Computer Vision and Deep Learning. The system identifies:

- 🏍️ **Helmet Violations**: Motorcyclists riding without helmets
- 🚗 **License Plate Violations**: Vehicles without visible number plates

## Features

- **Real-time Detection**: Process live camera feeds
- **Video Analysis**: Upload and analyze recorded footage
- **Django Web Interface**: User-friendly dashboard for monitoring
- **Violation Management**: Review, confirm, or dismiss detections
- **Analytics Dashboard**: Visualize violation trends and statistics
- **Multi-camera Support**: Monitor multiple traffic points
- **Evidence Storage**: Automatically save violation images with annotations

## Technology Stack

- **Backend**: Django 4.2
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **OCR**: EasyOCR for license plate reading
- **Deep Learning**: PyTorch
- **Frontend**: Bootstrap 5, Chart.js
- **Database**: SQLite (configurable to PostgreSQL/MySQL)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/traffic-violation-detector.git
cd traffic-violation-detector
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with PyTorch, install it separately:

```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Download AI Models

The YOLOv8 model will automatically download on first run. For better accuracy, you can train custom models:

```bash
# Create models directory
mkdir models

# YOLOv8 will auto-download to this directory on first use
```

### Step 5: Setup Django

```bash
# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser for admin access
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput
```

### Step 6: Add Sample Data (Optional)

```bash
python manage.py shell
```

```python
from detector.models import Camera

# Create sample cameras
Camera.objects.create(
    name="Main Street Camera",
    location="Main St & 5th Ave",
    is_active=True
)

Camera.objects.create(
    name="Highway Entrance",
    location="Highway 101 Entrance",
    is_active=True
)

exit()
```

## Running the Application

### Start Django Development Server

```bash
python manage.py runserver
```

Access the application at: `http://localhost:8000`

### Admin Panel

Access admin panel at: `http://localhost:8000/admin`

## Usage

### 1. Upload Video for Analysis

1. Navigate to **Upload** page
2. Select a camera location
3. Upload video file (MP4, AVI, MOV)
4. System processes video and detects violations automatically

### 2. Live Camera Monitoring

1. Go to **Dashboard**
2. Select a camera
3. Click **Start Live Detection**
4. System monitors feed in real-time

### 3. Review Violations

1. Navigate to **Violations** page
2. Filter by type, status, date, or camera
3. Click on a violation to view details
4. Confirm or mark as false positive

### 4. View Analytics

1. Go to **Analytics** page
2. View violation trends over time
3. See statistics by type, camera, and time of day

## Testing Without Django

You can test the AI detection independently:

```bash
# Test with webcam
python test_detector.py --webcam 0

# Test with video file
python test_detector.py --video sample_video.mp4

# Test with image
python test_detector.py --image sample_image.jpg
```

## Project Structure

```
traffic_violation_detector/
├── manage.py
├── requirements.txt
├── README.md
├── test_detector.py          # Standalone testing script
│
├── traffic_system/           # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── detector/                 # Main application
│   ├── models.py            # Database models
│   ├── views.py             # Web views
│   ├── urls.py              # URL routing
│   ├── admin.py             # Admin configuration
│   │
│   ├── ai_models/           # AI detection modules
│   │   ├── helmet_detector.py
│   │   ├── plate_detector.py
│   │   └── violation_processor.py
│   │
│   └── templates/           # HTML templates
│       └── detector/
│           ├── base.html
│           ├── home.html
│           ├── violations.html
│           └── analytics.html
│
├── media/                   # Uploaded files and violations
│   ├── uploads/
│   ├── violations/
│   └── processed/
│
└── models/                  # Pre-trained model weights
    └── yolov8n.pt
```

## Configuration

### System Settings

Adjust detection parameters in Django admin or `detector/models.py`:

- **Confidence Threshold**: Minimum confidence for detections (0.0-1.0)
- **Frame Skip**: Process every Nth frame (higher = faster but less accurate)
- **Rate Limiting**: Max violations per minute per location

### Camera Configuration

Add cameras through Django admin:

1. Go to `http://localhost:8000/admin`
2. Navigate to **Cameras**
3. Add camera details:
   - Name
   - Location
   - RTSP URL (for IP cameras)
   - Active status

## Training Custom Models

For better accuracy, train custom models on your specific data:

### Helmet Detection

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on custom dataset
model.train(
    data='helmet_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Save trained model
model.save('models/helmet_model.pt')
```

### License Plate Detection

Use existing plate detection models or train with:
- Indian license plate datasets
- ANPR (Automatic Number Plate Recognition) datasets

## API Endpoints

The system provides REST API endpoints:

- `GET /api/violations/` - List all violations
- `GET /api/violations/<id>/` - Get violation details
- `POST /upload/` - Upload video for processing

Example API usage:

```python
import requests

# Get violations
response = requests.get('http://localhost:8000/api/violations/')
violations = response.json()

for v in violations:
    print(f"Violation {v['id']}: {v['type']} at {v['time']}")
```

## Troubleshooting

### Common Issues

**1. YOLOv8 model not downloading**

```bash
# Manually download
pip install gdown
gdown https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mv yolov8n.pt models/
```

**2. OpenCV cannot open webcam**

- Check camera permissions
- Try different camera index (0, 1, 2)
- On Linux, add user to video group: `sudo usermod -a -G video $USER`

**3. CUDA out of memory**

- Reduce batch size in detection
- Use CPU instead: Set `gpu=False` in detectors
- Process fewer frames (increase frame_skip)

**4. Poor detection accuracy**

- Adjust confidence threshold
- Ensure good lighting in images/videos
- Train custom model on your specific scenarios
- Use higher resolution cameras

## Performance Optimization

### For Real-time Processing

1. **Use GPU**: Enable CUDA for 10x faster processing
2. **Frame Skipping**: Process every 3-5 frames instead of all
3. **Resolution**: Resize frames to 640x480 or 1280x720
4. **Batch Processing**: Process multiple frames together

### For Accuracy

1. **Higher Confidence**: Set threshold to 0.6-0.7
2. **Custom Training**: Train on your specific traffic scenarios
3. **Multiple Models**: Use ensemble of models for validation
4. **Post-processing**: Apply tracking to reduce false positives


## Acknowledgments

- **YOLOv8** by Ultralytics for object detection
- **EasyOCR** for license plate reading
- **Django** framework for web application
- **OpenCV** for computer vision operations

---

**Note**: This system is intended for traffic monitoring and research purposes. Ensure compliance with local privacy laws and regulations when deploying in production environments.