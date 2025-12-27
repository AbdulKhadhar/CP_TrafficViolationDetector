import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

class HelmetDetector:
    """Detects motorcycles and helmets using YOLOv8"""
    
    def __init__(self, model_path='models/yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the helmet detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Load YOLOv8 model
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print(f"Model not found at {model_path}, downloading pretrained model...")
            self.model = YOLO('yolov8n.pt')  # Will auto-download
        
        # COCO class IDs we care about
        self.MOTORCYCLE_CLASS = 3  # motorcycle in COCO dataset
        self.PERSON_CLASS = 0      # person in COCO dataset
        
    def detect_helmet_violation(self, frame):
        """
        Detect helmet violations in a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            violations: List of violation dictionaries
            annotated_frame: Frame with bounding boxes drawn
        """
        violations = []
        annotated_frame = frame.copy()
        
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        motorcycles = []
        persons = []
        
        # Extract detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, xyxy)
                
                if cls == self.MOTORCYCLE_CLASS:
                    motorcycles.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
                elif cls == self.PERSON_CLASS:
                    persons.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
        
        # Check for helmet violations
        for motorcycle in motorcycles:
            m_x1, m_y1, m_x2, m_y2 = motorcycle['bbox']
            
            # Find persons near/on the motorcycle
            riders = []
            for person in persons:
                p_x1, p_y1, p_x2, p_y2 = person['bbox']
                
                # Check if person overlaps with motorcycle
                if self._boxes_overlap((m_x1, m_y1, m_x2, m_y2), 
                                      (p_x1, p_y1, p_x2, p_y2)):
                    riders.append(person)
            
            # Check for helmet on each rider
            for rider in riders:
                has_helmet = self._detect_helmet_on_person(frame, rider['bbox'])
                
                if not has_helmet:
                    violation = {
                        'type': 'NO_HELMET',
                        'bbox': rider['bbox'],
                        'confidence': rider['confidence'],
                        'vehicle_bbox': motorcycle['bbox']
                    }
                    violations.append(violation)
                    
                    # Draw bounding box on annotated frame
                    r_x1, r_y1, r_x2, r_y2 = rider['bbox']
                    cv2.rectangle(annotated_frame, (r_x1, r_y1), (r_x2, r_y2), 
                                (0, 0, 255), 3)
                    cv2.putText(annotated_frame, 'NO HELMET', (r_x1, r_y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        return violations, annotated_frame
    
    def _boxes_overlap(self, box1, box2, threshold=0.3):
        """Check if two bounding boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        overlap_ratio = intersection / box2_area if box2_area > 0 else 0
        return overlap_ratio > threshold
    
    def _detect_helmet_on_person(self, frame, person_bbox):
        """
        Check if person is wearing helmet by analyzing head region
        This is a simplified version - for production, use a trained helmet classifier
        """
        x1, y1, x2, y2 = person_bbox
        
        # Get head region (top 30% of person bbox)
        head_height = int((y2 - y1) * 0.3)
        head_region = frame[y1:y1+head_height, x1:x2]
        
        if head_region.size == 0:
            return False
        
        # Simple color-based detection (helmets are often darker/solid colors)
        # Convert to HSV
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Define helmet color ranges (simplified)
        # In production, use a CNN classifier trained on helmet/no-helmet images
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 100])
        
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
        dark_ratio = np.sum(mask > 0) / mask.size
        
        # If head region is mostly dark, assume helmet
        # This is a VERY simplified heuristic - use ML model in production
        return dark_ratio > 0.4
    
    def process_video(self, video_path, output_path=None):
        """Process a video file and detect violations"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_violations = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for efficiency
            if frame_count % 5 == 0:
                violations, annotated = self.detect_helmet_violation(frame)
                
                if violations:
                    all_violations.extend(violations)
                
                if output_path:
                    out.write(annotated)
        
        cap.release()
        if output_path:
            out.release()
        
        return all_violations