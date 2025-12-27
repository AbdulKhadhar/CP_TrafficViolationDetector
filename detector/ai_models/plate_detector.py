import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re

class LicensePlateDetector:
    """Detects and reads license plates from vehicles"""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize license plate detector
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize EasyOCR reader
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
        except:
            self.reader = easyocr.Reader(['en'], gpu=False)
        
        # Load YOLOv8 for vehicle detection
        self.model = YOLO('yolov8n.pt')
        
        # Vehicle class IDs from COCO
        self.VEHICLE_CLASSES = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
    
    def detect_plate_violation(self, frame):
        """
        Detect vehicles without visible license plates
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            violations: List of violation dictionaries
            annotated_frame: Frame with bounding boxes
        """
        violations = []
        annotated_frame = frame.copy()
        
        # Detect vehicles
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in self.VEHICLE_CLASSES:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Extract vehicle region
                    vehicle_roi = frame[y1:y2, x1:x2]
                    
                    # Detect license plate in vehicle
                    plate_text, plate_conf, plate_bbox = self._detect_plate_in_roi(vehicle_roi)
                    
                    if plate_text is None or plate_conf < 0.5:
                        # No plate detected - violation
                        violation = {
                            'type': 'NO_PLATE',
                            'bbox': (x1, y1, x2, y2),
                            'vehicle_type': self.VEHICLE_CLASSES[cls],
                            'confidence': conf,
                            'plate_text': None
                        }
                        violations.append(violation)
                        
                        # Draw red bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    (0, 0, 255), 3)
                        cv2.putText(annotated_frame, 'NO PLATE', (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        # Plate detected - draw green box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f'Plate: {plate_text}', (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return violations, annotated_frame
    
    def _detect_plate_in_roi(self, roi):
        """
        Detect and read license plate in vehicle ROI
        
        Returns:
            plate_text: Extracted plate text or None
            confidence: OCR confidence score
            bbox: Bounding box of plate in ROI
        """
        if roi.size == 0:
            return None, 0.0, None
        
        # Preprocess image for better plate detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edged = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        plate_contour = None
        
        # Find rectangular contour (potential plate)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            if len(approx) == 4:  # Rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                # License plates typically have aspect ratio between 2:1 and 5:1
                if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                    plate_contour = approx
                    break
        
        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate_roi = roi[y:y+h, x:x+w]
            
            # Use OCR to read plate
            plate_text, confidence = self._read_plate_text(plate_roi)
            
            return plate_text, confidence, (x, y, w, h)
        
        # If no contour found, try OCR on entire ROI
        plate_text, confidence = self._read_plate_text(roi)
        return plate_text, confidence, None
    
    def _read_plate_text(self, plate_img):
        """
        Read text from plate image using OCR
        
        Returns:
            text: Extracted text or None
            confidence: Average confidence score
        """
        if plate_img.size == 0:
            return None, 0.0
        
        try:
            # Use EasyOCR
            results = self.reader.readtext(plate_img)
            
            if not results:
                return None, 0.0
            
            # Get text with highest confidence
            best_result = max(results, key=lambda x: x[2])
            text = best_result[1]
            confidence = best_result[2]
            
            # Clean up text (remove spaces, special chars)
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Validate plate format (basic check)
            if len(text) >= 4 and any(c.isdigit() for c in text):
                return text, confidence
            
            return None, 0.0
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return None, 0.0
    
    def extract_plate_number(self, frame, vehicle_bbox):
        """
        Extract plate number from specific vehicle bounding box
        
        Args:
            frame: Full frame image
            vehicle_bbox: (x1, y1, x2, y2) of vehicle
            
        Returns:
            plate_text: Extracted plate text
            confidence: Detection confidence
        """
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_roi = frame[y1:y2, x1:x2]
        
        plate_text, confidence, _ = self._detect_plate_in_roi(vehicle_roi)
        return plate_text, confidence