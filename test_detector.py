"""
Standalone Traffic Violation Detector
Test the AI detection without Django

Usage:
    python test_detector.py --video path/to/video.mp4
    python test_detector.py --webcam 0
    python test_detector.py --image path/to/image.jpg
"""

import cv2
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from detector.ai_models.helmet_detector import HelmetDetector
    from detector.ai_models.plate_detector import LicensePlateDetector
except ImportError:
    print("Warning: Could not import Django models. Using standalone mode.")
    # For standalone use, copy the detector classes here or adjust imports

class StandaloneDetector:
    """Standalone detector for testing"""
    
    def __init__(self):
        print("Initializing AI models...")
        self.helmet_detector = HelmetDetector(confidence_threshold=0.5)
        self.plate_detector = LicensePlateDetector(confidence_threshold=0.5)
        print("Models loaded successfully!")
    
    def process_image(self, image_path):
        """Process a single image"""
        print(f"Processing image: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Detect violations
        helmet_violations, helmet_annotated = self.helmet_detector.detect_helmet_violation(frame)
        plate_violations, plate_annotated = self.plate_detector.detect_plate_violation(frame)
        
        # Combine annotations
        combined = cv2.addWeighted(helmet_annotated, 0.5, plate_annotated, 0.5, 0)
        
        # Print results
        print(f"\n--- Detection Results ---")
        print(f"Helmet violations: {len(helmet_violations)}")
        print(f"License plate violations: {len(plate_violations)}")
        
        # Show result
        cv2.imshow('Traffic Violations', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save output
        output_path = f"output_{Path(image_path).name}"
        cv2.imwrite(output_path, combined)
        print(f"Output saved to: {output_path}")
    
    def process_video(self, video_path):
        """Process video file"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Output video
        output_path = f"output_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_helmet_violations = 0
        total_plate_violations = 0
        frame_count = 0
        
        print("\nProcessing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for speed
            if frame_count % 5 == 0:
                helmet_violations, helmet_annotated = self.helmet_detector.detect_helmet_violation(frame)
                plate_violations, plate_annotated = self.plate_detector.detect_plate_violation(frame)
                
                combined = cv2.addWeighted(helmet_annotated, 0.5, plate_annotated, 0.5, 0)
                
                total_helmet_violations += len(helmet_violations)
                total_plate_violations += len(plate_violations)
                
                # Write frame
                out.write(combined)
                
                # Show progress
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | "
                          f"Helmet: {total_helmet_violations} | "
                          f"Plate: {total_plate_violations}")
            else:
                out.write(frame)
        
        cap.release()
        out.release()
        
        print(f"\n--- Final Results ---")
        print(f"Total frames processed: {frame_count}")
        print(f"Total helmet violations: {total_helmet_violations}")
        print(f"Total plate violations: {total_plate_violations}")
        print(f"Output saved to: {output_path}")
    
    def process_webcam(self, camera_index=0):
        """Process live webcam feed"""
        print(f"Opening webcam {camera_index}...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam {camera_index}")
            return
        
        print("Webcam opened successfully!")
        print("Press 'q' to quit, 's' to save screenshot")
        
        total_helmet_violations = 0
        total_plate_violations = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better real-time performance
            if frame_count % 3 == 0:
                helmet_violations, helmet_annotated = self.helmet_detector.detect_helmet_violation(frame)
                plate_violations, plate_annotated = self.plate_detector.detect_plate_violation(frame)
                
                combined = cv2.addWeighted(helmet_annotated, 0.5, plate_annotated, 0.5, 0)
                
                total_helmet_violations += len(helmet_violations)
                total_plate_violations += len(plate_violations)
                
                # Add stats overlay
                cv2.putText(combined, f"Helmet: {total_helmet_violations}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined, f"Plate: {total_plate_violations}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Traffic Monitor - Press Q to quit', combined)
            else:
                cv2.imshow('Traffic Monitor - Press Q to quit', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n--- Session Summary ---")
        print(f"Total helmet violations: {total_helmet_violations}")
        print(f"Total plate violations: {total_plate_violations}")

def main():
    parser = argparse.ArgumentParser(description='AI Traffic Violation Detector')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--webcam', type=int, help='Webcam index (usually 0)')
    
    args = parser.parse_args()
    
    if not any([args.video, args.image, args.webcam is not None]):
        parser.print_help()
        return
    
    # Initialize detector
    detector = StandaloneDetector()
    
    # Process based on input
    if args.image:
        detector.process_image(args.image)
    elif args.video:
        detector.process_video(args.video)
    elif args.webcam is not None:
        detector.process_webcam(args.webcam)

if __name__ == '__main__':
    main()