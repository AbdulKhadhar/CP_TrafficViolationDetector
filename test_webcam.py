"""
Test Webcam Connection
Run this to debug webcam issues
"""

import cv2
import sys

print("=" * 50)
print("Webcam Connection Test")
print("=" * 50)
print()

# Test different camera indices
for index in range(3):
    print(f"Testing camera index {index}...")
    
    # Try with DirectShow (Windows)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        print(f"  ✓ Camera {index} opened successfully!")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Frame captured successfully! Shape: {frame.shape}")
            
            # Show frame for 2 seconds
            cv2.imshow(f'Camera {index}', frame)
            print(f"  Displaying frame for 2 seconds...")
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        else:
            print(f"  ✗ Failed to capture frame")
        
        cap.release()
        print()
    else:
        print(f"  ✗ Camera {index} could not be opened")
        print()

print("=" * 50)
print("Test Complete!")
print()
print("If no cameras opened:")
print("1. Make sure webcam is connected")
print("2. Close other apps using camera (Zoom, Teams, Skype)")
print("3. Check Windows Privacy Settings:")
print("   Settings → Privacy → Camera → Allow apps to access camera")
print("4. Try running as Administrator")
print()
print("If a camera opened successfully, use that index in Django")
print("=" * 50)

# Keep window open
input("Press Enter to exit...")