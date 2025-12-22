# Computer Vision for Robotics

## Overview
Computer vision is crucial for robots to perceive and understand their environment. It enables robots to detect objects, navigate spaces, recognize patterns, and interact with the physical world. This skill covers fundamental computer vision techniques specifically applied to robotics.

## Key Libraries
- **OpenCV**: Primary computer vision library
- **NumPy**: Array operations for image processing
- **PIL/Pillow**: Python Imaging Library for image manipulation
- **scikit-image**: Advanced image processing algorithms
- **TensorFlow/PyTorch**: Deep learning frameworks for AI vision
- **ROS2 CV Bridge**: Converting between ROS and OpenCV formats

## Essential Techniques

### 1. Image Preprocessing
```python
import cv2
import numpy as np

def preprocess_image(image, target_size=(640, 480)):
    """Basic image preprocessing for robotics applications"""
    # Resize image
    resized = cv2.resize(image, target_size)

    # Convert to grayscale if needed
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adjust brightness and contrast
    adjusted = cv2.convertScaleAbs(blurred, alpha=1.2, beta=30)

    return adjusted

def enhance_image_features(image):
    """Enhance features for better detection"""
    # Apply histogram equalization
    equalized = cv2.equalizeHist(image)

    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    enhanced = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)

    return enhanced
```

### 2. Object Detection and Tracking
```python
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        self.tracker = None
        self.tracking = False

    def detect_by_color(self, image, lower_hsv, upper_hsv):
        """Detect objects based on color range"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Apply morphological operations to clean up mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                detected_objects.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': cv2.contourArea(contour)
                })

        return detected_objects

    def detect_aruco_markers(self, image):
        """Detect ArUco markers for precise positioning"""
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            image, aruco_dict, parameters=parameters
        )

        marker_info = []
        if ids is not None:
            for i in range(len(ids)):
                # Calculate center of marker
                corner_points = corners[i][0]
                center_x = int(np.mean(corner_points[:, 0]))
                center_y = int(np.mean(corner_points[:, 1]))

                marker_info.append({
                    'id': int(ids[i][0]),
                    'corners': corner_points,
                    'center': (center_x, center_y)
                })

        return marker_info

    def start_tracking(self, image, bbox):
        """Initialize tracker for object tracking"""
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking = self.tracker.init(image, bbox)
        return self.tracking
```

### 3. Feature Detection and Matching
```python
import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def detect_and_compute_features(self, image):
        """Detect and compute SIFT features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_threshold=0.75):
        """Match features using FLANN matcher"""
        if desc1 is None or desc2 is None:
            return []

        matches = self.bf.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def find_object_pose(self, img1, img2):
        """Find object pose using feature matching"""
        kp1, desc1 = self.detect_and_compute_features(img1)
        kp2, desc2 = self.detect_and_compute_features(img2)

        matches = self.match_features(desc1, desc2)

        if len(matches) >= 10:
            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            return M, matches, mask
        else:
            return None, [], None
```

### 4. Depth Perception and Stereo Vision
```python
import cv2
import numpy as np

class StereoVision:
    def __init__(self, focal_length, baseline):
        self.focal_length = focal_length  # Camera focal length in pixels
        self.baseline = baseline  # Distance between stereo cameras in meters

    def calculate_depth(self, disparity_map):
        """Calculate depth from disparity map"""
        # Avoid division by zero
        disparity_map = np.where(disparity_map == 0, 1e-6, disparity_map)

        # Depth = (focal_length * baseline) / disparity
        depth_map = (self.focal_length * self.baseline) / disparity_map

        return depth_map

    def rectify_stereo_pair(self, left_img, right_img,
                           left_cam_mat, right_cam_mat,
                           left_dist_coeffs, right_dist_coeffs,
                           rotation, translation):
        """Rectify stereo image pair"""
        # Compute rectification parameters
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            left_cam_mat, left_dist_coeffs,
            right_cam_mat, right_dist_coeffs,
            left_img.shape[::-1], rotation, translation
        )

        # Compute undistortion and rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            left_cam_mat, left_dist_coeffs, R1, P1,
            left_img.shape[::-1], cv2.CV_32FC1
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            right_cam_mat, right_dist_coeffs, R2, P2,
            right_img.shape[::-1], cv2.CV_32FC1
        )

        # Apply rectification
        left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)

        return left_rectified, right_rectified
```

## Real-time Performance Tips
```python
import time
from collections import deque

class VisionPipeline:
    def __init__(self, max_fps=30):
        self.max_fps = max_fps
        self.frame_time = 1.0 / max_fps
        self.processing_times = deque(maxlen=10)

    def process_frame(self, image):
        """Process frame with performance monitoring"""
        start_time = time.time()

        # Your vision processing here
        processed_result = self.vision_algorithm(image)

        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        # Log performance
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        if fps < self.max_fps * 0.8:  # If significantly below target
            print(f"Warning: Current FPS {fps:.2f} below target {self.max_fps}")

        return processed_result

    def vision_algorithm(self, image):
        """Placeholder for actual vision algorithm"""
        # This would contain your specific computer vision implementation
        return image
```

## Best Practices
- Optimize algorithms for real-time performance
- Use appropriate image resolution for your application
- Implement proper error handling for camera failures
- Calibrate cameras for accurate measurements
- Consider lighting conditions in algorithm design
- Use GPU acceleration when available
- Validate results with ground truth when possible