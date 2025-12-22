# Python Programming for Robotics

## Overview
Python is widely used in robotics for rapid prototyping, simulation, and high-level control. Its simplicity and extensive libraries make it ideal for robotics applications, from basic control algorithms to complex AI implementations.

## Key Libraries
- **NumPy**: Numerical computations and array operations
- **SciPy**: Scientific computing and advanced algorithms
- **Matplotlib**: Data visualization and plotting
- **OpenCV-Python**: Computer vision and image processing
- **Pygame**: Game development and simulation interfaces
- **Serial**: Serial communication with microcontrollers
- **Math**: Mathematical functions for kinematics and control

## Essential Concepts
### 1. Data Structures for Robotics
```python
import numpy as np

# Pose representation (position + orientation)
class Pose:
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.position = np.array([x, y, z])
        self.orientation = np.array([roll, pitch, yaw])  # Euler angles

    def distance_to(self, other_pose):
        return np.linalg.norm(self.position - other_pose.position)

# Transform matrices for coordinate transformations
def create_rotation_matrix(yaw, pitch, roll):
    # Create rotation matrix from Euler angles
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # Rotation matrix around ZYX axes
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R
```

### 2. Robot Control Patterns
```python
import time
import threading

class RobotController:
    def __init__(self):
        self.current_pose = Pose()
        self.target_pose = Pose()
        self.is_running = False
        self.velocity = np.zeros(3)  # Linear velocity
        self.angular_velocity = np.zeros(3)  # Angular velocity

    def move_to_position(self, target_x, target_y, target_z):
        """Simple proportional controller to move to target position"""
        kp = 1.0  # Proportional gain

        while self.is_running:
            # Calculate error
            error_x = target_x - self.current_pose.position[0]
            error_y = target_y - self.current_pose.position[1]
            error_z = target_z - self.current_pose.position[2]

            # Simple proportional control
            self.velocity[0] = kp * error_x
            self.velocity[1] = kp * error_y
            self.velocity[2] = kp * error_z

            # Check if close enough to target
            if abs(error_x) < 0.01 and abs(error_y) < 0.01 and abs(error_z) < 0.01:
                break

            time.sleep(0.01)  # 100 Hz control loop

    def start_control_loop(self):
        """Start the control loop in a separate thread"""
        self.is_running = True
        control_thread = threading.Thread(target=self._control_loop)
        control_thread.start()

    def _control_loop(self):
        """Internal control loop"""
        while self.is_running:
            # Update robot state
            self._update_state()
            time.sleep(0.01)  # 100 Hz
```

### 3. Sensor Data Processing
```python
import statistics
from collections import deque

class SensorFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.readings = deque(maxlen=window_size)

    def add_reading(self, value):
        """Add a new sensor reading and return filtered value"""
        self.readings.append(value)
        return self.get_filtered_value()

    def get_filtered_value(self):
        """Apply median filter to reduce noise"""
        if len(self.readings) == 0:
            return 0
        return statistics.median(self.readings)

class SensorFusion:
    def __init__(self):
        self.accel_filter = SensorFilter(window_size=5)
        self.gyro_filter = SensorFilter(window_size=5)
        self.magnetometer_filter = SensorFilter(window_size=5)

    def fuse_imu_data(self, accel_data, gyro_data, mag_data):
        """Fuse IMU sensor data to estimate orientation"""
        # Apply filters
        filtered_accel = self.accel_filter.add_reading(accel_data)
        filtered_gyro = self.gyro_filter.add_reading(gyro_data)
        filtered_mag = self.magnetometer_filter.add_reading(mag_data)

        # Complementary filter implementation
        # Combine accelerometer (orientation relative to gravity)
        # with gyroscope (angular velocity integration)
        dt = 0.01  # Time step
        alpha = 0.98  # Weight for gyro vs accel

        # Simplified complementary filter
        estimated_orientation = alpha * (self.last_orientation + gyro_data * dt) + \
                               (1 - alpha) * accel_data

        self.last_orientation = estimated_orientation
        return estimated_orientation
```

## Best Practices
- Use type hints for better code readability
- Implement proper exception handling
- Use context managers for resource management
- Profile code for real-time performance
- Follow PEP 8 style guidelines
- Use virtual environments for dependency management
- Write unit tests for critical functions