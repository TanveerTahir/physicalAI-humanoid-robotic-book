# Robotics Simulation

## Overview
Robotics simulation is essential for developing, testing, and validating robotic systems before deploying them on real hardware. Simulation allows for safe testing, rapid prototyping, and experimentation without the risk of damaging expensive hardware or causing safety issues.

## Key Simulation Platforms
- **Gazebo**: Physics-based simulation with realistic dynamics
- **Isaac Sim**: NVIDIA's simulation platform for AI and robotics
- **Unity Robotics**: Game engine-based simulation with rich graphics
- **Webots**: General-purpose robot simulator with built-in physics
- **PyBullet**: Physics engine for robotics and reinforcement learning
- **Mujoco**: Advanced physics simulation for research

## Essential Simulation Concepts

### 1. Physics Simulation Fundamentals
```python
import numpy as np
import math

class PhysicsSimulator:
    def __init__(self, gravity=9.81, time_step=0.001):
        self.gravity = gravity
        self.time_step = time_step
        self.objects = []

    def update_dynamics(self, obj, force, torque):
        """Update object dynamics using Newton-Euler equations"""
        # Linear motion: F = ma -> a = F/m
        acceleration = force / obj.mass
        obj.velocity += acceleration * self.time_step
        obj.position += obj.velocity * self.time_step

        # Rotational motion: τ = Iα -> α = τ/I
        angular_acceleration = torque / obj.moment_of_inertia
        obj.angular_velocity += angular_acceleration * self.time_step
        obj.orientation += obj.angular_velocity * self.time_step

        # Apply gravity
        gravity_force = np.array([0, -obj.mass * self.gravity, 0])
        obj.velocity += gravity_force / obj.mass * self.time_step
        obj.position += obj.velocity * self.time_step

    def detect_collision(self, obj1, obj2):
        """Simple sphere-sphere collision detection"""
        distance = np.linalg.norm(obj1.position - obj2.position)
        min_distance = obj1.radius + obj2.radius

        if distance < min_distance:
            # Calculate collision response
            collision_normal = (obj2.position - obj1.position) / distance
            relative_velocity = obj2.velocity - obj1.velocity
            velocity_along_normal = np.dot(relative_velocity, collision_normal)

            # Only resolve if objects are moving toward each other
            if velocity_along_normal < 0:
                # Calculate impulse for collision response
                impulse = -(1 + obj1.restitution) * velocity_along_normal
                impulse /= (1/obj1.mass + 1/obj2.mass)

                # Apply impulse
                obj1.velocity += impulse * collision_normal / obj1.mass
                obj2.velocity -= impulse * collision_normal / obj2.mass

            return True
        return False

    def apply_constraints(self, obj):
        """Apply environmental constraints (e.g., ground plane)"""
        # Ground collision
        if obj.position[1] - obj.radius < 0:  # Y is up in this coordinate system
            obj.position[1] = obj.radius
            obj.velocity[1] = -obj.velocity[1] * obj.restitution  # Bounce
```

### 2. Gazebo Integration with ROS2
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import TransformBroadcaster
import math

class GazeboRobotInterface(Node):
    def __init__(self):
        super().__init__('gazebo_robot_interface')

        # Publishers for Gazebo control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers for Gazebo feedback
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # Robot state
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.laser_data = None
        self.camera_image = None

    def odom_callback(self, msg):
        """Update robot pose from Gazebo odom"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.current_pose.position.x
        t.transform.translation.y = self.current_pose.position.y
        t.transform.translation.z = self.current_pose.position.z
        t.transform.rotation = self.current_pose.orientation
        self.tf_broadcaster.sendTransform(t)

    def move_robot(self, linear_vel, angular_vel):
        """Send velocity commands to simulated robot"""
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)

    def navigate_to_goal(self, goal_x, goal_y):
        """Simple navigation to goal position"""
        # Calculate error
        error_x = goal_x - self.current_pose.position.x
        error_y = goal_y - self.current_pose.position.y
        distance = math.sqrt(error_x**2 + error_y**2)

        # Calculate angle to goal
        goal_yaw = math.atan2(error_y, error_x)
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Simple proportional controller
        kp_linear = 0.5
        kp_angular = 1.0

        # Align with goal first
        angle_error = goal_yaw - current_yaw
        if abs(angle_error) > 0.1:  # 0.1 rad tolerance
            self.move_robot(0.0, kp_angular * angle_error)
        elif distance > 0.1:  # 0.1m tolerance
            self.move_robot(kp_linear * distance, 0.0)
        else:
            self.move_robot(0.0, 0.0)  # Reached goal

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)
```

### 3. Isaac Sim Integration
```python
import omni
from pxr import Usd, UsdGeom, Gf, Sdf
import carb
import numpy as np

class IsaacSimController:
    def __init__(self):
        self.world = None
        self.robot = None
        self.sensors = []

    def setup_environment(self, stage_path="/Isaac/Environments/SimpleRoom"):
        """Setup Isaac Sim environment"""
        # Get the world interface
        self.world = omni.isaac.core.utils.world.World()

        # Add ground plane
        omni.isaac.core.utils.stage.add_ground_plane("/World/defaultGround")

        # Add robot
        self.robot = self.add_robot("/World/Robot", "/Isaac/Robots/Carter/carter_urdf.usd")

        # Add sensors
        self.add_camera("/World/Robot/Camera", position=[0.2, 0.0, 0.1])
        self.add_lidar("/World/Robot/Lidar", position=[0.1, 0.0, 0.2])

    def add_robot(self, prim_path, usd_path):
        """Add robot to simulation"""
        import omni.isaac.core.robots as robots
        return robots.Robot(
            prim_path=prim_path,
            usd_path=usd_path,
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

    def add_camera(self, prim_path, position=[0.0, 0.0, 0.0]):
        """Add RGB camera to robot"""
        from omni.isaac.sensor import Camera
        camera = Camera(
            prim_path=prim_path,
            frequency=30,
            resolution=(640, 480)
        )
        camera.set_position(position)
        self.sensors.append(camera)
        return camera

    def add_lidar(self, prim_path, position=[0.0, 0.0, 0.0]):
        """Add LIDAR sensor to robot"""
        from omni.isaac.range_sensor import _range_sensor
        lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

        # Create LIDAR sensor
        lidar_config = {
            "sensor_period": 0.033,  # 30 Hz
            "samples_per_scan": 360,
            "max_range": 10.0,
            "min_range": 0.1
        }
        self.sensors.append(prim_path)
        return prim_path

    def run_simulation(self, steps=1000):
        """Run simulation for specified number of steps"""
        self.world.reset()

        for i in range(steps):
            # Get sensor data
            camera_data = self.get_camera_data()
            lidar_data = self.get_lidar_data()

            # Process sensor data and control robot
            control_commands = self.process_sensor_data(camera_data, lidar_data)
            self.execute_control_commands(control_commands)

            # Step simulation
            self.world.step(render=True)

    def process_sensor_data(self, camera_data, lidar_data):
        """Process sensor data to generate control commands"""
        # Placeholder for sensor processing logic
        # This would typically involve:
        # - Obstacle detection from LIDAR
        # - Object detection from camera
        # - Path planning based on sensor input
        return {"linear_velocity": 1.0, "angular_velocity": 0.0}
```

### 4. Unity Robotics Simulation
```python
# Unity Robotics Hub communication using ROS.NET or similar
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan, Image
import numpy as np

class UnityRobotController:
    def __init__(self):
        rospy.init_node('unity_robot_controller')

        # Publishers for Unity simulation
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # Subscribers for Unity feedback
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        self.laser_data = None
        self.image_data = None

    def laser_callback(self, msg):
        """Process LIDAR data from Unity simulation"""
        self.laser_data = msg.ranges

        # Detect obstacles
        min_distance = min([d for d in self.laser_data if d > 0])
        if min_distance < 0.5:  # 0.5m safety distance
            self.stop_robot()
            rospy.logwarn("Obstacle detected! Stopping robot.")

    def move_to_waypoint(self, x, y, theta=0.0):
        """Send navigation goal to Unity simulation"""
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0

        # Convert theta to quaternion
        from tf.transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, theta)
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]

        self.goal_pub.publish(goal)

    def follow_trajectory(self, waypoints):
        """Follow a sequence of waypoints in simulation"""
        for waypoint in waypoints:
            x, y = waypoint
            rospy.loginfo(f"Moving to waypoint: ({x}, {y})")
            self.move_to_waypoint(x, y)

            # Wait until goal is reached (simplified)
            rospy.sleep(2.0)
```

## Simulation Best Practices

### 1. Performance Optimization
```python
class SimulationOptimizer:
    def __init__(self):
        self.simulation_speed = 1.0  # Real-time by default
        self.collision_detection = True
        self.graphics_quality = "medium"

    def adjust_simulation_speed(self, target_fps):
        """Adjust simulation speed for performance"""
        if target_fps < 30:  # Slow performance
            self.simulation_speed = 0.5  # Run at half speed
            self.collision_detection = False  # Skip collision checks
        elif target_fps > 60:  # Fast performance
            self.simulation_speed = 2.0  # Run at double speed
            self.collision_detection = True

    def optimize_physics(self):
        """Optimize physics parameters for simulation"""
        # Reduce solver iterations for faster simulation
        self.physics_solver_iterations = 10  # Default is often 200

        # Use simplified collision meshes
        self.use_convex_hulls = True

        # Reduce simulation substeps
        self.simulation_substeps = 1  # Default is often 4-8
```

### 2. Validation and Verification
```python
class SimulationValidator:
    def __init__(self):
        self.metrics = {
            'position_error': [],
            'velocity_error': [],
            'timing_accuracy': []
        }

    def validate_with_real_robot(self, sim_data, real_data):
        """Compare simulation results with real robot data"""
        pos_error = np.linalg.norm(sim_data['position'] - real_data['position'])
        vel_error = np.linalg.norm(sim_data['velocity'] - real_data['velocity'])

        self.metrics['position_error'].append(pos_error)
        self.metrics['velocity_error'].append(vel_error)

    def calculate_validation_metrics(self):
        """Calculate validation metrics"""
        avg_pos_error = np.mean(self.metrics['position_error'])
        avg_vel_error = np.mean(self.metrics['velocity_error'])

        print(f"Average Position Error: {avg_pos_error:.3f}m")
        print(f"Average Velocity Error: {avg_vel_error:.3f}m/s")

        # Return True if simulation is considered valid
        return avg_pos_error < 0.1 and avg_vel_error < 0.05  # 10cm, 5cm/s thresholds
```

## Best Practices
- Start with simple models and gradually increase complexity
- Validate simulation results against real-world data
- Use appropriate time steps for numerical stability
- Implement proper logging and visualization
- Consider computational constraints for real-time simulation
- Use realistic sensor noise models
- Test edge cases and failure scenarios in simulation
- Document simulation assumptions and limitations