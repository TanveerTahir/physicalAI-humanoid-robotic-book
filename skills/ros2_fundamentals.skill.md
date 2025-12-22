# ROS2 Fundamentals

## Overview
ROS2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

## Key Concepts
- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous, goal-oriented communication with feedback
- **Packages**: Organizational unit containing nodes, data, and configuration

## Essential Commands
```bash
# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build packages
colcon build --packages-select <package_name>

# Source the workspace
source install/setup.bash

# Run a node
ros2 run <package_name> <node_name>

# List active nodes
ros2 node list

# List topics
ros2 topic list

# Echo a topic
ros2 topic echo <topic_name> <msg_type>

# Call a service
ros2 service call <service_name> <srv_type> <args>
```

## Python Implementation
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()
```

## Best Practices
- Use meaningful names for nodes, topics, and services
- Implement proper error handling and logging
- Follow ROS2 naming conventions
- Use parameters for configurable values
- Implement lifecycle nodes for complex systems