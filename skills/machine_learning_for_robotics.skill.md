# Machine Learning for Robotics

## Overview
Machine learning in robotics enables robots to learn from experience, adapt to new situations, and perform complex tasks that are difficult to program explicitly. This skill covers reinforcement learning, computer vision, sensor fusion, and other ML techniques specifically applied to robotics applications.

## Key Libraries and Frameworks
- **TensorFlow/PyTorch**: Deep learning frameworks
- **OpenCV**: Computer vision and image processing
- **scikit-learn**: Traditional ML algorithms
- **Stable-Baselines3**: Reinforcement learning library
- **ROS2**: Robot Operating System for integration
- **NumPy/Pandas**: Data processing and manipulation
- **Matplotlib/Seaborn**: Visualization tools

## Essential ML Techniques for Robotics

### 1. Reinforcement Learning for Robot Control
```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr

        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RobotNavigationAgent:
    def __init__(self, state_size, action_size):
        self.agent = DQNAgent(state_size, action_size)
        self.navigation_history = []

    def get_robot_state(self, sensor_data):
        """Convert robot sensor data to RL state representation"""
        # Example: combine LIDAR, camera, and pose data
        lidar_data = sensor_data.get('lidar', [])
        pose_data = sensor_data.get('pose', [0, 0, 0])  # x, y, theta
        velocity_data = sensor_data.get('velocity', [0, 0])  # linear, angular

        # Normalize and combine into state vector
        state = np.concatenate([
            np.array(lidar_data) / 10.0,  # Normalize LIDAR to 0-1
            np.array(pose_data) / 10.0,   # Normalize pose
            np.array(velocity_data)       # Velocity as-is
        ])
        return state

    def navigate_step(self, sensor_data, goal_position):
        """Perform one step of navigation using RL agent"""
        current_state = self.get_robot_state(sensor_data)
        action = self.agent.act(current_state)

        # Convert action to robot command
        # (e.g., action 0: move forward, action 1: turn left, etc.)
        robot_command = self.convert_action_to_command(action)

        # Calculate reward based on proximity to goal
        reward = self.calculate_navigation_reward(sensor_data, goal_position)

        return robot_command, reward

    def convert_action_to_command(self, action):
        """Convert discrete action to continuous robot command"""
        # Example mapping: action 0-3 to different movement directions
        commands = {
            0: {'linear': 1.0, 'angular': 0.0},    # Move forward
            1: {'linear': 0.0, 'angular': 1.0},    # Turn left
            2: {'linear': 0.0, 'angular': -1.0},   # Turn right
            3: {'linear': -0.5, 'angular': 0.0}    # Move backward
        }
        return commands.get(action, commands[0])
```

### 2. Sensor Fusion with Machine Learning
```python
import numpy as np
from scipy.stats import multivariate_normal
from filterpy.kalman import KalmanFilter
import torch
import torch.nn as nn

class SensorFusionNet(nn.Module):
    def __init__(self, sensor_count, feature_size, output_size):
        super(SensorFusionNet, self).__init__()
        self.sensor_count = sensor_count

        # Process each sensor input
        self.sensor_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            ) for _ in range(sensor_count)
        ])

        # Attention mechanism to weight sensor inputs
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4)

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(32 * sensor_count, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, sensor_inputs):
        """
        sensor_inputs: list of sensor data [sensor1_data, sensor2_data, ...]
        """
        encoded_sensors = []

        # Encode each sensor input
        for i, sensor_data in enumerate(sensor_inputs):
            encoded = self.sensor_encoders[i](sensor_data)
            encoded_sensors.append(encoded)

        # Stack encoded sensors
        stacked_sensors = torch.stack(encoded_sensors, dim=0)  # [sensor_count, batch, features]

        # Apply attention mechanism
        attended_sensors, _ = self.attention(
            stacked_sensors, stacked_sensors, stacked_sensors
        )

        # Flatten and pass through fusion layer
        fused_input = attended_sensors.view(-1, self.sensor_count * 32)
        output = self.fusion_layer(fused_input)

        return output

class KalmanFilterFusion:
    def __init__(self, dim_x, dim_z, dim_u=0):
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z, dim_u=dim_u)

        # Initialize with reasonable values
        self.kf.x = np.zeros((dim_x, 1))  # Initial state
        self.kf.P *= 1000  # Uncertainty in initial state
        self.kf.R = np.eye(dim_z) * 0.1  # Measurement noise
        self.kf.Q = np.eye(dim_x) * 0.1  # Process noise

    def update_sensors(self, sensor_measurements):
        """
        sensor_measurements: dict with sensor names and measurements
        e.g., {'lidar': [1.0, 2.0], 'camera': [1.5, 2.1], 'imu': [0.1]}
        """
        # Fuse sensor measurements using weighted average
        # based on each sensor's reliability
        weights = self.calculate_sensor_weights(sensor_measurements)

        fused_measurement = np.zeros(self.kf.dim_z)
        for i, (sensor_name, measurement) in enumerate(sensor_measurements.items()):
            if i < self.kf.dim_z:
                # Weight the measurement by sensor reliability
                weighted_measurement = np.array(measurement) * weights[sensor_name]
                fused_measurement += weighted_measurement

        # Update Kalman filter
        self.kf.update(fused_measurement)

    def predict(self):
        """Predict next state"""
        self.kf.predict()
        return self.kf.x.flatten()

    def calculate_sensor_weights(self, sensor_measurements):
        """Calculate reliability weights for each sensor"""
        weights = {}

        # Example: assign weights based on sensor characteristics
        for sensor_name in sensor_measurements.keys():
            if 'lidar' in sensor_name:
                weights[sensor_name] = 0.4  # High reliability for position
            elif 'camera' in sensor_name:
                weights[sensor_name] = 0.3  # Medium reliability
            elif 'imu' in sensor_name:
                weights[sensor_name] = 0.3  # Good for orientation/acceleration
            else:
                weights[sensor_name] = 0.1  # Default low weight

        return weights
```

### 3. Deep Learning for Robot Perception
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

class PerceptionCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(PerceptionCNN, self).__init__()

        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output

class VisionLanguageActionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_actions=5):
        super(VisionLanguageActionModel, self).__init__()

        # Vision processing
        self.vision_encoder = PerceptionCNN(num_classes=100)  # Feature extractor

        # Language processing
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(100 + embed_dim, 256),  # Combined vision and language features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, image, text_indices):
        # Process vision
        vision_features = self.vision_encoder(image)

        # Process language
        text_embedded = self.text_embedding(text_indices)
        text_features, _ = self.lstm(text_embedded)
        # Take the last output
        text_features = text_features[:, -1, :]  # [batch, embed_dim]

        # Concatenate vision and language features
        combined_features = torch.cat([vision_features, text_features], dim=1)

        # Generate action predictions
        action_logits = self.fusion(combined_features)
        return action_logits

class RobotPerceptionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.object_detector = PerceptionCNN(num_classes=20)  # 20 object classes
        self.vla_model = VisionLanguageActionModel(vocab_size=10000, num_actions=6)

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def detect_objects(self, image):
        """Detect objects in image using trained CNN"""
        self.object_detector.eval()

        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.object_detector(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # Get top predictions
        top_probs, top_classes = torch.topk(probabilities, 3)

        detections = []
        for prob, class_idx in zip(top_probs[0], top_classes[0]):
            detections.append({
                'class': int(class_idx),
                'confidence': float(prob),
                'label': self.class_id_to_label(int(class_idx))
            })

        return detections

    def execute_language_command(self, image, command_text, tokenizer):
        """Execute robot action based on language command and visual input"""
        self.vla_model.eval()

        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Tokenize text command
        text_indices = tokenizer.encode(command_text).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits = self.vla_model(image_tensor, text_indices)
            action_probs = torch.softmax(action_logits, dim=1)
            predicted_action = torch.argmax(action_probs, dim=1).item()

        return self.action_id_to_command(predicted_action)

    def class_id_to_label(self, class_id):
        """Convert class ID to human-readable label"""
        labels = {
            0: "person", 1: "robot", 2: "chair", 3: "table", 4: "cup",
            5: "bottle", 6: "book", 7: "laptop", 8: "phone", 9: "remote"
            # Add more labels as needed
        }
        return labels.get(class_id, f"object_{class_id}")

    def action_id_to_command(self, action_id):
        """Convert action ID to robot command"""
        commands = {
            0: {"type": "move_forward", "params": {"distance": 1.0}},
            1: {"type": "turn_left", "params": {"angle": 90}},
            2: {"type": "turn_right", "params": {"angle": 90}},
            3: {"type": "grasp", "params": {}},
            4: {"type": "move_to_object", "params": {"object_id": None}},
            5: {"type": "stop", "params": {}}
        }
        return commands.get(action_id, {"type": "invalid", "params": {}})
```

### 4. Learning from Demonstration (LfD)
```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

class LearningFromDemonstration:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # Use Gaussian Process for demonstration learning
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)

        self.demonstrations = []
        self.is_trained = False

    def add_demonstration(self, states, actions):
        """Add a demonstration trajectory"""
        if len(states) != len(actions):
            raise ValueError("States and actions must have the same length")

        self.demonstrations.append({
            'states': np.array(states),
            'actions': np.array(actions)
        })

    def train(self):
        """Train the model on all demonstrations"""
        if not self.demonstrations:
            raise ValueError("No demonstrations to train on")

        # Combine all demonstrations
        all_states = []
        all_actions = []

        for demo in self.demonstrations:
            all_states.extend(demo['states'])
            all_actions.extend(demo['actions'])

        X = np.array(all_states)
        y = np.array(all_actions)

        # Normalize data
        X_normalized = self.scaler_X.fit_transform(X)
        y_normalized = self.scaler_y.fit_transform(y)

        # Train Gaussian Process
        self.gp_model.fit(X_normalized, y_normalized)
        self.is_trained = True

    def predict_action(self, state):
        """Predict action for given state"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        state_normalized = self.scaler_X.transform([state])
        action_normalized = self.gp_model.predict(state_normalized)
        action = self.scaler_y.inverse_transform(action_normalized)

        return action.flatten()

    def generate_trajectory(self, initial_state, horizon=100):
        """Generate a trajectory using the learned policy"""
        trajectory = []
        current_state = initial_state.copy()

        for _ in range(horizon):
            action = self.predict_action(current_state)
            trajectory.append({
                'state': current_state.copy(),
                'action': action.copy()
            })

            # Simulate state transition (simplified)
            # In practice, use the actual robot dynamics model
            current_state = self.simulate_dynamics(current_state, action)

        return trajectory

    def simulate_dynamics(self, state, action):
        """Simulate one step of robot dynamics"""
        # Simplified dynamics model - replace with actual robot model
        next_state = state + action * 0.01  # Simple integration
        return next_state

class BehavioralCloningAgent:
    def __init__(self, state_dim, action_dim):
        self.lfd = LearningFromDemonstration(state_dim, action_dim)
        self.behavior_model = None

    def learn_from_expert(self, expert_demos):
        """Learn to imitate expert behavior"""
        for demo in expert_demos:
            states = demo['states']
            actions = demo['actions']
            self.lfd.add_demonstration(states, actions)

        self.lfd.train()

    def execute_behavior(self, current_state):
        """Execute learned behavior"""
        predicted_action = self.lfd.predict_action(current_state)
        return predicted_action
```

## Practical Implementation Tips

### 1. Real-time ML Inference
```python
import time
from collections import deque

class RealTimeMLInference:
    def __init__(self, model, max_fps=30):
        self.model = model
        self.max_fps = max_fps
        self.frame_interval = 1.0 / max_fps
        self.last_inference_time = 0
        self.inference_times = deque(maxlen=10)

    def should_infer(self):
        """Check if it's time to run inference"""
        current_time = time.time()
        return (current_time - self.last_inference_time) >= self.frame_interval

    def run_inference(self, input_data):
        """Run model inference with performance monitoring"""
        if not self.should_infer():
            return None

        start_time = time.time()

        with torch.no_grad():
            result = self.model(input_data)

        end_time = time.time()
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)

        self.last_inference_time = end_time

        # Monitor performance
        avg_time = sum(self.inference_times) / len(self.inference_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0

        if current_fps < self.max_fps * 0.8:
            print(f"Warning: ML inference running at {current_fps:.2f} FPS, below target {self.max_fps}")

        return result
```

### 2. Data Collection and Preprocessing
```python
import pickle
import os
from datetime import datetime

class RobotDataCollector:
    def __init__(self, save_dir="robot_data"):
        self.save_dir = save_dir
        self.episode_data = []
        self.episode_count = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def start_episode(self):
        """Start collecting data for a new episode"""
        self.episode_data = []
        self.episode_start_time = time.time()

    def record_step(self, state, action, reward, next_state, done):
        """Record a single step of robot interaction"""
        step_data = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'timestamp': time.time()
        }
        self.episode_data.append(step_data)

    def end_episode(self, success=False):
        """End the current episode and save data"""
        episode_info = {
            'episode_id': self.episode_count,
            'data': self.episode_data,
            'success': success,
            'duration': time.time() - self.episode_start_time,
            'total_reward': sum([step['reward'] for step in self.episode_data])
        }

        # Save episode data
        filename = f"episode_{self.episode_count:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(episode_info, f)

        self.episode_count += 1
        print(f"Episode {self.episode_count-1} saved: {filepath}")
```

## Best Practices
- Collect diverse training data that covers the operational environment
- Use simulation for initial training, then transfer to real robot (sim-to-real)
- Implement proper data preprocessing and normalization
- Monitor model performance in real-time and trigger retraining when needed
- Use appropriate model complexity for computational constraints
- Implement safety mechanisms to handle uncertain predictions
- Validate models thoroughly in simulation before real-world deployment
- Keep models updated with new experiences and data