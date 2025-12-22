# Control Systems for Robotics

## Overview
Control systems are fundamental to robotics, enabling precise movement, manipulation, and interaction with the environment. This skill covers classical and modern control techniques specifically applied to robotic systems, including PID controllers, trajectory planning, and advanced control strategies.

## Key Control Concepts
- **Feedback Control**: Using sensor measurements to adjust control outputs
- **Feedforward Control**: Predictive control based on desired trajectories
- **System Modeling**: Mathematical representation of robot dynamics
- **Stability Analysis**: Ensuring controlled behavior over time
- **Robustness**: Maintaining performance despite uncertainties

## Essential Control Techniques

### 1. PID Controllers
```python
import numpy as np
import time

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(None, None)):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        self.output_limits = output_limits  # (min, max) output limits

        self.reset()

    def reset(self):
        """Reset the PID controller state"""
        self.setpoint = 0.0
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None

    def compute(self, measurement, dt=None):
        """Compute control output based on measurement"""
        current_time = time.time()

        if dt is None:
            if self.previous_time is None:
                dt = 0.01  # Default time step
            else:
                dt = current_time - self.previous_time
        self.previous_time = current_time

        # Calculate error
        error = self.setpoint - measurement

        # Proportional term
        proportional = self.kp * error

        # Integral term (with anti-windup)
        self.integral += error * dt
        # Anti-windup: prevent integral term from growing too large
        if self.output_limits[0] is not None:
            self.integral = max(self.integral, self.output_limits[0] / self.ki if self.ki != 0 else self.integral)
        if self.output_limits[1] is not None:
            self.integral = min(self.integral, self.output_limits[1] / self.ki if self.ki != 0 else self.integral)

        integral_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        derivative_term = self.kd * derivative

        # Store current error for next iteration
        self.previous_error = error

        # Calculate total output
        output = proportional + integral_term + derivative_term

        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(output, self.output_limits[0])
        if self.output_limits[1] is not None:
            output = min(output, self.output_limits[1])

        return output

class RobotArmController:
    def __init__(self, joint_count=6):
        self.joint_count = joint_count
        # Create PID controllers for each joint
        self.joint_controllers = [
            PIDController(kp=50.0, ki=1.0, kd=10.0, output_limits=(-10, 10))
            for _ in range(joint_count)
        ]
        self.current_joint_positions = np.zeros(joint_count)
        self.target_joint_positions = np.zeros(joint_count)

    def set_target_positions(self, target_positions):
        """Set target joint positions"""
        self.target_joint_positions = np.array(target_positions)
        for i, controller in enumerate(self.joint_controllers):
            controller.setpoint = self.target_joint_positions[i]

    def update_control(self, current_positions, dt=0.01):
        """Update control for all joints"""
        self.current_joint_positions = np.array(current_positions)

        control_outputs = []
        for i, controller in enumerate(self.joint_controllers):
            output = controller.compute(self.current_joint_positions[i], dt)
            control_outputs.append(output)

        return np.array(control_outputs)
```

### 2. Trajectory Planning
```python
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class TrajectoryPlanner:
    def __init__(self):
        self.current_trajectory = None
        self.trajectory_time = 0.0

    def generate_minimal_jerk_trajectory(self, start_pos, end_pos, duration, dt=0.01):
        """Generate minimal jerk trajectory between two points"""
        t = np.arange(0, duration + dt, dt)

        # Minimal jerk trajectory coefficients
        # Position: s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        a0 = start_pos
        a1 = 0  # Start with zero velocity
        a2 = 0  # Start with zero acceleration
        a3 = 10 * (end_pos - start_pos) / (duration**3)
        a4 = -15 * (end_pos - start_pos) / (duration**4)
        a5 = 6 * (end_pos - start_pos) / (duration**5)

        # Calculate trajectory
        trajectory = np.zeros((len(t), 3))  # [position, velocity, acceleration]
        for i, time in enumerate(t):
            trajectory[i, 0] = a0 + a1*time + a2*(time**2) + a3*(time**3) + a4*(time**4) + a5*(time**5)  # Position
            trajectory[i, 1] = a1 + 2*a2*time + 3*a3*(time**2) + 4*a4*(time**3) + 5*a5*(time**4)  # Velocity
            trajectory[i, 2] = 2*a2 + 6*a3*time + 12*a4*(time**2) + 20*a5*(time**3)  # Acceleration

        return t, trajectory

    def generate_spline_trajectory(self, waypoints, duration, dt=0.01):
        """Generate smooth trajectory through waypoints using cubic splines"""
        t = np.linspace(0, duration, len(waypoints))
        spline = CubicSpline(t, waypoints, bc_type='clamped')  # Zero velocity at endpoints

        t_fine = np.arange(0, duration, dt)
        positions = spline(t_fine)

        # Calculate velocity and acceleration
        velocities = spline.derivative(nu=1)(t_fine)
        accelerations = spline.derivative(nu=2)(t_fine)

        trajectory = np.column_stack([positions, velocities, accelerations])
        return t_fine, trajectory

class CartesianTrajectoryController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.joint_trajectory_controller = TrajectoryPlanner()
        self.cartesian_trajectory_controller = TrajectoryPlanner()

    def plan_cartesian_trajectory(self, start_pose, end_pose, duration, dt=0.01):
        """Plan trajectory in Cartesian space and convert to joint space"""
        # Generate Cartesian trajectory
        t, cartesian_traj = self.cartesian_trajectory_controller.generate_minimal_jerk_trajectory(
            start_pose, end_pose, duration, dt
        )

        # Convert Cartesian trajectory to joint space using inverse kinematics
        joint_trajectories = []
        for i in range(len(cartesian_traj)):
            # This is a simplified example - in practice, use robot's IK solver
            joint_pos = self.robot_model.inverse_kinematics(cartesian_traj[i, 0])  # Position only
            joint_trajectories.append(joint_pos)

        return t, np.array(joint_trajectories)

    def execute_trajectory(self, trajectory, robot_interface):
        """Execute planned trajectory on robot"""
        for i, (time_step, joint_positions) in enumerate(zip(trajectory[0], trajectory[1])):
            robot_interface.set_joint_positions(joint_positions)
            time.sleep(0.01)  # Match the trajectory time step
```

### 3. Advanced Control Strategies
```python
class ModelPredictiveController:
    def __init__(self, prediction_horizon=10, control_horizon=5):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

    def solve_optimization(self, current_state, reference_trajectory, model):
        """Solve MPC optimization problem"""
        # This is a simplified implementation
        # In practice, use quadratic programming solver

        # Predict future states using model
        predicted_states = []
        current_x = current_state.copy()

        for k in range(self.prediction_horizon):
            # Apply control input (simplified - assumes constant control)
            control_input = np.zeros(model.control_dim)
            next_state = model.predict(current_x, control_input)
            predicted_states.append(next_state)
            current_x = next_state

        # Calculate cost based on tracking error
        total_cost = 0
        for k in range(self.prediction_horizon):
            if k < len(reference_trajectory):
                tracking_error = reference_trajectory[k] - predicted_states[k]
                total_cost += tracking_error.T @ tracking_error  # Quadratic cost

        return total_cost

class AdaptiveController:
    def __init__(self, initial_params):
        self.params = initial_params.copy()
        self.param_history = [initial_params.copy()]

    def update_params(self, error, phi):
        """Update controller parameters using gradient descent"""
        # Simple gradient-based parameter update
        learning_rate = 0.01
        param_update = learning_rate * error * phi
        self.params += param_update
        self.param_history.append(self.params.copy())

    def control(self, state, reference):
        """Generate control signal using adaptive parameters"""
        # Example: adaptive state feedback control
        control_signal = -self.params @ state + reference
        return control_signal

class RobustController:
    def __init__(self, nominal_model, uncertainty_bounds):
        self.nominal_model = nominal_model
        self.uncertainty_bounds = uncertainty_bounds

    def design_hinf_controller(self):
        """Design H-infinity controller for robustness"""
        # This would typically use specialized control design tools
        # Simplified implementation showing the concept

        # Calculate robust control gains
        # In practice, use tools like MATLAB's hinfsyn or Python's slycot
        K = self.calculate_robust_gains()
        return K

    def calculate_robust_gains(self):
        """Calculate robust control gains"""
        # Simplified robust gain calculation
        # This is a placeholder for actual H-infinity design
        return np.array([1.0, 1.0, 1.0])  # Placeholder gains
```

### 4. Multi-Input Multi-Output (MIMO) Control
```python
class MIMOController:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.gain_matrix = np.eye(num_inputs)  # Initialize as identity

    def compute_control(self, error_vector):
        """Compute control vector for MIMO system"""
        # u = K * e where u is control vector, K is gain matrix, e is error vector
        control_vector = self.gain_matrix @ error_vector
        return control_vector

    def tune_decoupling_controller(self, plant_matrix):
        """Tune controller to minimize coupling between inputs/outputs"""
        # Calculate decoupling matrix
        decoupling_matrix = np.linalg.inv(plant_matrix)

        # Update gain matrix to account for plant dynamics
        self.gain_matrix = decoupling_matrix @ np.eye(self.num_inputs)

class RobotLocomotionController:
    def __init__(self):
        # For a bipedal robot, we might have multiple control objectives:
        # - Balance control (pitch, roll)
        # - Position control (x, y)
        # - Orientation control (yaw)
        self.balance_controller = PIDController(kp=10.0, ki=1.0, kd=5.0)
        self.position_controller = PIDController(kp=5.0, ki=0.5, kd=2.0)
        self.orientation_controller = PIDController(kp=8.0, ki=0.8, kd=3.0)

    def compute_balance_control(self, imu_data, target_orientation):
        """Compute control for maintaining balance"""
        # Extract orientation from IMU
        current_orientation = self.extract_orientation(imu_data)

        # Calculate orientation error
        orientation_error = self.calculate_orientation_error(
            current_orientation, target_orientation
        )

        # Generate balance control commands
        roll_control = self.balance_controller.compute(orientation_error[0])
        pitch_control = self.balance_controller.compute(orientation_error[1])

        return roll_control, pitch_control

    def extract_orientation(self, imu_data):
        """Extract orientation from IMU data"""
        # Simplified orientation extraction
        # In practice, use sensor fusion (e.g., complementary filter, Kalman filter)
        return np.array([imu_data.roll, imu_data.pitch, imu_data.yaw])

    def calculate_orientation_error(self, current, target):
        """Calculate orientation error"""
        error = target - current
        # Handle angle wrapping for yaw
        if abs(error[2]) > np.pi:
            error[2] = error[2] - np.sign(error[2]) * 2 * np.pi
        return error
```

## Real-time Control Implementation
```python
import threading
import time
from collections import deque

class RealTimeController:
    def __init__(self, control_frequency=100):  # 100 Hz control
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.is_running = False
        self.control_thread = None

        # Control components
        self.controllers = {}
        self.state_buffer = deque(maxlen=10)
        self.command_buffer = deque(maxlen=10)

    def add_controller(self, name, controller):
        """Add a controller to the real-time system"""
        self.controllers[name] = controller

    def start_control_loop(self):
        """Start the real-time control loop"""
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the real-time control loop"""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()

    def _control_loop(self):
        """Internal real-time control loop"""
        last_time = time.time()

        while self.is_running:
            current_time = time.time()

            # Maintain control rate
            if current_time - last_time >= self.control_period:
                try:
                    # Read current state
                    current_state = self.read_robot_state()

                    # Update controllers
                    control_commands = {}
                    for name, controller in self.controllers.items():
                        command = controller.compute(current_state[name])
                        control_commands[name] = command

                    # Send commands to robot
                    self.send_commands(control_commands)

                    # Update timing
                    last_time = current_time

                    # Store for diagnostics
                    self.state_buffer.append(current_state)
                    self.command_buffer.append(control_commands)

                except Exception as e:
                    print(f"Control loop error: {e}")
                    # Implement safe behavior
                    self.emergency_stop()
            else:
                # Sleep briefly to avoid busy waiting
                time.sleep(0.001)

    def read_robot_state(self):
        """Read current robot state"""
        # Placeholder - implement based on robot interface
        return {"position": 0.0, "velocity": 0.0}

    def send_commands(self, commands):
        """Send control commands to robot"""
        # Placeholder - implement based on robot interface
        pass

    def emergency_stop(self):
        """Emergency stop procedure"""
        for name, controller in self.controllers.items():
            if hasattr(controller, 'reset'):
                controller.reset()
        self.send_commands({name: 0.0 for name in self.controllers.keys()})
```

## Best Practices
- Always implement safety limits and emergency stops
- Use appropriate control frequencies for your system
- Implement proper sensor fusion for state estimation
- Test controllers with realistic simulation before hardware deployment
- Monitor control performance and adjust gains as needed
- Implement logging for debugging and tuning
- Consider computational constraints in real-time systems
- Validate stability margins and robustness