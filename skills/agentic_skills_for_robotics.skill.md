# Agentic Skills for Robotics

## Overview
Agentic skills in robotics refer to the ability of robotic systems to act autonomously, make decisions, plan actions, and adapt to changing environments. These skills enable robots to function as intelligent agents that can perceive their environment, reason about situations, plan sequences of actions, and execute complex tasks with minimal human intervention.

## Key Concepts
- **Autonomous Decision Making**: Robots making independent choices based on sensor data and goals
- **Multi-step Planning**: Breaking down complex tasks into manageable subtasks
- **Reactive Behavior**: Responding appropriately to environmental changes
- **Learning from Experience**: Improving performance through interaction
- **Goal-oriented Actions**: Focusing on achieving specific objectives

## Essential Agentic Techniques

### 1. Autonomous Planning and Reasoning
```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class RobotTask:
    id: str
    name: str
    description: str
    priority: int
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0
    estimated_duration: Optional[float] = None

class PlanningAgent:
    def __init__(self):
        self.tasks: Dict[str, RobotTask] = {}
        self.robot_capabilities = {}
        self.environment_model = {}
        self.goals = []

    def add_task(self, task: RobotTask):
        """Add a task to the planning system"""
        self.tasks[task.id] = task

    def plan_task_sequence(self, goal: str) -> List[RobotTask]:
        """Plan sequence of tasks to achieve a goal"""
        # Analyze goal and break into subtasks
        subtasks = self.analyze_goal(goal)

        # Check dependencies and resolve conflicts
        ordered_tasks = self.resolve_dependencies(subtasks)

        # Optimize task sequence
        optimized_tasks = self.optimize_task_sequence(ordered_tasks)

        return optimized_tasks

    def analyze_goal(self, goal: str) -> List[RobotTask]:
        """Analyze goal and decompose into subtasks"""
        # This would use more sophisticated planning algorithms
        # For now, we'll implement a simple decomposition
        if "navigate to" in goal.lower():
            location = goal.split("navigate to")[-1].strip()
            return [
                RobotTask(
                    id=f"check_path_{location}",
                    name="Check Path",
                    description=f"Verify path to {location} is clear",
                    priority=1,
                    dependencies=[]
                ),
                RobotTask(
                    id=f"navigate_{location}",
                    name="Navigate",
                    description=f"Navigate to {location}",
                    priority=2,
                    dependencies=["check_path_{location}"]
                )
            ]
        elif "pick up" in goal.lower():
            object_name = goal.split("pick up")[-1].strip()
            return [
                RobotTask(
                    id=f"locate_{object_name}",
                    name="Locate Object",
                    description=f"Locate {object_name}",
                    priority=1,
                    dependencies=[]
                ),
                RobotTask(
                    id=f"approach_{object_name}",
                    name="Approach Object",
                    description=f"Approach {object_name}",
                    priority=2,
                    dependencies=[f"locate_{object_name}"]
                ),
                RobotTask(
                    id=f"grasp_{object_name}",
                    name="Grasp Object",
                    description=f"Grasp {object_name}",
                    priority=3,
                    dependencies=[f"approach_{object_name}"]
                )
            ]

        return []

    def resolve_dependencies(self, tasks: List[RobotTask]) -> List[RobotTask]:
        """Resolve task dependencies and create execution order"""
        # Topological sort to handle dependencies
        resolved_tasks = []
        pending_tasks = tasks.copy()

        while pending_tasks:
            ready_tasks = []
            for task in pending_tasks:
                # Check if all dependencies are satisfied
                dependencies_met = True
                for dep_id in task.dependencies:
                    if dep_id not in [t.id for t in resolved_tasks]:
                        dependencies_met = False
                        break

                if dependencies_met:
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency detected
                logging.warning("Circular dependency detected in task planning")
                break

            # Add ready tasks to resolved list
            for task in ready_tasks:
                resolved_tasks.append(task)
                pending_tasks.remove(task)

        return resolved_tasks

    def optimize_task_sequence(self, tasks: List[RobotTask]) -> List[RobotTask]:
        """Optimize task sequence for efficiency"""
        # Sort by priority and dependencies
        return sorted(tasks, key=lambda t: t.priority)

    def assign_tasks_to_robots(self, tasks: List[RobotTask], robots: List[Dict]) -> Dict[str, List[RobotTask]]:
        """Assign tasks to available robots based on capabilities"""
        robot_assignments = {robot['id']: [] for robot in robots}

        for task in tasks:
            # Find suitable robot based on capabilities
            suitable_robots = [
                robot for robot in robots
                if self.is_robot_suitable(robot, task)
            ]

            if suitable_robots:
                # Assign to robot with least current workload
                assigned_robot = min(
                    suitable_robots,
                    key=lambda r: len(robot_assignments[r['id']])
                )
                robot_assignments[assigned_robot['id']].append(task)
                task.assigned_robot = assigned_robot['id']
            else:
                logging.warning(f"No suitable robot found for task {task.id}")

        return robot_assignments

    def is_robot_suitable(self, robot: Dict, task: RobotTask) -> bool:
        """Check if robot is suitable for given task"""
        required_capabilities = self.get_required_capabilities(task)

        for capability in required_capabilities:
            if capability not in robot.get('capabilities', []):
                return False

        return True

    def get_required_capabilities(self, task: RobotTask) -> List[str]:
        """Get required capabilities for a task"""
        if "navigate" in task.name.lower():
            return ["navigation", "path_planning"]
        elif "grasp" in task.name.lower():
            return ["manipulation", "grasping"]
        elif "detect" in task.name.lower():
            return ["vision", "object_detection"]

        return ["basic_mobility"]

class ReasoningAgent:
    def __init__(self):
        self.belief_base = {}
        self.context = {}
        self.reasoning_history = []

    def update_beliefs(self, sensor_data: Dict):
        """Update belief base with new sensor information"""
        for key, value in sensor_data.items():
            self.belief_base[key] = {
                'value': value,
                'timestamp': time.time(),
                'confidence': self.calculate_confidence(value)
            }

    def calculate_confidence(self, value) -> float:
        """Calculate confidence in sensor reading"""
        # Simplified confidence calculation
        # In practice, this would use sensor fusion and uncertainty modeling
        return 0.9  # High confidence for now

    def reason_about_state(self, query: str) -> Dict[str, Any]:
        """Reason about current state based on beliefs"""
        if query == "is_path_clear":
            obstacles = self.belief_base.get('obstacles', {}).get('value', [])
            return {
                'result': len(obstacles) == 0,
                'confidence': self.belief_base.get('obstacles', {}).get('confidence', 0.5)
            }

        elif query == "is_object_present":
            objects = self.belief_base.get('detected_objects', {}).get('value', [])
            return {
                'result': len(objects) > 0,
                'confidence': self.belief_base.get('detected_objects', {}).get('confidence', 0.5)
            }

        return {'result': None, 'confidence': 0.0}

    def make_decision(self, options: List[Dict]) -> Dict:
        """Make decision based on current beliefs and context"""
        best_option = None
        best_score = float('-inf')

        for option in options:
            score = self.evaluate_option(option)
            if score > best_score:
                best_score = score
                best_option = option

        self.reasoning_history.append({
            'options': options,
            'decision': best_option,
            'score': best_score,
            'timestamp': time.time()
        })

        return best_option

    def evaluate_option(self, option: Dict) -> float:
        """Evaluate an option based on current state"""
        score = 0.0

        # Consider success probability
        success_prob = option.get('success_probability', 0.5)
        score += success_prob * 10

        # Consider resource cost
        resource_cost = option.get('resource_cost', 1.0)
        score -= resource_cost * 2

        # Consider time requirement
        time_required = option.get('time_required', 1.0)
        score -= time_required * 0.1

        # Consider safety factors
        safety_factor = option.get('safety_factor', 1.0)
        score += safety_factor * 5

        return score
```

### 2. Multi-Agent Coordination
```python
import asyncio
from typing import Set, Callable
import random

class RobotAgent:
    def __init__(self, robot_id: str, capabilities: List[str]):
        self.id = robot_id
        self.capabilities = set(capabilities)
        self.current_task = None
        self.position = (0, 0, 0)
        self.status = "idle"
        self.neighbors: Set['RobotAgent'] = set()
        self.message_queue = asyncio.Queue()
        self.task_queue = asyncio.Queue()

    async def communicate_with(self, other_robot: 'RobotAgent', message: Dict):
        """Send message to another robot"""
        await other_robot.message_queue.put({
            'from': self.id,
            'to': other_robot.id,
            'message': message,
            'timestamp': time.time()
        })

    async def broadcast_message(self, message: Dict):
        """Broadcast message to all neighbors"""
        for neighbor in self.neighbors:
            await self.communicate_with(neighbor, message)

    async def coordinate_with_team(self, task: RobotTask):
        """Coordinate with team members for task execution"""
        # Announce task availability
        await self.broadcast_message({
            'type': 'task_announcement',
            'task': task,
            'robot_id': self.id
        })

        # Wait for coordination responses
        responses = []
        start_time = time.time()

        while time.time() - start_time < 5.0:  # Wait 5 seconds for responses
            try:
                response = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=0.1
                )

                if response['message']['type'] == 'task_response':
                    responses.append(response['message'])
            except asyncio.TimeoutError:
                continue

        # Process responses and coordinate
        coordination_result = self.process_coordination_responses(responses, task)
        return coordination_result

    def process_coordination_responses(self, responses: List[Dict], task: RobotTask) -> Dict:
        """Process coordination responses from team"""
        if not responses:
            # No coordination needed, proceed alone
            return {'action': 'proceed_alone', 'task': task}

        # Analyze responses
        available_robots = []
        for response in responses:
            if response['status'] == 'available':
                available_robots.append(response['robot_id'])

        if len(available_robots) == 0:
            return {'action': 'proceed_alone', 'task': task}

        # Decide on coordination strategy
        if task.name.lower() in ['transport', 'assembly', 'mapping']:
            # These tasks benefit from multiple robots
            return {
                'action': 'coordinate',
                'task': task,
                'team_members': [self.id] + available_robots
            }
        else:
            # Single robot sufficient
            return {'action': 'proceed_alone', 'task': task}

    async def execute_task_with_coordination(self, task: RobotTask):
        """Execute task with team coordination"""
        coordination_result = await self.coordinate_with_team(task)

        if coordination_result['action'] == 'proceed_alone':
            return await self.execute_task_alone(task)
        elif coordination_result['action'] == 'coordinate':
            return await self.execute_task_as_team(
                task,
                coordination_result['team_members']
            )

    async def execute_task_alone(self, task: RobotTask):
        """Execute task as single robot"""
        self.current_task = task
        self.status = "executing"

        # Simulate task execution
        for progress in range(0, 101, 10):
            task.progress = progress / 100.0
            await asyncio.sleep(0.5)  # Simulate work
            await self.broadcast_message({
                'type': 'task_progress',
                'task_id': task.id,
                'progress': task.progress
            })

        self.status = "idle"
        self.current_task = None
        return {'status': 'completed', 'task_id': task.id}

    async def execute_task_as_team(self, task: RobotTask, team_members: List[str]):
        """Execute task as coordinated team"""
        # Assign roles to team members
        roles = self.assign_roles(team_members, task)

        # Execute coordinated task
        results = {}
        for robot_id, role in roles.items():
            # Simulate team execution
            results[robot_id] = await self.simulate_team_execution(robot_id, role, task)

        return results

    def assign_roles(self, team_members: List[str], task: RobotTask) -> Dict[str, str]:
        """Assign roles to team members based on capabilities"""
        roles = {}

        # Primary executor
        roles[team_members[0]] = "primary"

        # Support roles
        for i, robot_id in enumerate(team_members[1:], 1):
            if i <= 2:
                roles[robot_id] = "support"
            else:
                roles[robot_id] = "backup"

        return roles

    async def simulate_team_execution(self, robot_id: str, role: str, task: RobotTask):
        """Simulate team-based task execution"""
        # Simulate role-specific actions
        await asyncio.sleep(random.uniform(1.0, 3.0))
        return {'robot_id': robot_id, 'role': role, 'status': 'completed'}

class MultiAgentCoordinator:
    def __init__(self):
        self.robots: Dict[str, RobotAgent] = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks = []

    def register_robot(self, robot: RobotAgent):
        """Register a robot with the coordinator"""
        self.robots[robot.id] = robot

    def add_task(self, task: RobotTask):
        """Add task to coordination queue"""
        self.task_queue.put_nowait(task)

    async def coordinate_tasks(self):
        """Coordinate task execution among robots"""
        while True:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Find suitable robots for task
                suitable_robots = [
                    robot for robot in self.robots.values()
                    if robot.status == "idle" and self.is_robot_suitable(robot, task)
                ]

                if suitable_robots:
                    # Assign to most suitable robot
                    robot = min(
                        suitable_robots,
                        key=lambda r: len(r.neighbors)  # Prefer robots with more connections
                    )

                    # Execute with coordination
                    result = await robot.execute_task_with_coordination(task)
                    self.completed_tasks.append(result)
                else:
                    # No suitable robots, put task back
                    self.task_queue.put_nowait(task)

            except asyncio.TimeoutError:
                continue

    def is_robot_suitable(self, robot: RobotAgent, task: RobotTask) -> bool:
        """Check if robot is suitable for task"""
        required_capabilities = self.get_required_capabilities(task)
        return all(cap in robot.capabilities for cap in required_capabilities)

    def get_required_capabilities(self, task: RobotTask) -> List[str]:
        """Get required capabilities for task"""
        if "navigate" in task.name.lower():
            return ["navigation"]
        elif "grasp" in task.name.lower():
            return ["manipulation"]
        elif "detect" in task.name.lower():
            return ["vision"]
        return []
```

### 3. Learning and Adaptation
```python
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from collections import deque

class AdaptiveLearningAgent:
    def __init__(self):
        self.experience_buffer = deque(maxlen=1000)
        self.performance_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.performance_history = []

    def record_experience(self, state: Dict, action: str, reward: float, next_state: Dict):
        """Record experience for learning"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': time.time()
        }
        self.experience_buffer.append(experience)

    def extract_features(self, state: Dict) -> np.ndarray:
        """Extract features from state for learning"""
        features = []

        # Position features
        pos = state.get('position', {'x': 0, 'y': 0, 'z': 0})
        features.extend([pos['x'], pos['y'], pos['z']])

        # Orientation features
        orient = state.get('orientation', {'roll': 0, 'pitch': 0, 'yaw': 0})
        features.extend([orient['roll'], orient['pitch'], orient['yaw']])

        # Sensor features
        sensors = state.get('sensor_data', {})
        features.append(sensors.get('battery_level', 100))
        features.append(sensors.get('obstacle_distance', 10.0))
        features.append(sensors.get('temperature', 25.0))

        return np.array(features).reshape(1, -1)

    def update_model(self):
        """Update learning model with new experiences"""
        if len(self.experience_buffer) < 10:
            return

        states = []
        rewards = []

        for exp in self.experience_buffer:
            state_features = self.extract_features(exp['state'])
            states.append(state_features.flatten())
            rewards.append(exp['reward'])

        X = np.array(states)
        y = np.array(rewards)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Update model
        if not self.is_trained:
            self.performance_model.partial_fit(X_scaled, y)
            self.is_trained = True
        else:
            self.performance_model.partial_fit(X_scaled, y)

    def predict_performance(self, state: Dict) -> float:
        """Predict performance for given state"""
        if not self.is_trained:
            return 0.5  # Default prediction

        features = self.extract_features(state)
        features_scaled = self.scaler.transform(features)
        prediction = self.performance_model.predict(features_scaled)[0]

        # Ensure prediction is in reasonable range
        return max(0.0, min(1.0, prediction))

    def adapt_behavior(self, current_state: Dict, possible_actions: List[str]) -> str:
        """Adapt behavior based on learned performance"""
        best_action = possible_actions[0]
        best_predicted_performance = float('-inf')

        for action in possible_actions:
            # Simulate state after action
            simulated_state = self.simulate_state_after_action(current_state, action)
            predicted_performance = self.predict_performance(simulated_state)

            if predicted_performance > best_predicted_performance:
                best_predicted_performance = predicted_performance
                best_action = action

        return best_action

    def simulate_state_after_action(self, state: Dict, action: str) -> Dict:
        """Simulate how state would change after action"""
        # This is a simplified simulation
        # In practice, this would use more sophisticated models
        simulated_state = state.copy()

        if action == "move_forward":
            pos = simulated_state.get('position', {'x': 0, 'y': 0, 'z': 0})
            pos['x'] += 0.1  # Move forward 10cm
            simulated_state['position'] = pos
        elif action == "turn_left":
            orient = simulated_state.get('orientation', {'yaw': 0})
            orient['yaw'] += 0.1  # Turn 5.7 degrees
            simulated_state['orientation'] = orient

        return simulated_state

class ReinforcementLearningAgent:
    def __init__(self, action_space: List[str], state_size: int):
        self.action_space = action_space
        self.state_size = state_size
        self.q_table = {}  # For discrete state-action space
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate

        # For continuous state spaces, use function approximation
        self.use_function_approximation = True
        self.q_network = None  # Would be a neural network in practice

    def get_state_key(self, state: Dict) -> str:
        """Convert continuous state to discrete key for Q-table"""
        # Discretize continuous state values
        pos = state.get('position', {'x': 0, 'y': 0})
        discretized_x = round(pos['x'] * 10)  # 10cm resolution
        discretized_y = round(pos['y'] * 10)  # 10cm resolution

        return f"{discretized_x},{discretized_y}"

    def choose_action(self, state: Dict) -> str:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(self.action_space)
        else:
            # Exploit: choose best known action
            return self.get_best_action(state)

    def get_best_action(self, state: Dict) -> str:
        """Get best action for given state"""
        state_key = self.get_state_key(state)

        if state_key not in self.q_table:
            # Initialize Q-values for this state
            self.q_table[state_key] = {action: 0.0 for action in self.action_space}

        # Find action with highest Q-value
        q_values = self.q_table[state_key]
        return max(q_values, key=q_values.get)

    def update_q_value(self, state: Dict, action: str, reward: float, next_state: Dict):
        """Update Q-value using Q-learning update rule"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Initialize Q-table entries if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.action_space}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.action_space}

        # Get current Q-value
        current_q = self.q_table[state_key][action]

        # Get maximum Q-value for next state
        next_max_q = max(self.q_table[next_state_key].values())

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )

        self.q_table[state_key][action] = new_q

    def learn_from_experience(self, experiences: List[Dict]):
        """Learn from batch of experiences"""
        for exp in experiences:
            self.update_q_value(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state']
            )
```

### 4. Goal-Oriented Behavior
```python
from typing import Protocol, runtime_checkable
import heapq

@runtime_checkable
class Executable(Protocol):
    def execute(self, context: Dict) -> bool:
        """Execute the action and return success status"""
        ...

class GoToAction:
    def __init__(self, target_position: Dict[str, float]):
        self.target_position = target_position

    def execute(self, context: Dict) -> bool:
        """Navigate to target position"""
        current_pos = context.get('current_position', {'x': 0, 'y': 0, 'z': 0})

        # Calculate distance to target
        dx = self.target_position['x'] - current_pos['x']
        dy = self.target_position['y'] - current_pos['y']
        dz = self.target_position['z'] - current_pos['z']
        distance = (dx**2 + dy**2 + dz**2)**0.5

        # Check if already at target
        if distance < 0.1:  # 10cm threshold
            return True

        # Simulate navigation
        # In practice, this would call navigation stack
        context['current_position'] = self.target_position
        return True

class PickUpAction:
    def __init__(self, object_name: str):
        self.object_name = object_name

    def execute(self, context: Dict) -> bool:
        """Pick up specified object"""
        detected_objects = context.get('detected_objects', [])

        # Check if object is present
        if self.object_name not in detected_objects:
            return False

        # Simulate pick-up
        # In practice, this would call manipulation stack
        context['carrying_object'] = self.object_name
        context['detected_objects'] = [obj for obj in detected_objects if obj != self.object_name]
        return True

class GoalPlanner:
    def __init__(self):
        self.actions = {
            'go_to': GoToAction,
            'pick_up': PickUpAction,
            # Add more action types as needed
        }

    def plan_to_goal(self, goal: str, context: Dict) -> List[Executable]:
        """Plan sequence of actions to achieve goal"""
        if goal.startswith('go_to:'):
            target_str = goal[6:]  # Remove 'go_to:' prefix
            target_coords = self.parse_coordinates(target_str)
            return [GoToAction(target_coords)]

        elif goal.startswith('pick_up:'):
            object_name = goal[8:]  # Remove 'pick_up:' prefix
            return [PickUpAction(object_name)]

        elif goal.startswith('go_to_and_pick_up:'):
            parts = goal[19:].split(' and ')
            target_str = parts[0]
            object_name = parts[1]

            target_coords = self.parse_coordinates(target_str)
            return [
                GoToAction(target_coords),
                PickUpAction(object_name)
            ]

        return []

    def parse_coordinates(self, coord_str: str) -> Dict[str, float]:
        """Parse coordinate string like 'x=1.0,y=2.0,z=0.0'"""
        coords = {}
        for part in coord_str.split(','):
            key, value = part.split('=')
            coords[key.strip()] = float(value.strip())
        return coords

    def execute_plan(self, plan: List[Executable], context: Dict) -> bool:
        """Execute planned sequence of actions"""
        for i, action in enumerate(plan):
            try:
                success = action.execute(context)
                if not success:
                    logging.error(f"Action {i} failed: {action}")
                    return False
            except Exception as e:
                logging.error(f"Error executing action {i}: {e}")
                return False

        return True

class BehaviorTree:
    def __init__(self):
        self.root = None

    def set_root(self, node):
        self.root = node

    def tick(self, context: Dict) -> bool:
        """Execute one cycle of the behavior tree"""
        if self.root:
            return self.root.tick(context)
        return False

class BTNode:
    def tick(self, context: Dict) -> bool:
        raise NotImplementedError

class SequenceNode(BTNode):
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, context: Dict) -> bool:
        """Execute children in sequence until one fails"""
        for child in self.children:
            if not child.tick(context):
                return False
        return True

class SelectorNode(BTNode):
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, context: Dict) -> bool:
        """Execute children until one succeeds"""
        for child in self.children:
            if child.tick(context):
                return True
        return False

class ConditionNode(BTNode):
    def __init__(self, condition_func: Callable[[Dict], bool]):
        self.condition_func = condition_func

    def tick(self, context: Dict) -> bool:
        """Check condition"""
        return self.condition_func(context)

class ActionNode(BTNode):
    def __init__(self, action: Executable):
        self.action = action

    def tick(self, context: Dict) -> bool:
        """Execute action"""
        return self.action.execute(context)

class GoalOrientedAgent:
    def __init__(self):
        self.planner = GoalPlanner()
        self.behavior_tree = BehaviorTree()
        self.current_goals = []
        self.context = {}

    def set_goal(self, goal: str):
        """Set new goal for the agent"""
        self.current_goals.append(goal)

    def update(self):
        """Update agent state and work towards goals"""
        for goal in self.current_goals:
            plan = self.planner.plan_to_goal(goal, self.context)
            success = self.planner.execute_plan(plan, self.context)

            if success:
                self.current_goals.remove(goal)
                logging.info(f"Goal achieved: {goal}")

    def create_behavior_tree(self, goal: str):
        """Create behavior tree for specific goal"""
        # Example: Navigate to charging station when battery is low
        battery_low_condition = ConditionNode(
            lambda ctx: ctx.get('battery_level', 100) < 20
        )

        go_to_charger_action = ActionNode(
            GoToAction({'x': 0, 'y': 0, 'z': 0})  # Charger location
        )

        sequence = SequenceNode([
            battery_low_condition,
            go_to_charger_action
        ])

        self.behavior_tree.set_root(sequence)

    def run_behavior_tree(self):
        """Run the behavior tree"""
        return self.behavior_tree.tick(self.context)
```

## Best Practices for Agentic Robotics

### 1. Performance Optimization
```python
import asyncio
from functools import wraps
import time

def async_timer(func):
    """Decorator to time async functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.3f}s")
        return result
    return wrapper

class OptimizedAgent:
    def __init__(self):
        self.task_scheduler = asyncio.PriorityQueue()
        self.action_cache = {}
        self.computation_budget = 0.1  # 100ms per cycle

    @async_timer
    async def plan_with_budget(self, goal: str, time_limit: float = 0.05):
        """Plan with computation time budget"""
        start_time = time.time()

        # Use iterative deepening or other time-bounded planning
        plan = await self.iterative_plan(goal, time_limit)

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            logging.warning(f"Planning exceeded time budget: {elapsed:.3f}s")

        return plan

    async def iterative_plan(self, goal: str, time_limit: float):
        """Iterative planning with time limit"""
        start_time = time.time()

        # Start with simple plan
        plan = self.get_simple_plan(goal)

        # Refine plan if time permits
        while time.time() - start_time < time_limit * 0.8:  # Use 80% of budget
            refined_plan = self.refine_plan(plan)
            if self.is_better_plan(refined_plan, plan):
                plan = refined_plan

        return plan

    def get_simple_plan(self, goal: str):
        """Get simple, fast plan"""
        # Implement fast planning algorithm
        return [f"simple_action_for_{goal}"]

    def refine_plan(self, plan: List):
        """Refine existing plan"""
        # Implement plan refinement
        return plan

    def is_better_plan(self, plan1: List, plan2: List) -> bool:
        """Compare plan quality"""
        # Implement plan comparison logic
        return len(plan1) <= len(plan2)
```

### 2. Safety and Error Handling
```python
class SafeAgent:
    def __init__(self):
        self.safety_constraints = []
        self.emergency_protocols = []
        self.fallback_behaviors = []

    def add_safety_constraint(self, constraint: Callable[[Dict], bool]):
        """Add safety constraint that must be satisfied"""
        self.safety_constraints.append(constraint)

    def is_safe_to_execute(self, action: Executable, context: Dict) -> bool:
        """Check if action is safe to execute"""
        for constraint in self.safety_constraints:
            if not constraint(context):
                return False
        return True

    async def execute_with_safety(self, action: Executable, context: Dict):
        """Execute action with safety checks"""
        if not self.is_safe_to_execute(action, context):
            # Trigger emergency protocol
            await self.execute_emergency_protocol(context)
            return False

        try:
            result = await asyncio.wait_for(
                action.execute(context),
                timeout=10.0  # 10 second timeout
            )
            return result
        except asyncio.TimeoutError:
            logging.error("Action execution timed out")
            await self.execute_fallback_behavior(context)
            return False
        except Exception as e:
            logging.error(f"Action execution failed: {e}")
            await self.execute_fallback_behavior(context)
            return False

    async def execute_emergency_protocol(self, context: Dict):
        """Execute emergency safety protocol"""
        for protocol in self.emergency_protocols:
            await protocol(context)

    async def execute_fallback_behavior(self, context: Dict):
        """Execute fallback behavior when primary action fails"""
        for behavior in self.fallback_behaviors:
            if await behavior(context):
                break
```

## Best Practices
- Implement proper task prioritization and resource allocation
- Use behavior trees or finite state machines for complex behaviors
- Include safety constraints and emergency protocols
- Implement learning mechanisms to improve performance over time
- Use simulation for testing before real-world deployment
- Include fallback behaviors for error recovery
- Monitor agent performance and adapt strategies
- Coordinate effectively in multi-robot scenarios