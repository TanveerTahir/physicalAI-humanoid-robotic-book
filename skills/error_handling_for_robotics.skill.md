# Error Handling for Robotics

## Overview
Error handling in robotics is critical for ensuring safe, reliable, and robust operation of robotic systems. Unlike traditional software systems, robot errors can have physical consequences, making proper error detection, recovery, and safety mechanisms essential. This skill covers comprehensive error handling strategies for hardware failures, communication issues, sensor errors, and operational anomalies.

## Key Error Categories
- **Hardware Failures**: Motor, sensor, or component malfunctions
- **Communication Errors**: Network, protocol, or data transmission issues
- **Sensor Errors**: Inaccurate readings, calibration issues, or sensor fusion problems
- **Software Errors**: Algorithm failures, memory issues, or logic errors
- **Environmental Errors**: Unexpected obstacles, lighting changes, or dynamic environments
- **Safety Errors**: Conditions that could cause harm to robot, humans, or environment

## Essential Error Handling Techniques

### 1. Comprehensive Error Detection and Classification
```python
import logging
import traceback
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import time

class ErrorLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    HARDWARE = "hardware"
    COMMUNICATION = "communication"
    SENSOR = "sensor"
    SOFTWARE = "software"
    ENVIRONMENTAL = "environmental"
    SAFETY = "safety"
    USER = "user"

class ErrorCode(Enum):
    # Hardware errors
    MOTOR_FAILURE = "MTR001"
    SENSOR_MALFUNCTION = "SNS001"
    POWER_LOW = "PWR001"
    ACTUATOR_FAULT = "ACT001"

    # Communication errors
    CONNECTION_LOST = "NET001"
    MESSAGE_TIMEOUT = "NET002"
    PROTOCOL_ERROR = "NET003"
    BANDWIDTH_EXCEEDED = "NET004"

    # Sensor errors
    CALIBRATION_NEEDED = "CAL001"
    OUT_OF_RANGE = "RNG001"
    NOISE_HIGH = "NIS001"
    FUSION_FAILED = "FUS001"

    # Software errors
    ALGORITHM_FAILED = "ALG001"
    MEMORY_EXHAUSTED = "MEM001"
    INVALID_STATE = "STA001"
    TIMEOUT_EXPIRED = "TMO001"

    # Environmental errors
    OBSTACLE_DETECTED = "OBS001"
    NAVIGATION_BLOCKED = "NAV001"
    LIGHTING_CHANGED = "LGT001"
    SURFACE_UNSTABLE = "SFC001"

    # Safety errors
    COLLISION_IMMINENT = "COL001"
    SAFETY_LIMIT_EXCEEDED = "SLM001"
    EMERGENCY_STOP = "EMS001"
    TEMPERATURE_HIGH = "TMP001"

@dataclass
class RobotError:
    error_code: ErrorCode
    category: ErrorCategory
    level: ErrorLevel
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    severity_score: int = 0  # 1-10 scale

    def to_dict(self) -> Dict:
        return {
            'error_code': self.error_code.value,
            'category': self.category.value,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'severity_score': self.severity_score
        }

class ErrorDetector:
    def __init__(self):
        self.error_thresholds = {
            'temperature': 60.0,  # Celsius
            'current': 10.0,      # Amperes
            'voltage': (10.0, 14.0),  # Volts (min, max)
            'collision_force': 50.0,   # Newtons
            'position_error': 0.1      # Meters
        }
        self.anomaly_detectors = {}
        self.error_history = []

    def detect_hardware_errors(self, sensor_data: Dict) -> List[RobotError]:
        """Detect hardware-related errors"""
        errors = []

        # Check temperature
        temp = sensor_data.get('temperature', 25.0)
        if temp > self.error_thresholds['temperature']:
            errors.append(RobotError(
                error_code=ErrorCode.TEMPERATURE_HIGH,
                category=ErrorCategory.HARDWARE,
                level=ErrorLevel.WARNING,
                message=f"Temperature too high: {temp}°C",
                timestamp=datetime.now(),
                context={'temperature': temp},
                severity_score=7
            ))

        # Check current draw
        current = sensor_data.get('current_draw', 0.0)
        if current > self.error_thresholds['current']:
            errors.append(RobotError(
                error_code=ErrorCode.POWER_LOW,
                category=ErrorCategory.HARDWARE,
                level=ErrorLevel.ERROR,
                message=f"Current draw too high: {current}A",
                timestamp=datetime.now(),
                context={'current_draw': current},
                severity_score=8
            ))

        # Check voltage
        voltage = sensor_data.get('voltage', 12.0)
        min_volt, max_volt = self.error_thresholds['voltage']
        if voltage < min_volt or voltage > max_volt:
            errors.append(RobotError(
                error_code=ErrorCode.POWER_LOW,
                category=ErrorCategory.HARDWARE,
                level=ErrorLevel.WARNING,
                message=f"Voltage out of range: {voltage}V",
                timestamp=datetime.now(),
                context={'voltage': voltage, 'range': [min_volt, max_volt]},
                severity_score=6
            ))

        return errors

    def detect_communication_errors(self, connection_status: Dict) -> List[RobotError]:
        """Detect communication-related errors"""
        errors = []

        if not connection_status.get('is_connected', True):
            errors.append(RobotError(
                error_code=ErrorCode.CONNECTION_LOST,
                category=ErrorCategory.COMMUNICATION,
                level=ErrorLevel.ERROR,
                message="Connection to robot lost",
                timestamp=datetime.now(),
                context=connection_status,
                severity_score=9
            ))

        timeout_count = connection_status.get('timeout_count', 0)
        if timeout_count > 5:  # More than 5 timeouts in recent period
            errors.append(RobotError(
                error_code=ErrorCode.MESSAGE_TIMEOUT,
                category=ErrorCategory.COMMUNICATION,
                level=ErrorLevel.WARNING,
                message=f"High message timeout count: {timeout_count}",
                timestamp=datetime.now(),
                context=connection_status,
                severity_score=5
            ))

        return errors

    def detect_sensor_errors(self, sensor_readings: Dict) -> List[RobotError]:
        """Detect sensor-related errors"""
        errors = []

        # Check for sensor range errors
        for sensor_name, reading in sensor_readings.items():
            if isinstance(reading, (int, float)):
                if reading > 10000:  # Unusually high value
                    errors.append(RobotError(
                        error_code=ErrorCode.OUT_OF_RANGE,
                        category=ErrorCategory.SENSOR,
                        level=ErrorLevel.WARNING,
                        message=f"Sensor {sensor_name} reading out of range: {reading}",
                        timestamp=datetime.now(),
                        context={'sensor': sensor_name, 'value': reading},
                        severity_score=4
                    ))

        # Check for sensor fusion errors
        lidar_data = sensor_readings.get('lidar', [])
        camera_data = sensor_readings.get('camera', [])

        if lidar_data and camera_data:
            # Check consistency between sensors
            lidar_obstacles = len([d for d in lidar_data if d < 0.5])  # Close obstacles
            camera_obstacles = len(camera_data.get('objects', []))

            if abs(lidar_obstacles - camera_obstacles) > 3:  # Significant discrepancy
                errors.append(RobotError(
                    error_code=ErrorCode.FUSION_FAILED,
                    category=ErrorCategory.SENSOR,
                    level=ErrorLevel.WARNING,
                    message="Sensor fusion inconsistency detected",
                    timestamp=datetime.now(),
                    context={
                        'lidar_obstacles': lidar_obstacles,
                        'camera_obstacles': camera_obstacles
                    },
                    severity_score=5
                ))

        return errors

    def detect_navigation_errors(self, navigation_data: Dict) -> List[RobotError]:
        """Detect navigation-related errors"""
        errors = []

        # Check for navigation blockage
        obstacle_distance = navigation_data.get('min_obstacle_distance', float('inf'))
        if obstacle_distance < 0.2:  # Less than 20cm
            errors.append(RobotError(
                error_code=ErrorCode.NAVIGATION_BLOCKED,
                category=ErrorCategory.ENVIRONMENTAL,
                level=ErrorLevel.ERROR,
                message=f"Navigation blocked: obstacle at {obstacle_distance}m",
                timestamp=datetime.now(),
                context={'obstacle_distance': obstacle_distance},
                severity_score=8
            ))

        # Check for position errors
        position_error = navigation_data.get('position_error', 0.0)
        if position_error > self.error_thresholds['position_error']:
            errors.append(RobotError(
                error_code=ErrorCode.INVALID_STATE,
                category=ErrorCategory.SOFTWARE,
                level=ErrorLevel.WARNING,
                message=f"Position error too large: {position_error}m",
                timestamp=datetime.now(),
                context={'position_error': position_error},
                severity_score=6
            ))

        return errors

    def detect_safety_errors(self, safety_data: Dict) -> List[RobotError]:
        """Detect safety-related errors"""
        errors = []

        # Check for collision imminence
        collision_risk = safety_data.get('collision_risk', 0.0)
        if collision_risk > 0.8:  # 80% risk threshold
            errors.append(RobotError(
                error_code=ErrorCode.COLLISION_IMMINENT,
                category=ErrorCategory.SAFETY,
                level=ErrorLevel.CRITICAL,
                message=f"High collision risk: {collision_risk}",
                timestamp=datetime.now(),
                context={'collision_risk': collision_risk},
                severity_score=10
            ))

        # Check safety limits
        velocity = safety_data.get('velocity', 0.0)
        max_safe_velocity = safety_data.get('max_safe_velocity', 1.0)

        if abs(velocity) > max_safe_velocity:
            errors.append(RobotError(
                error_code=ErrorCode.SAFETY_LIMIT_EXCEEDED,
                category=ErrorCategory.SAFETY,
                level=ErrorLevel.ERROR,
                message=f"Velocity limit exceeded: {velocity} > {max_safe_velocity}",
                timestamp=datetime.now(),
                context={'velocity': velocity, 'limit': max_safe_velocity},
                severity_score=9
            ))

        return errors
```

### 2. Error Recovery and Fallback Mechanisms
```python
import asyncio
from typing import Union, Tuple

class ErrorRecoveryManager:
    def __init__(self):
        self.recovery_strategies = {}
        self.fallback_behaviors = {}
        self.emergency_protocols = {}
        self.retry_policies = {}
        self.state_backups = {}

    def register_recovery_strategy(self, error_code: ErrorCode, strategy_func: Callable):
        """Register a recovery strategy for specific error code"""
        self.recovery_strategies[error_code] = strategy_func

    def register_fallback_behavior(self, error_category: ErrorCategory, behavior_func: Callable):
        """Register fallback behavior for error category"""
        self.fallback_behaviors[error_category] = behavior_func

    def register_emergency_protocol(self, severity_level: int, protocol_func: Callable):
        """Register emergency protocol for severity level"""
        self.emergency_protocols[severity_level] = protocol_func

    async def handle_error(self, error: RobotError) -> bool:
        """Handle error with appropriate recovery strategy"""
        logging.log(
            self._get_logging_level(error.level),
            f"Handling error {error.error_code.value}: {error.message}"
        )

        # Log error for analysis
        self.log_error(error)

        # Execute emergency protocol if severity is high
        if error.severity_score >= 8:
            await self.execute_emergency_protocol(error)
            return False  # Emergency protocols typically stop normal operation

        # Try recovery strategy
        recovery_result = await self.attempt_recovery(error)

        # If recovery fails, try fallback behavior
        if not recovery_result:
            fallback_result = await self.execute_fallback_behavior(error)
            return fallback_result

        return True

    async def attempt_recovery(self, error: RobotError) -> bool:
        """Attempt to recover from error using registered strategy"""
        if error.error_code in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error.error_code]
                return await recovery_func(error)
            except Exception as e:
                logging.error(f"Recovery failed: {e}")
                return False

        # Default recovery based on category
        return await self.default_recovery(error)

    async def default_recovery(self, error: RobotError) -> bool:
        """Default recovery based on error category"""
        if error.category == ErrorCategory.HARDWARE:
            return await self.hardware_recovery(error)
        elif error.category == ErrorCategory.COMMUNICATION:
            return await self.communication_recovery(error)
        elif error.category == ErrorCategory.SENSOR:
            return await self.sensor_recovery(error)
        elif error.category == ErrorCategory.SOFTWARE:
            return await self.software_recovery(error)
        else:
            return False  # No default recovery for other categories

    async def hardware_recovery(self, error: RobotError) -> bool:
        """Recovery strategy for hardware errors"""
        if error.error_code == ErrorCode.MOTOR_FAILURE:
            # Try to reset motor controller
            logging.info("Attempting motor reset...")
            # Implementation would call motor reset API
            return True
        elif error.error_code == ErrorCode.TEMPERATURE_HIGH:
            # Reduce motor power to cool down
            logging.info("Reducing power to cool down...")
            # Implementation would reduce power consumption
            return True
        elif error.error_code == ErrorCode.POWER_LOW:
            # Navigate to charging station
            logging.info("Navigating to charging station...")
            # Implementation would call navigation to charging station
            return True

        return False

    async def communication_recovery(self, error: RobotError) -> bool:
        """Recovery strategy for communication errors"""
        if error.error_code == ErrorCode.CONNECTION_LOST:
            # Attempt reconnection
            logging.info("Attempting to reconnect...")
            # Implementation would try to reestablish connection
            return True
        elif error.error_code == ErrorCode.MESSAGE_TIMEOUT:
            # Increase timeout values temporarily
            logging.info("Increasing timeout values...")
            # Implementation would adjust communication parameters
            return True

        return False

    async def sensor_recovery(self, error: RobotError) -> bool:
        """Recovery strategy for sensor errors"""
        if error.error_code == ErrorCode.CALIBRATION_NEEDED:
            # Initiate recalibration
            logging.info("Initiating sensor recalibration...")
            # Implementation would call calibration routine
            return True
        elif error.error_code == ErrorCode.OUT_OF_RANGE:
            # Switch to backup sensor if available
            logging.info("Switching to backup sensor...")
            # Implementation would switch sensor sources
            return True

        return False

    async def software_recovery(self, error: RobotError) -> bool:
        """Recovery strategy for software errors"""
        if error.error_code == ErrorCode.MEMORY_EXHAUSTED:
            # Clear caches and reduce computational load
            logging.info("Clearing caches and reducing load...")
            # Implementation would free memory and reduce processing
            return True
        elif error.error_code == ErrorCode.TIMEOUT_EXPIRED:
            # Increase timeout or simplify algorithm
            logging.info("Adjusting timeout parameters...")
            # Implementation would adjust timing parameters
            return True

        return False

    async def execute_fallback_behavior(self, error: RobotError) -> bool:
        """Execute fallback behavior for error category"""
        if error.category in self.fallback_behaviors:
            try:
                fallback_func = self.fallback_behaviors[error.category]
                return await fallback_func(error)
            except Exception as e:
                logging.error(f"Fallback behavior failed: {e}")
                return False

        # Default fallback behavior
        return await self.default_fallback(error)

    async def default_fallback(self, error: RobotError) -> bool:
        """Default fallback behavior"""
        if error.category in [ErrorCategory.SAFETY, ErrorCategory.ENVIRONMENTAL]:
            # Stop all movement for safety
            logging.info("Stopping all movement for safety")
            # Implementation would stop all actuators
            return True
        else:
            # Continue with reduced functionality
            logging.info("Continuing with reduced functionality")
            return True

    async def execute_emergency_protocol(self, error: RobotError):
        """Execute emergency protocol for high-severity errors"""
        # Log emergency
        logging.critical(f"EMERGENCY: {error.message}")

        # Stop all operations immediately
        await self.emergency_stop()

        # Save current state for analysis
        await self.save_emergency_state(error)

        # Alert operators
        await self.alert_operators(error)

    async def emergency_stop(self):
        """Immediate emergency stop of all robot functions"""
        logging.critical("EMERGENCY STOP ACTIVATED")
        # Implementation would immediately stop all motors and actuators
        # This should be implemented at the hardware level if possible

    async def save_emergency_state(self, error: RobotError):
        """Save current state for post-emergency analysis"""
        state_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'error': error.to_dict(),
            'robot_state': {},  # Would capture current robot state
            'sensor_data': {},  # Would capture current sensor readings
            'last_commands': [],  # Would capture recent commands
        }

        # Save to persistent storage
        # Implementation would save to file or database
        logging.info("Emergency state saved for analysis")

    async def alert_operators(self, error: RobotError):
        """Alert human operators about emergency"""
        # Send notification via multiple channels
        # Implementation would use email, SMS, dashboard, etc.
        logging.info(f"Alert sent to operators: {error.message}")

    def log_error(self, error: RobotError):
        """Log error for analysis and monitoring"""
        self.error_history.append(error)
        # Also log to external monitoring system if available
        logging.log(
            self._get_logging_level(error.level),
            f"ERROR: {error.error_code.value} - {error.message}",
            extra={'error_context': error.context}
        )

    def _get_logging_level(self, error_level: ErrorLevel):
        """Convert error level to logging level"""
        level_map = {
            ErrorLevel.DEBUG: logging.DEBUG,
            ErrorLevel.INFO: logging.INFO,
            ErrorLevel.WARNING: logging.WARNING,
            ErrorLevel.ERROR: logging.ERROR,
            ErrorLevel.CRITICAL: logging.CRITICAL
        }
        return level_map[error_level]
```

### 3. Retry Logic and Circuit Breaker Patterns
```python
import random
from functools import wraps

class RetryManager:
    def __init__(self):
        self.default_retry_policy = {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 30.0,
            'backoff_factor': 2.0,
            'jitter': True,
            'retryable_errors': [
                ErrorCode.CONNECTION_LOST,
                ErrorCode.MESSAGE_TIMEOUT,
                ErrorCode.POWER_LOW  # Sometimes temporary
            ]
        }

    def should_retry(self, error: RobotError, attempt: int) -> bool:
        """Determine if error should be retried"""
        if attempt >= self.default_retry_policy['max_attempts']:
            return False

        if error.error_code not in self.default_retry_policy['retryable_errors']:
            return False

        # Don't retry critical safety errors
        if error.category == ErrorCategory.SAFETY and error.severity_score > 7:
            return False

        return True

    async def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before retry with exponential backoff"""
        base_delay = self.default_retry_policy['base_delay']
        backoff_factor = self.default_retry_policy['backoff_factor']
        max_delay = self.default_retry_policy['max_delay']

        delay = base_delay * (backoff_factor ** (attempt - 1))

        # Apply jitter to prevent thundering herd
        if self.default_retry_policy['jitter']:
            delay *= random.uniform(0.5, 1.5)

        return min(delay, max_delay)

    async def retry_with_policy(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry policy"""
        last_error = None

        for attempt in range(1, self.default_retry_policy['max_attempts'] + 1):
            try:
                result = await operation(*args, **kwargs)
                return result
            except Exception as e:
                # Convert to RobotError if not already
                if not isinstance(e, RobotError):
                    error = RobotError(
                        error_code=ErrorCode.SOFTWARE,
                        category=ErrorCategory.SOFTWARE,
                        level=ErrorLevel.ERROR,
                        message=str(e),
                        timestamp=datetime.now(),
                        context={'operation': operation.__name__, 'attempt': attempt},
                        stack_trace=traceback.format_exc()
                    )
                else:
                    error = e

                last_error = error

                # Check if we should retry
                if not self.should_retry(error, attempt):
                    break

                # Wait before retry
                if attempt < self.default_retry_policy['max_attempts']:
                    delay = await self.calculate_delay(attempt)
                    logging.warning(
                        f"Attempt {attempt} failed, retrying in {delay:.2f}s: {error.message}"
                    )
                    await asyncio.sleep(delay)

        # All attempts failed
        raise last_error

def retry_on_error(retry_manager: RetryManager = None):
    """Decorator for automatic retry on errors"""
    if retry_manager is None:
        retry_manager = RetryManager()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_manager.retry_with_policy(func, *args, **kwargs)
        return wrapper
    return decorator

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """Call operation with circuit breaker protection"""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")

        if self.state == 'HALF_OPEN':
            try:
                result = await operation(*args, **kwargs)
                self._on_success()
                return result
            except Exception:
                self._on_failure()
                raise

        if self.state == 'CLOSED':
            try:
                result = await operation(*args, **kwargs)
                return result
            except Exception as e:
                self._on_failure()
                raise e

    def _on_success(self):
        """Called when operation succeeds"""
        self.failure_count = 0
        self.state = 'CLOSED'

    def _on_failure(self):
        """Called when operation fails"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

    def reset(self):
        """Reset circuit breaker to CLOSED state"""
        self.failure_count = 0
        self.state = 'CLOSED'
        self.last_failure_time = None

class FaultTolerantRobotInterface:
    def __init__(self):
        self.retry_manager = RetryManager()
        self.circuit_breakers = {}
        self.health_check_interval = 30.0  # seconds

    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation"""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker()
        return self.circuit_breakers[operation_name]

    @retry_on_error()
    async def send_command_with_retry(self, command: str, robot_id: str) -> Dict:
        """Send command to robot with retry logic"""
        cb = self.get_circuit_breaker(f"send_command_{robot_id}")
        return await cb.call(self._send_command, command, robot_id)

    async def _send_command(self, command: str, robot_id: str) -> Dict:
        """Internal method to send command"""
        # Implementation would send actual command to robot
        # This is a placeholder
        if random.random() < 0.1:  # Simulate 10% failure rate
            raise Exception(f"Command failed for robot {robot_id}")
        return {"status": "success", "command": command}

    async def monitor_robot_health(self, robot_id: str):
        """Monitor robot health and update circuit breakers"""
        while True:
            try:
                # Perform health check
                health_status = await self.check_robot_health(robot_id)

                # Update circuit breaker based on health
                cb = self.get_circuit_breaker(f"robot_{robot_id}")
                if health_status['status'] == 'healthy':
                    cb.reset()
                else:
                    # Don't explicitly fail, just note the health status
                    pass

            except Exception as e:
                logging.error(f"Health check failed for robot {robot_id}: {e}")

            await asyncio.sleep(self.health_check_interval)

    async def check_robot_health(self, robot_id: str) -> Dict:
        """Check robot health status"""
        # Implementation would check actual robot health
        # This is a placeholder
        return {
            "status": "healthy" if random.random() > 0.05 else "unhealthy",  # 5% failure rate
            "metrics": {
                "cpu_usage": random.uniform(10, 80),
                "memory_usage": random.uniform(20, 70),
                "temperature": random.uniform(25, 45),
                "battery_level": random.uniform(20, 100)
            }
        }
```

### 4. Monitoring and Alerting Systems
```python
import json
from datetime import timedelta
from collections import defaultdict, deque

class RobotMonitoringSystem:
    def __init__(self):
        self.error_stats = defaultdict(deque)
        self.performance_metrics = {}
        self.alert_rules = []
        self.alert_history = deque(maxlen=1000)
        self.metric_thresholds = {
            'error_rate': 0.05,  # 5% error rate threshold
            'response_time': 2.0,  # 2 second response time threshold
            'memory_usage': 80.0,  # 80% memory usage threshold
            'cpu_usage': 85.0,     # 85% CPU usage threshold
            'temperature': 55.0    # 55°C temperature threshold
        }

    def add_alert_rule(self, condition: Callable, action: Callable, description: str = ""):
        """Add alert rule with condition and action"""
        self.alert_rules.append({
            'condition': condition,
            'action': action,
            'description': description,
            'last_triggered': None
        })

    def record_error(self, error: RobotError):
        """Record error for statistical analysis"""
        self.error_stats[error.category.value].append(error)
        # Keep only recent errors (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        while (self.error_stats[error.category.value] and
               self.error_stats[error.category.value][0].timestamp < cutoff_time):
            self.error_stats[error.category.value].popleft()

    def record_metric(self, robot_id: str, metric_name: str, value: float):
        """Record performance metric"""
        if robot_id not in self.performance_metrics:
            self.performance_metrics[robot_id] = {}

        if metric_name not in self.performance_metrics[robot_id]:
            self.performance_metrics[robot_id][metric_name] = deque(maxlen=100)

        self.performance_metrics[robot_id][metric_name].append({
            'value': value,
            'timestamp': datetime.now()
        })

    def calculate_error_rate(self, category: ErrorCategory, time_window_minutes: int = 5) -> float:
        """Calculate error rate for category in time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_errors = [
            e for e in self.error_stats[category.value]
            if e.timestamp >= cutoff_time
        ]

        total_operations = self.get_total_operations_in_window(time_window_minutes)
        if total_operations == 0:
            return 0.0

        return len(recent_errors) / total_operations

    def get_total_operations_in_window(self, time_window_minutes: int) -> int:
        """Get total operations in time window (placeholder implementation)"""
        # In a real system, this would track all operations
        return 100  # Placeholder

    def check_alerts(self):
        """Check all alert rules and trigger actions if conditions are met"""
        for rule in self.alert_rules:
            try:
                if rule['condition']():
                    if (rule['last_triggered'] is None or
                        datetime.now() - rule['last_triggered'] > timedelta(minutes=1)):
                        # Don't trigger too frequently
                        asyncio.create_task(rule['action']())
                        rule['last_triggered'] = datetime.now()

                        # Log alert
                        alert_entry = {
                            'timestamp': datetime.now(),
                            'rule_description': rule['description'],
                            'triggered': True
                        }
                        self.alert_history.append(alert_entry)
            except Exception as e:
                logging.error(f"Alert rule evaluation failed: {e}")

    def setup_default_alerts(self):
        """Setup default monitoring alerts"""

        # High error rate alert
        def high_error_rate_condition():
            return self.calculate_error_rate(ErrorCategory.SOFTWARE) > 0.1  # 10%

        async def high_error_rate_action():
            logging.critical("HIGH ERROR RATE DETECTED - SYSTEM UNSTABLE")
            # Implementation would send alerts, reduce load, etc.

        self.add_alert_rule(
            high_error_rate_condition,
            high_error_rate_action,
            "High software error rate"
        )

        # High temperature alert
        def high_temperature_condition():
            for robot_id, metrics in self.performance_metrics.items():
                if 'temperature' in metrics:
                    recent_temp = metrics['temperature'][-1]['value'] if metrics['temperature'] else 0
                    if recent_temp > self.metric_thresholds['temperature']:
                        return True
            return False

        async def high_temperature_action():
            logging.critical("HIGH TEMPERATURE DETECTED - REDUCING POWER")
            # Implementation would reduce power, activate cooling, etc.

        self.add_alert_rule(
            high_temperature_condition,
            high_temperature_action,
            "High temperature detected"
        )

        # Memory usage alert
        def high_memory_condition():
            for robot_id, metrics in self.performance_metrics.items():
                if 'memory_usage' in metrics:
                    recent_memory = metrics['memory_usage'][-1]['value'] if metrics['memory_usage'] else 0
                    if recent_memory > self.metric_thresholds['memory_usage']:
                        return True
            return False

        async def high_memory_action():
            logging.warning("HIGH MEMORY USAGE - CLEARING CACHES")
            # Implementation would clear caches, reduce processing, etc.

        self.add_alert_rule(
            high_memory_condition,
            high_memory_action,
            "High memory usage"
        )

    def get_system_health_report(self) -> Dict:
        """Generate comprehensive system health report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'error_summary': {},
            'performance_metrics': {},
            'active_alerts': len([a for a in self.alert_history if a['triggered']]),
            'last_alerts': list(self.alert_history)[-5:]  # Last 5 alerts
        }

        # Summarize errors by category
        for category in ErrorCategory:
            error_rate = self.calculate_error_rate(category)
            report['error_summary'][category.value] = {
                'rate': error_rate,
                'count': len(self.error_stats[category.value]),
                'status': 'warning' if error_rate > 0.05 else 'normal'
            }

        # Include performance metrics
        for robot_id, metrics in self.performance_metrics.items():
            robot_metrics = {}
            for metric_name, values in metrics.items():
                if values:  # If there are values
                    recent_value = values[-1]['value']
                    robot_metrics[metric_name] = {
                        'current': recent_value,
                        'status': 'normal' if recent_value < self.metric_thresholds.get(metric_name, 100) else 'warning'
                    }
            report['performance_metrics'][robot_id] = robot_metrics

        # Determine overall status
        critical_errors = any(
            summary['status'] == 'warning'
            for summary in report['error_summary'].values()
        )
        if critical_errors:
            report['overall_status'] = 'warning'

        return report

class DashboardInterface:
    def __init__(self, monitoring_system: RobotMonitoringSystem):
        self.monitoring_system = monitoring_system
        self.dashboard_data = {}

    def update_dashboard(self):
        """Update dashboard with current system status"""
        health_report = self.monitoring_system.get_system_health_report()

        self.dashboard_data = {
            'last_update': datetime.now().isoformat(),
            'health_report': health_report,
            'error_timeline': self.get_error_timeline(),
            'robot_status': self.get_robot_status(),
            'system_metrics': self.get_system_metrics()
        }

    def get_error_timeline(self) -> List[Dict]:
        """Get timeline of recent errors"""
        timeline = []
        for category, errors in self.monitoring_system.error_stats.items():
            for error in errors[-10:]:  # Last 10 errors per category
                timeline.append({
                    'timestamp': error.timestamp.isoformat(),
                    'category': error.category.value,
                    'message': error.message,
                    'severity': error.severity_score
                })

        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        return timeline[-50:]  # Return last 50 errors

    def get_robot_status(self) -> Dict:
        """Get status of all monitored robots"""
        status = {}
        for robot_id, metrics in self.monitoring_system.performance_metrics.items():
            robot_status = 'normal'

            for metric_name, values in metrics.items():
                if values:
                    recent_value = values[-1]['value']
                    threshold = self.monitoring_system.metric_thresholds.get(metric_name, 100)
                    if recent_value > threshold:
                        robot_status = 'warning'
                        break

            status[robot_id] = {
                'status': robot_status,
                'metrics': {name: values[-1]['value'] if values else 0
                           for name, values in metrics.items()}
            }

        return status

    def get_system_metrics(self) -> Dict:
        """Get overall system metrics"""
        total_errors = sum(len(errors) for errors in self.monitoring_system.error_stats.values())
        active_robots = len(self.monitoring_system.performance_metrics)

        return {
            'total_errors': total_errors,
            'active_robots': active_robots,
            'error_rate': total_errors / max(1, active_robots * 100),  # Simplified calculation
            'uptime': self.calculate_uptime()
        }

    def calculate_uptime(self) -> float:
        """Calculate system uptime"""
        # Simplified uptime calculation
        # In practice, this would track system start time
        return 0.99  # Placeholder for 99% uptime
```

## Best Practices for Robotics Error Handling

### 1. Defensive Programming Techniques
```python
def validate_robot_command(command: Dict) -> Tuple[bool, List[str]]:
    """Validate robot command before execution"""
    errors = []

    # Check required fields
    required_fields = ['robot_id', 'command_type', 'parameters']
    for field in required_fields:
        if field not in command:
            errors.append(f"Missing required field: {field}")

    # Validate robot ID format
    if 'robot_id' in command and not isinstance(command['robot_id'], str):
        errors.append("Robot ID must be a string")

    # Validate command type
    allowed_commands = ['move', 'rotate', 'grasp', 'navigate', 'stop']
    if 'command_type' in command and command['command_type'] not in allowed_commands:
        errors.append(f"Invalid command type: {command['command_type']}")

    # Validate parameters
    if 'parameters' in command and not isinstance(command['parameters'], dict):
        errors.append("Parameters must be a dictionary")

    return len(errors) == 0, errors

def safe_execute_with_validation(func: Callable, *args, **kwargs) -> Any:
    """Execute function with input validation and error handling"""
    try:
        # Validate inputs if validation function exists
        validate_func_name = f"validate_{func.__name__}"
        if validate_func_name in globals():
            is_valid, validation_errors = globals()[validate_func_name](*args, **kwargs)
            if not is_valid:
                raise ValueError(f"Validation failed: {validation_errors}")

        # Execute function with timeout
        result = asyncio.wait_for(func(*args, **kwargs), timeout=30.0)
        return result

    except asyncio.TimeoutError:
        logging.error(f"Function {func.__name__} timed out")
        raise RobotError(
            error_code=ErrorCode.TIMEOUT_EXPIRED,
            category=ErrorCategory.SOFTWARE,
            level=ErrorLevel.ERROR,
            message=f"Function {func.__name__} timed out",
            timestamp=datetime.now(),
            context={'function': func.__name__},
            severity_score=7
        )
    except Exception as e:
        logging.error(f"Function {func.__name__} failed: {e}")
        raise RobotError(
            error_code=ErrorCode.SOFTWARE,
            category=ErrorCategory.SOFTWARE,
            level=ErrorLevel.ERROR,
            message=str(e),
            timestamp=datetime.now(),
            context={'function': func.__name__, 'exception': str(e)},
            severity_score=5
        )
```

### 2. Graceful Degradation Strategies
```python
class GracefulDegradationManager:
    def __init__(self):
        self.fallback_levels = {
            'full_functionality': ['navigation', 'manipulation', 'vision', 'communication'],
            'reduced_functionality': ['navigation', 'communication'],
            'safe_mode': ['communication'],
            'emergency_mode': []
        }
        self.current_mode = 'full_functionality'

    async def degrade_gracefully(self, error: RobotError):
        """Degrade system functionality based on error severity"""
        if error.severity_score >= 9:
            # Critical error - go to emergency mode
            await self.set_mode('emergency_mode')
        elif error.severity_score >= 7:
            # High severity - go to safe mode
            await self.set_mode('safe_mode')
        elif error.severity_score >= 5:
            # Medium severity - go to reduced functionality
            await self.set_mode('reduced_functionality')
        # For lower severity, maintain current mode

    async def set_mode(self, mode: str):
        """Set system to specific operational mode"""
        if mode != self.current_mode:
            logging.info(f"Switching from {self.current_mode} to {mode}")

            # Disable functionality based on current mode
            if self.current_mode == 'full_functionality':
                await self.disable_manipulation()
                await self.disable_vision_systems()
            elif self.current_mode == 'reduced_functionality':
                await self.disable_navigation()

            # Enable functionality based on new mode
            if mode == 'reduced_functionality':
                await self.enable_navigation()
                await self.enable_communication()
            elif mode == 'safe_mode':
                await self.enable_communication()
            elif mode == 'emergency_mode':
                await self.emergency_stop()

            self.current_mode = mode

    async def enable_navigation(self):
        """Enable navigation systems"""
        # Implementation would enable navigation stack
        pass

    async def disable_navigation(self):
        """Disable navigation systems"""
        # Implementation would disable navigation stack
        pass

    async def enable_manipulation(self):
        """Enable manipulation systems"""
        # Implementation would enable manipulation stack
        pass

    async def disable_manipulation(self):
        """Disable manipulation systems"""
        # Implementation would disable manipulation stack
        pass

    async def enable_vision_systems(self):
        """Enable vision systems"""
        # Implementation would enable vision processing
        pass

    async def disable_vision_systems(self):
        """Disable vision systems"""
        # Implementation would disable vision processing
        pass

    async def enable_communication(self):
        """Enable communication systems"""
        # Implementation would ensure communication is active
        pass

    async def emergency_stop(self):
        """Emergency stop all systems"""
        # Implementation would stop all actuators and processes
        pass
```

## Best Practices
- Implement layered error handling (component, system, and application level)
- Use circuit breakers to prevent cascading failures
- Implement comprehensive logging and monitoring
- Design graceful degradation strategies for system failures
- Use timeouts to prevent indefinite waiting
- Implement proper state management during error recovery
- Test error scenarios thoroughly in simulation before deployment
- Maintain emergency stop capabilities at the hardware level
- Use redundant systems for critical functions
- Regularly analyze error patterns to improve system reliability