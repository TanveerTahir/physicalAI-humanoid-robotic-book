# Hardware Integration for Robotics

## Overview
Hardware integration is the process of connecting, configuring, and operating physical components in a robotic system. This skill covers interfacing with sensors, actuators, microcontrollers, and communication protocols essential for building functional robots.

## Key Hardware Components
- **Sensors**: IMU, LIDAR, cameras, encoders, force/torque sensors
- **Actuators**: Servos, stepper motors, DC motors, pneumatic systems
- **Computing Platforms**: Single-board computers, microcontrollers, GPUs
- **Communication Interfaces**: I2C, SPI, UART, CAN, Ethernet
- **Power Systems**: Batteries, voltage regulators, power distribution

## Essential Hardware Integration Techniques

### 1. Sensor Integration
```python
import smbus2
import spidev
import serial
import time
import struct
import threading
from collections import deque

class SensorInterface:
    def __init__(self):
        self.i2c_bus = None
        self.spi_device = None
        self.serial_port = None
        self.sensor_data = {}
        self.data_lock = threading.Lock()

    def initialize_i2c(self, bus_number=1):
        """Initialize I2C communication"""
        try:
            self.i2c_bus = smbus2.SMBus(bus_number)
            print(f"I2C bus {bus_number} initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize I2C: {e}")
            return False

    def read_i2c_sensor(self, address, register, length=1):
        """Read data from I2C sensor"""
        if self.i2c_bus is None:
            raise Exception("I2C bus not initialized")

        try:
            with self.data_lock:
                if length == 1:
                    data = self.i2c_bus.read_byte_data(address, register)
                else:
                    data = self.i2c_bus.read_i2c_block_data(address, register, length)
            return data
        except Exception as e:
            print(f"I2C read error: {e}")
            return None

    def write_i2c_register(self, address, register, value):
        """Write to I2C sensor register"""
        if self.i2c_bus is None:
            raise Exception("I2C bus not initialized")

        try:
            with self.data_lock:
                self.i2c_bus.write_byte_data(address, register, value)
            return True
        except Exception as e:
            print(f"I2C write error: {e}")
            return False

    def initialize_spi(self, device="/dev/spidev0.0", max_speed_hz=500000):
        """Initialize SPI communication"""
        try:
            self.spi_device = spidev.SpiDev()
            self.spi_device.open(0, 0)  # Bus 0, Device 0
            self.spi_device.max_speed_hz = max_speed_hz
            self.spi_device.mode = 0
            print(f"SPI device {device} initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize SPI: {e}")
            return False

    def read_spi_sensor(self, device_address, command_bytes):
        """Read data from SPI sensor"""
        if self.spi_device is None:
            raise Exception("SPI device not initialized")

        try:
            with self.data_lock:
                # Send command and receive response
                response = self.spi_device.xfer2(command_bytes)
            return response
        except Exception as e:
            print(f"SPI read error: {e}")
            return None

    def initialize_serial(self, port="/dev/ttyUSB0", baudrate=115200):
        """Initialize serial communication"""
        try:
            self.serial_port = serial.Serial(port, baudrate=baudrate, timeout=1)
            print(f"Serial port {port} initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize serial: {e}")
            return False

    def read_serial_data(self):
        """Read data from serial port"""
        if self.serial_port is None:
            raise Exception("Serial port not initialized")

        try:
            if self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8').strip()
                return line
        except Exception as e:
            print(f"Serial read error: {e}")
            return None

    def parse_imu_data(self, raw_data):
        """Parse raw IMU data (example for MPU6050)"""
        # MPU6050: 14 bytes - 2 bytes each for accel_x, accel_y, accel_z, temp, gyro_x, gyro_y, gyro_z
        if len(raw_data) < 14:
            return None

        # Unpack raw data as signed 16-bit integers
        accel_x, accel_y, accel_z, temp, gyro_x, gyro_y, gyro_z = struct.unpack(
            '>hhhhh', raw_data[:14]
        )

        # Convert to meaningful units
        accelerometer = {
            'x': accel_x / 16384.0,  # Assuming ±2g range
            'y': accel_y / 16384.0,
            'z': accel_z / 16384.0
        }

        temperature = temp / 340.0 + 36.53  # Convert to Celsius

        gyroscope = {
            'x': gyro_x / 131.0,  # Assuming ±250°/s range
            'y': gyro_y / 131.0,
            'z': gyro_z / 131.0
        }

        return {
            'accelerometer': accelerometer,
            'gyroscope': gyroscope,
            'temperature': temperature
        }

class IMUSensor:
    def __init__(self, i2c_address=0x68):
        self.i2c_address = i2c_address
        self.interface = SensorInterface()

    def initialize(self):
        """Initialize IMU sensor"""
        if not self.interface.initialize_i2c():
            return False

        # Verify device ID (MPU6050 should return 0x68)
        device_id = self.interface.read_i2c_sensor(self.i2c_address, 0x75, 1)
        if device_id != 0x68:
            print(f"Invalid device ID: {device_id}, expected: 0x68")
            return False

        # Configure IMU
        # Wake up device
        self.interface.write_i2c_register(self.i2c_address, 0x6B, 0x00)
        # Set digital low pass filter
        self.interface.write_i2c_register(self.i2c_address, 0x1A, 0x03)
        # Set accelerometer range to ±2g
        self.interface.write_i2c_register(self.i2c_address, 0x1C, 0x00)
        # Set gyroscope range to ±250°/s
        self.interface.write_i2c_register(self.i2c_address, 0x1B, 0x00)

        return True

    def read_sensor_data(self):
        """Read IMU sensor data"""
        raw_data = self.interface.read_i2c_sensor(self.i2c_address, 0x3B, 14)
        if raw_data is not None:
            return self.interface.parse_imu_data(raw_data)
        return None
```

### 2. Actuator Control
```python
import RPi.GPIO as GPIO
import pigpio
import time
import threading

class ActuatorController:
    def __init__(self):
        self.pi = pigpio.pi()  # Use pigpio for hardware PWM
        self.servo_pins = {}
        self.motor_pins = {}
        self.pwm_channels = {}
        self.is_initialized = False

    def initialize_pwm(self):
        """Initialize PWM controller"""
        if not self.pi.connected:
            print("Failed to connect to pigpio daemon")
            return False

        self.is_initialized = True
        return True

    def setup_servo(self, pin, min_pulse=500, max_pulse=2500):
        """Setup servo motor on specified pin"""
        if not self.is_initialized:
            raise Exception("PWM controller not initialized")

        self.servo_pins[pin] = {'min_pulse': min_pulse, 'max_pulse': max_pulse}
        # Set servo to neutral position (1500µs)
        self.pi.set_servo_pulsewidth(pin, 1500)
        print(f"Servo configured on pin {pin}")

    def set_servo_position(self, pin, angle):
        """Set servo position (0-180 degrees)"""
        if pin not in self.servo_pins:
            raise Exception(f"Servo on pin {pin} not configured")

        # Convert angle to pulse width (500-2500 µs)
        min_pulse = self.servo_pins[pin]['min_pulse']
        max_pulse = self.servo_pins[pin]['max_pulse']

        pulse_width = min_pulse + (max_pulse - min_pulse) * (angle / 180.0)
        self.pi.set_servo_pulsewidth(pin, int(pulse_width))

    def setup_dc_motor(self, pin_pwm, pin_dir1, pin_dir2):
        """Setup DC motor with direction control"""
        if not self.is_initialized:
            raise Exception("PWM controller not initialized")

        self.motor_pins[pin_pwm] = {
            'pwm': pin_pwm,
            'dir1': pin_dir1,
            'dir2': pin_dir2
        }

        # Set up GPIO pins
        self.pi.set_mode(pin_dir1, pigpio.OUTPUT)
        self.pi.set_mode(pin_dir2, pigpio.OUTPUT)

        print(f"DC motor configured on pins PWM:{pin_pwm}, DIR1:{pin_dir1}, DIR2:{pin_dir2}")

    def set_motor_speed(self, pin_pwm, speed):
        """Set DC motor speed and direction (-100 to 100)"""
        if pin_pwm not in self.motor_pins:
            raise Exception(f"Motor on pin {pin_pwm} not configured")

        motor_config = self.motor_pins[pin_pwm]

        # Set direction based on sign of speed
        if speed > 0:
            self.pi.write(motor_config['dir1'], 1)
            self.pi.write(motor_config['dir2'], 0)
        elif speed < 0:
            self.pi.write(motor_config['dir1'], 0)
            self.pi.write(motor_config['dir2'], 1)
        else:
            self.pi.write(motor_config['dir1'], 0)
            self.pi.write(motor_config['dir2'], 0)

        # Set speed (0-100%)
        pwm_duty = min(abs(speed), 100) * 10000 / 100  # Convert to 0-10000 range
        self.pi.set_PWM_dutycycle(pin_pwm, int(pwm_duty))

    def setup_stepper_motor(self, pin_step, pin_dir, pin_enable=None):
        """Setup stepper motor control"""
        stepper_config = {
            'step': pin_step,
            'dir': pin_dir,
            'enable': pin_enable
        }

        # Set up GPIO pins
        self.pi.set_mode(pin_step, pigpio.OUTPUT)
        self.pi.set_mode(pin_dir, pigpio.OUTPUT)

        if pin_enable:
            self.pi.set_mode(pin_enable, pigpio.OUTPUT)
            self.pi.write(pin_enable, 0)  # Enable motor

        return stepper_config

    def step_motor(self, stepper_config, steps, direction=1, delay=0.001):
        """Step the stepper motor"""
        self.pi.write(stepper_config['dir'], direction)

        for _ in range(abs(steps)):
            self.pi.write(stepper_config['step'], 1)
            time.sleep(delay)
            self.pi.write(stepper_config['step'], 0)
            time.sleep(delay)

class RobotArmController:
    def __init__(self):
        self.actuator_controller = ActuatorController()
        self.joint_count = 6
        self.joint_pins = [12, 13, 14, 15, 16, 17]  # Example GPIO pins

    def initialize(self):
        """Initialize robot arm actuators"""
        if not self.actuator_controller.initialize_pwm():
            return False

        # Setup servo motors for each joint
        for i, pin in enumerate(self.joint_pins):
            self.actuator_controller.setup_servo(pin)

        return True

    def move_joint(self, joint_index, angle):
        """Move specific joint to angle"""
        if joint_index < 0 or joint_index >= self.joint_count:
            raise Exception(f"Invalid joint index: {joint_index}")

        pin = self.joint_pins[joint_index]
        self.actuator_controller.set_servo_position(pin, angle)

    def move_to_pose(self, joint_angles):
        """Move all joints to specified angles"""
        if len(joint_angles) != self.joint_count:
            raise Exception(f"Expected {self.joint_count} joint angles, got {len(joint_angles)}")

        for i, angle in enumerate(joint_angles):
            self.move_joint(i, angle)
```

### 3. Communication Protocols
```python
import can
import socket
import struct
import threading

class CANBusInterface:
    def __init__(self, channel='can0', bustype='socketcan'):
        self.channel = channel
        self.bustype = bustype
        self.bus = None
        self.listeners = {}
        self.is_running = False

    def initialize(self):
        """Initialize CAN bus interface"""
        try:
            self.bus = can.Bus(channel=self.channel, bustype=self.bustype)
            self.is_running = True
            print(f"CAN bus {self.channel} initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize CAN bus: {e}")
            return False

    def send_message(self, arbitration_id, data):
        """Send CAN message"""
        if self.bus is None:
            raise Exception("CAN bus not initialized")

        msg = can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True)
        try:
            self.bus.send(msg)
            return True
        except Exception as e:
            print(f"Failed to send CAN message: {e}")
            return False

    def add_listener(self, arbitration_id, callback):
        """Add listener for specific CAN message ID"""
        if arbitration_id not in self.listeners:
            self.listeners[arbitration_id] = []
        self.listeners[arbitration_id].append(callback)

    def start_listening(self):
        """Start listening for CAN messages in background thread"""
        if not self.is_running:
            return

        def listen_thread():
            for msg in self.bus:
                if msg.arbitration_id in self.listeners:
                    for callback in self.listeners[msg.arbitration_id]:
                        try:
                            callback(msg)
                        except Exception as e:
                            print(f"Error in CAN message callback: {e}")

        listener_thread = threading.Thread(target=listen_thread, daemon=True)
        listener_thread.start()

class EthernetRobotInterface:
    def __init__(self, host='192.168.1.100', port=5005):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False

    def connect(self):
        """Connect to robot via Ethernet"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            print(f"Connected to robot at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False

    def send_command(self, command_type, data):
        """Send command to robot via Ethernet"""
        if not self.is_connected:
            raise Exception("Not connected to robot")

        try:
            # Format: [command_type:4][data_length:4][data]
            data_bytes = struct.pack('<I', len(data)) + data
            command_bytes = struct.pack('<I', command_type) + data_bytes

            self.socket.sendall(command_bytes)
            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False

    def receive_response(self, timeout=1.0):
        """Receive response from robot"""
        if not self.is_connected:
            raise Exception("Not connected to robot")

        try:
            self.socket.settimeout(timeout)
            response = self.socket.recv(1024)  # 1KB buffer
            return response
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Failed to receive response: {e}")
            return None

    def disconnect(self):
        """Disconnect from robot"""
        if self.socket:
            self.socket.close()
        self.is_connected = False
```

### 4. Power Management and Safety
```python
import time
import threading
from collections import deque

class PowerManager:
    def __init__(self):
        self.battery_voltage = 0.0
        self.current_draw = 0.0
        self.power_consumption_history = deque(maxlen=100)
        self.is_monitoring = False
        self.shutdown_threshold = 10.0  # 10% battery level
        self.emergency_stop = False

    def monitor_power(self, voltage_pin, current_pin):
        """Monitor power consumption in background"""
        def power_monitor_thread():
            while self.is_monitoring:
                # Read voltage and current (example implementation)
                voltage = self.read_analog_voltage(voltage_pin)
                current = self.read_analog_current(current_pin)

                self.battery_voltage = voltage
                self.current_draw = current

                # Calculate power consumption
                instantaneous_power = voltage * current
                self.power_consumption_history.append(instantaneous_power)

                # Check for low battery
                battery_level = self.estimate_battery_level(voltage)
                if battery_level < self.shutdown_threshold:
                    self.trigger_low_battery_shutdown()

                time.sleep(1.0)  # Update every second

        self.is_monitoring = True
        monitor_thread = threading.Thread(target=power_monitor_thread, daemon=True)
        monitor_thread.start()

    def read_analog_voltage(self, pin):
        """Read voltage from analog pin"""
        # Example implementation - replace with actual ADC reading
        # This would typically use an ADC like MCP3008
        return 12.6  # Example voltage

    def read_analog_current(self, pin):
        """Read current from analog pin"""
        # Example implementation - replace with actual current sensor reading
        return 2.5  # Example current in amps

    def estimate_battery_level(self, voltage):
        """Estimate battery level based on voltage"""
        # Example for 12V lead-acid battery
        if voltage > 12.6:
            return 100
        elif voltage > 12.4:
            return 90
        elif voltage > 12.2:
            return 80
        elif voltage > 12.0:
            return 60
        elif voltage > 11.8:
            return 40
        elif voltage > 11.6:
            return 20
        else:
            return 5

    def trigger_low_battery_shutdown(self):
        """Handle low battery condition"""
        print("WARNING: Low battery detected! Initiating safe shutdown...")
        # Stop all motors
        # Save current state
        # Navigate to charging station if possible
        # Enter low power mode

    def get_power_stats(self):
        """Get current power consumption statistics"""
        if not self.power_consumption_history:
            return {
                'voltage': self.battery_voltage,
                'current': self.current_draw,
                'power': 0.0,
                'battery_level': self.estimate_battery_level(self.battery_voltage)
            }

        avg_power = sum(self.power_consumption_history) / len(self.power_consumption_history)
        return {
            'voltage': self.battery_voltage,
            'current': self.current_draw,
            'power': avg_power,
            'battery_level': self.estimate_battery_level(self.battery_voltage)
        }

class SafetySystem:
    def __init__(self):
        self.emergency_stop_active = False
        self.safety_limits = {
            'max_velocity': 1.0,  # m/s
            'max_acceleration': 2.0,  # m/s²
            'max_current': 10.0,  # amps
            'max_temperature': 60.0  # Celsius
        }
        self.sensor_limits = {
            'min_distance': 0.1,  # meters
            'max_force': 50.0  # Newtons
        }

    def check_safety_conditions(self, robot_state, sensor_data):
        """Check if current state is safe"""
        # Check velocity limits
        if abs(robot_state.get('linear_velocity', 0)) > self.safety_limits['max_velocity']:
            self.trigger_safety_stop("Velocity limit exceeded")
            return False

        # Check acceleration limits
        if abs(robot_state.get('linear_acceleration', 0)) > self.safety_limits['max_acceleration']:
            self.trigger_safety_stop("Acceleration limit exceeded")
            return False

        # Check current draw
        if robot_state.get('current_draw', 0) > self.safety_limits['max_current']:
            self.trigger_safety_stop("Current limit exceeded")
            return False

        # Check temperature
        if robot_state.get('temperature', 0) > self.safety_limits['max_temperature']:
            self.trigger_safety_stop("Temperature limit exceeded")
            return False

        # Check sensor data
        if 'lidar' in sensor_data:
            min_distance = min(sensor_data['lidar']) if sensor_data['lidar'] else float('inf')
            if min_distance < self.sensor_limits['min_distance']:
                self.trigger_safety_stop(f"Obstacle too close: {min_distance:.2f}m")
                return False

        return True

    def trigger_safety_stop(self, reason):
        """Trigger emergency safety stop"""
        print(f"SAFETY STOP: {reason}")
        self.emergency_stop_active = True
        # Stop all actuators
        # Log the event
        # Potentially alert operator