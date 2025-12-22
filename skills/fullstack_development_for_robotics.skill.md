# Fullstack Development for Robotics

## Overview
Fullstack development for robotics encompasses the complete development lifecycle from frontend user interfaces to backend services, including communication with robotic hardware. This skill integrates frontend technologies for robot monitoring and control, backend systems for data processing and API services, and direct hardware integration for seamless end-to-end robotic applications.

## Key Technologies and Components
- **Frontend**: React/Vue/Angular for dashboards and control interfaces
- **Backend**: Python/Node.js for API services and data processing
- **Real-time Communication**: WebSocket, ROSbridge for robot communication
- **Data Storage**: PostgreSQL/MongoDB for telemetry and mission data
- **Hardware Integration**: Direct communication with sensors and actuators
- **DevOps**: Docker, CI/CD for deployment and maintenance

## Essential Fullstack Techniques

### 1. Integrated Frontend-Backend Architecture
```python
# backend/app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import logging
import redis
from datetime import datetime

app = FastAPI(title="Robotics Fullstack API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for real-time data sharing
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class RobotCommand(BaseModel):
    robot_id: str
    command_type: str
    parameters: Dict = {}
    priority: int = 1

class RobotState(BaseModel):
    robot_id: str
    position: Dict[str, float]
    orientation: Dict[str, float]
    velocity: Dict[str, float]
    battery_level: float
    status: str
    timestamp: datetime

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.robot_states: Dict[str, RobotState] = {}

    async def connect(self, websocket: WebSocket, robot_id: str):
        await websocket.accept()
        self.active_connections[robot_id] = websocket

        # Send initial state if available
        if robot_id in self.robot_states:
            await self.send_robot_state(robot_id, self.robot_states[robot_id])

    def disconnect(self, robot_id: str):
        if robot_id in self.active_connections:
            del self.active_connections[robot_id]

    async def broadcast_robot_state(self, robot_id: str, state: RobotState):
        """Broadcast state to all connected clients"""
        self.robot_states[robot_id] = state
        redis_client.set(f"robot:{robot_id}:state", state.json())

        # Send to robot connection
        if robot_id in self.active_connections:
            try:
                await self.active_connections[robot_id].send_text(state.json())
            except WebSocketDisconnect:
                self.disconnect(robot_id)

    async def send_command_to_robot(self, robot_id: str, command: RobotCommand):
        """Send command to robot through WebSocket"""
        if robot_id in self.active_connections:
            try:
                await self.active_connections[robot_id].send_text(
                    json.dumps({"type": "command", "data": command.dict()})
                )
            except WebSocketDisconnect:
                self.disconnect(robot_id)
                raise HTTPException(status_code=404, detail="Robot not connected")

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

@app.websocket("/ws/robot/{robot_id}")
async def websocket_endpoint(websocket: WebSocket, robot_id: str):
    await websocket_manager.connect(websocket, robot_id)
    try:
        while True:
            data = await websocket.receive_text()
            command_data = json.loads(data)

            if command_data.get("type") == "state_update":
                state = RobotState(**command_data["data"])
                await websocket_manager.broadcast_robot_state(robot_id, state)
    except WebSocketDisconnect:
        websocket_manager.disconnect(robot_id)

@app.post("/api/robots/{robot_id}/command")
async def send_command(robot_id: str, command: RobotCommand):
    """Send command to robot via API"""
    try:
        await websocket_manager.send_command_to_robot(robot_id, command)
        return {"status": "success", "message": "Command sent successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error sending command: {e}")
        raise HTTPException(status_code=500, detail="Failed to send command")

@app.get("/api/robots/{robot_id}/state")
async def get_robot_state(robot_id: str):
    """Get current robot state"""
    state_json = redis_client.get(f"robot:{robot_id}:state")
    if state_json:
        return json.loads(state_json)
    raise HTTPException(status_code=404, detail="Robot state not found")

@app.get("/api/robots")
async def get_all_robots():
    """Get list of all connected robots"""
    robot_states = {}
    for robot_id in websocket_manager.active_connections.keys():
        state_json = redis_client.get(f"robot:{robot_id}:state")
        if state_json:
            robot_states[robot_id] = json.loads(state_json)
    return robot_states

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```jsx
// frontend/src/components/RobotDashboard.jsx
import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import Robot3DViewer from './Robot3DViewer';
import RobotControls from './RobotControls';

const RobotDashboard = () => {
  const [robotData, setRobotData] = useState({
    position: { x: 0, y: 0, z: 0 },
    orientation: { roll: 0, pitch: 0, yaw: 0 },
    velocity: { linear: 0, angular: 0 },
    battery: 100,
    status: 'idle'
  });

  const [sensorHistory, setSensorHistory] = useState([]);
  const [commandHistory, setCommandHistory] = useState([]);
  const [connectedRobots, setConnectedRobots] = useState([]);
  const [selectedRobot, setSelectedRobot] = useState('');
  const [socket, setSocket] = useState(null);

  const wsRef = useRef(null);

  useEffect(() => {
    // Connect to backend WebSocket
    const ws = new WebSocket(`ws://localhost:8000/ws/robot/${selectedRobot || 'default'}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'state_update') {
        setRobotData(data.data);
        updateSensorHistory(data.data);
      } else if (data.type === 'command_response') {
        updateCommandHistory(data.data);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    return () => {
      ws.close();
    };
  }, [selectedRobot]);

  useEffect(() => {
    // Fetch connected robots
    fetch('http://localhost:8000/api/robots')
      .then(response => response.json())
      .then(data => {
        setConnectedRobots(Object.keys(data));
        if (!selectedRobot && Object.keys(data).length > 0) {
          setSelectedRobot(Object.keys(data)[0]);
        }
      })
      .catch(error => console.error('Error fetching robots:', error));
  }, [selectedRobot]);

  const updateSensorHistory = (data) => {
    setSensorHistory(prev => {
      const newHistory = [...prev, {
        timestamp: Date.now(),
        battery: data.battery_level,
        linear_velocity: data.velocity.linear,
        angular_velocity: data.velocity.angular,
        x: data.position.x,
        y: data.position.y
      }];
      return newHistory.slice(-50); // Keep last 50 data points
    });
  };

  const updateCommandHistory = (command) => {
    setCommandHistory(prev => [
      { ...command, timestamp: new Date().toISOString() },
      ...prev.slice(0, 9) // Keep last 10 commands
    ]);
  };

  const sendCommand = async (command) => {
    if (!selectedRobot) return;

    try {
      const response = await fetch(`http://localhost:8000/api/robots/${selectedRobot}/command`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(command),
      });

      if (!response.ok) {
        throw new Error('Failed to send command');
      }

      updateCommandHistory(command);
    } catch (error) {
      console.error('Error sending command:', error);
    }
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h1>Robotics Dashboard</h1>
        <div className="robot-selector">
          <label>Select Robot: </label>
          <select
            value={selectedRobot}
            onChange={(e) => setSelectedRobot(e.target.value)}
          >
            {connectedRobots.map(robotId => (
              <option key={robotId} value={robotId}>{robotId}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="status-panel">
          <h2>Robot Status</h2>
          <div className="status-grid">
            <div className="status-item">
              <label>Position</label>
              <div>X: {robotData.position.x.toFixed(2)} Y: {robotData.position.y.toFixed(2)}</div>
            </div>
            <div className="status-item">
              <label>Orientation</label>
              <div>Roll: {robotData.orientation.roll.toFixed(2)} Pitch: {robotData.orientation.pitch.toFixed(2)}</div>
            </div>
            <div className="status-item">
              <label>Battery</label>
              <div className={`battery-level ${robotData.battery > 20 ? 'good' : 'low'}`}>
                {robotData.battery}%
              </div>
            </div>
            <div className="status-item">
              <label>Status</label>
              <div className={`status-indicator ${robotData.status}`}>
                {robotData.status}
              </div>
            </div>
          </div>
        </div>

        <div className="controls-panel">
          <h2>Controls</h2>
          <RobotControls onCommand={sendCommand} />
        </div>

        <div className="visualization-panel">
          <h2>3D Visualization</h2>
          <Robot3DViewer robotState={robotData} />
        </div>

        <div className="sensors-panel">
          <h2>Sensor Data</h2>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={sensorHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="battery" stroke="#8884d8" />
                <Line type="monotone" dataKey="linear_velocity" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sensorHistory.slice(-10)}> {/* Last 10 data points */}
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="x" fill="#8884d8" />
                <Bar dataKey="y" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="commands-panel">
          <h2>Command History</h2>
          <div className="command-list">
            {commandHistory.map((cmd, index) => (
              <div key={index} className="command-item">
                <span className="command-type">{cmd.command_type}</span>
                <span className="command-params">{JSON.stringify(cmd.parameters)}</span>
                <span className="command-time">{new Date(cmd.timestamp).toLocaleTimeString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RobotDashboard;
```

### 2. Real-time Data Streaming and Visualization
```python
# backend/app/telemetry_service.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import asyncio
import json

Base = declarative_base()

class RobotTelemetry(Base):
    __tablename__ = "robot_telemetry"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    orientation_roll = Column(Float)
    orientation_pitch = Column(Float)
    orientation_yaw = Column(Float)
    battery_level = Column(Float)
    velocity_linear = Column(Float)
    velocity_angular = Column(Float)
    status = Column(String)
    sensor_data = Column(JSON)

class TelemetryService:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def store_telemetry(self, robot_state: RobotState):
        """Store robot telemetry data"""
        db = self.SessionLocal()
        try:
            telemetry = RobotTelemetry(
                robot_id=robot_state.robot_id,
                timestamp=robot_state.timestamp,
                position_x=robot_state.position.get('x', 0),
                position_y=robot_state.position.get('y', 0),
                position_z=robot_state.position.get('z', 0),
                orientation_roll=robot_state.orientation.get('roll', 0),
                orientation_pitch=robot_state.orientation.get('pitch', 0),
                orientation_yaw=robot_state.orientation.get('yaw', 0),
                battery_level=robot_state.battery_level,
                velocity_linear=robot_state.velocity.get('linear', 0),
                velocity_angular=robot_state.velocity.get('angular', 0),
                status=robot_state.status,
                sensor_data={}  # Add sensor data as needed
            )
            db.add(telemetry)
            db.commit()
        finally:
            db.close()

    async def get_telemetry_stream(self, robot_id: str,
                                 callback: callable,
                                 interval: float = 1.0):
        """Stream telemetry data to callback function"""
        while True:
            try:
                data = self.get_recent_telemetry(robot_id, limit=10)
                if data:
                    await callback(data)
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Telemetry stream error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    def get_recent_telemetry(self, robot_id: str, limit: int = 100) -> List[Dict]:
        """Get recent telemetry data for visualization"""
        db = self.SessionLocal()
        try:
            telemetry_data = db.query(RobotTelemetry).filter(
                RobotTelemetry.robot_id == robot_id
            ).order_by(RobotTelemetry.timestamp.desc()).limit(limit).all()

            return [{
                'timestamp': t.timestamp.isoformat(),
                'position': {
                    'x': t.position_x,
                    'y': t.position_y,
                    'z': t.position_z
                },
                'orientation': {
                    'roll': t.orientation_roll,
                    'pitch': t.orientation_pitch,
                    'yaw': t.orientation_yaw
                },
                'battery_level': t.battery_level,
                'velocity': {
                    'linear': t.velocity_linear,
                    'angular': t.velocity_angular
                },
                'status': t.status
            } for t in telemetry_data]
        finally:
            db.close()

    def get_telemetry_analytics(self, robot_id: str, days: int = 7) -> Dict:
        """Get analytics for telemetry data"""
        db = self.SessionLocal()
        try:
            from datetime import datetime, timedelta
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            telemetry_data = db.query(RobotTelemetry).filter(
                RobotTelemetry.robot_id == robot_id,
                RobotTelemetry.timestamp >= start_time,
                RobotTelemetry.timestamp <= end_time
            ).all()

            if not telemetry_data:
                return {"error": "No data available"}

            df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'battery_level': t.battery_level,
                'position_x': t.position_x,
                'position_y': t.position_y,
                'velocity_linear': t.velocity_linear,
                'status': t.status
            } for t in telemetry_data])

            analytics = {
                'avg_battery': float(df['battery_level'].mean()),
                'total_distance': float(self.calculate_total_distance(df)),
                'avg_velocity': float(df['velocity_linear'].mean()),
                'operation_time': float(self.calculate_operation_time(df)),
                'data_points': len(df)
            }

            return analytics
        finally:
            db.close()

    def calculate_total_distance(self, df) -> float:
        """Calculate total distance traveled"""
        if len(df) < 2:
            return 0.0

        distances = []
        for i in range(1, len(df)):
            dx = df.iloc[i]['position_x'] - df.iloc[i-1]['position_x']
            dy = df.iloc[i]['position_y'] - df.iloc[i-1]['position_y']
            distance = np.sqrt(dx**2 + dy**2)
            distances.append(distance)

        return sum(distances)

    def calculate_operation_time(self, df) -> float:
        """Calculate total operation time in hours"""
        if len(df) == 0:
            return 0.0

        time_diff = df['timestamp'].max() - df['timestamp'].min()
        return time_diff.total_seconds() / 3600
```

```jsx
// frontend/src/components/TelemetryChart.jsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const TelemetryChart = ({ robotId, data, metrics = ['battery', 'velocity'] }) => {
  const [chartData, setChartData] = useState([]);
  const [selectedMetrics, setSelectedMetrics] = useState(metrics);

  useEffect(() => {
    if (data && data.length > 0) {
      // Process telemetry data for chart
      const processedData = data.map(item => ({
        time: new Date(item.timestamp).toLocaleTimeString(),
        battery: item.battery_level,
        velocity: item.velocity.linear,
        x: item.position.x,
        y: item.position.y
      }));
      setChartData(processedData);
    }
  }, [data]);

  const toggleMetric = (metric) => {
    if (selectedMetrics.includes(metric)) {
      setSelectedMetrics(selectedMetrics.filter(m => m !== metric));
    } else {
      setSelectedMetrics([...selectedMetrics, metric]);
    }
  };

  const renderChartLines = () => {
    return selectedMetrics.map(metric => {
      const colors = {
        battery: '#8884d8',
        velocity: '#82ca9d',
        x: '#ffc658',
        y: '#ff7300'
      };

      return (
        <Line
          key={metric}
          type="monotone"
          dataKey={metric}
          stroke={colors[metric] || '#000'}
          activeDot={{ r: 8 }}
        />
      );
    });
  };

  return (
    <div className="telemetry-chart">
      <div className="chart-controls">
        {['battery', 'velocity', 'x', 'y'].map(metric => (
          <label key={metric} className="metric-checkbox">
            <input
              type="checkbox"
              checked={selectedMetrics.includes(metric)}
              onChange={() => toggleMetric(metric)}
            />
            {metric.charAt(0).toUpperCase() + metric.slice(1)}
          </label>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Legend />
          {renderChartLines()}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TelemetryChart;
```

### 3. Hardware Integration with Fullstack Architecture
```python
# backend/app/hardware_interface.py
import serial
import smbus2
import spidev
import time
import threading
from typing import Dict, Any, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class HardwareCommand:
    device_type: str
    device_id: str
    command: str
    parameters: Dict[str, Any]

class HardwareInterfaceManager:
    def __init__(self):
        self.serial_connections = {}
        self.i2c_bus = None
        self.spi_device = None
        self.hardware_status = {}
        self.command_queue = asyncio.Queue()
        self.is_running = False

    def initialize_serial(self, port: str, baudrate: int = 115200) -> bool:
        """Initialize serial connection to hardware"""
        try:
            ser = serial.Serial(port, baudrate=baudrate, timeout=1)
            self.serial_connections[port] = ser
            self.hardware_status[port] = {'connected': True, 'type': 'serial'}
            return True
        except Exception as e:
            print(f"Failed to initialize serial {port}: {e}")
            self.hardware_status[port] = {'connected': False, 'error': str(e)}
            return False

    def initialize_i2c(self, bus_number: int = 1):
        """Initialize I2C bus"""
        try:
            self.i2c_bus = smbus2.SMBus(bus_number)
            self.hardware_status[f'i2c_{bus_number}'] = {'connected': True, 'type': 'i2c'}
            return True
        except Exception as e:
            print(f"Failed to initialize I2C bus {bus_number}: {e}")
            self.hardware_status[f'i2c_{bus_number}'] = {'connected': False, 'error': str(e)}
            return False

    def send_serial_command(self, port: str, command: str) -> Optional[str]:
        """Send command via serial connection"""
        if port not in self.serial_connections:
            raise Exception(f"Serial port {port} not initialized")

        ser = self.serial_connections[port]
        try:
            ser.write(command.encode() + b'\n')
            response = ser.readline().decode().strip()
            return response
        except Exception as e:
            print(f"Serial command error: {e}")
            return None

    def read_i2c_sensor(self, address: int, register: int, length: int = 1) -> Optional[bytes]:
        """Read from I2C sensor"""
        if not self.i2c_bus:
            raise Exception("I2C bus not initialized")

        try:
            if length == 1:
                data = self.i2c_bus.read_byte_data(address, register)
            else:
                data = self.i2c_bus.read_i2c_block_data(address, register, length)
            return data
        except Exception as e:
            print(f"I2C read error: {e}")
            return None

    async def execute_hardware_command(self, cmd: HardwareCommand) -> Dict[str, Any]:
        """Execute hardware command based on device type"""
        if cmd.device_type == 'serial':
            response = self.send_serial_command(cmd.device_id, cmd.command)
            return {
                'status': 'success' if response else 'failed',
                'response': response,
                'command': cmd.command
            }
        elif cmd.device_type == 'i2c_sensor':
            # Parse command for I2C operations
            parts = cmd.command.split(':')
            if len(parts) >= 2:
                address = int(parts[0], 16)
                register = int(parts[1], 16)
                length = int(parts[2]) if len(parts) > 2 else 1

                data = self.read_i2c_sensor(address, register, length)
                return {
                    'status': 'success' if data is not None else 'failed',
                    'data': data,
                    'command': cmd.command
                }

        return {'status': 'unknown_device_type', 'command': cmd.command}

    async def start_command_processor(self):
        """Start processing hardware commands from queue"""
        self.is_running = True
        while self.is_running:
            try:
                cmd = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                result = await self.execute_hardware_command(cmd)
                # Could send result back to frontend via WebSocket
                print(f"Hardware command result: {result}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Command processing error: {e}")

    def stop_command_processor(self):
        """Stop command processor"""
        self.is_running = False

class RobotHardwareInterface:
    def __init__(self, hardware_manager: HardwareInterfaceManager):
        self.hw_manager = hardware_manager
        self.motor_controller = None
        self.sensor_array = {}

    def initialize_robot_hardware(self) -> bool:
        """Initialize all robot hardware components"""
        success = True

        # Initialize motor controller via serial
        if not self.hw_manager.initialize_serial('/dev/ttyUSB0', 115200):
            success = False

        # Initialize I2C sensors
        if not self.hw_manager.initialize_i2c(1):
            success = False

        return success

    async def move_robot(self, linear_vel: float, angular_vel: float):
        """Send movement command to robot"""
        command = HardwareCommand(
            device_type='serial',
            device_id='/dev/ttyUSB0',
            command=f'MOVE:{linear_vel}:{angular_vel}',
            parameters={'linear': linear_vel, 'angular': angular_vel}
        )
        await self.hw_manager.command_queue.put(command)

    async def read_sensors(self) -> Dict[str, Any]:
        """Read all sensor data"""
        sensor_data = {}

        # Read IMU via I2C (example address 0x68)
        imu_data = self.hw_manager.read_i2c_sensor(0x68, 0x3B, 14)
        if imu_data:
            sensor_data['imu'] = self._parse_imu_data(imu_data)

        # Read other sensors as needed
        # This would include encoders, LIDAR, cameras, etc.

        return sensor_data

    def _parse_imu_data(self, raw_data: bytes) -> Dict[str, float]:
        """Parse raw IMU data"""
        if len(raw_data) < 14:
            return {}

        # Unpack raw data (example for MPU6050)
        import struct
        try:
            accel_x, accel_y, accel_z, temp, gyro_x, gyro_y, gyro_z = struct.unpack(
                '>hhhhh', raw_data[:14]
            )

            return {
                'accelerometer': {
                    'x': accel_x / 16384.0,
                    'y': accel_y / 16384.0,
                    'z': accel_z / 16384.0
                },
                'gyroscope': {
                    'x': gyro_x / 131.0,
                    'y': gyro_y / 131.0,
                    'z': gyro_z / 131.0
                },
                'temperature': temp / 340.0 + 36.53
            }
        except Exception as e:
            print(f"IMU data parsing error: {e}")
            return {}
```

### 4. Mission Planning and Execution Fullstack
```python
# backend/app/mission_service.py
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import asyncio
import json
from typing import List, Dict, Any

Base = declarative_base()

class Mission(Base):
    __tablename__ = "missions"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(String, index=True)
    mission_name = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String, default="pending")  # pending, active, completed, failed
    waypoints = Column(JSON)
    tasks = Column(JSON)
    completed_tasks = Column(JSON, default=lambda: [])
    metrics = Column(JSON)
    is_active = Column(Boolean, default=False)

class MissionService:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_mission(self, robot_id: str, mission_name: str,
                      waypoints: List[Dict], tasks: List[Dict]) -> str:
        """Create a new mission"""
        db = self.SessionLocal()
        try:
            mission = Mission(
                robot_id=robot_id,
                mission_name=mission_name,
                waypoints=waypoints,
                tasks=tasks,
                status="pending"
            )
            db.add(mission)
            db.commit()
            return str(mission.id)
        finally:
            db.close()

    def get_active_missions(self, robot_id: str = None) -> List[Dict]:
        """Get active missions"""
        db = self.SessionLocal()
        try:
            query = db.query(Mission)
            if robot_id:
                query = query.filter(Mission.robot_id == robot_id)
            query = query.filter(Mission.is_active == True)

            missions = query.all()
            return [{
                'id': m.id,
                'robot_id': m.robot_id,
                'mission_name': m.mission_name,
                'status': m.status,
                'waypoints': m.waypoints,
                'tasks': m.tasks,
                'completed_tasks': m.completed_tasks,
                'start_time': m.start_time.isoformat() if m.start_time else None
            } for m in missions]
        finally:
            db.close()

    def update_mission_status(self, mission_id: int, status: str,
                            completed_task: str = None):
        """Update mission status"""
        db = self.SessionLocal()
        try:
            mission = db.query(Mission).filter(Mission.id == mission_id).first()
            if mission:
                mission.status = status
                if completed_task and completed_task not in mission.completed_tasks:
                    if not mission.completed_tasks:
                        mission.completed_tasks = []
                    mission.completed_tasks.append(completed_task)

                if status in ["completed", "failed"]:
                    mission.is_active = False
                    mission.end_time = datetime.utcnow()

                db.commit()
        finally:
            db.close()

    async def execute_mission(self, mission_id: int, robot_interface):
        """Execute mission with robot interface"""
        db = self.SessionLocal()
        try:
            mission = db.query(Mission).filter(Mission.id == mission_id).first()
            if not mission:
                return False

            # Update mission status to active
            mission.status = "active"
            mission.is_active = True
            db.commit()

            # Execute tasks sequentially
            for i, task in enumerate(mission.tasks):
                try:
                    # Update progress
                    progress = (i / len(mission.tasks)) * 100
                    print(f"Mission {mission_id} progress: {progress:.1f}%")

                    # Execute task based on type
                    if task['type'] == 'navigate':
                        await robot_interface.navigate_to(task['waypoint'])
                    elif task['type'] == 'inspect':
                        await robot_interface.inspect_area(task['area'])
                    elif task['type'] == 'manipulate':
                        await robot_interface.manipulate_object(task['object'])

                    # Update completed tasks
                    self.update_mission_status(mission_id, "active", task['id'])

                except Exception as e:
                    print(f"Task {task['id']} failed: {e}")
                    self.update_mission_status(mission_id, "failed")
                    return False

            # Mission completed successfully
            self.update_mission_status(mission_id, "completed")
            return True

        finally:
            db.close()

# Frontend mission planning component
```

```jsx
// frontend/src/components/MissionPlanner.jsx
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const MissionPlanner = ({ robotId, onMissionCreated }) => {
  const [waypoints, setWaypoints] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [missionName, setMissionName] = useState('');
  const [selectedPosition, setSelectedPosition] = useState(null);
  const [mapCenter] = useState([51.505, -0.09]); // Default to London
  const [mapZoom] = useState(13);

  const handleMapClick = (e) => {
    const newWaypoint = {
      id: Date.now(),
      lat: e.latlng.lat,
      lng: e.latlng.lng,
      sequence: waypoints.length
    };
    setWaypoints([...waypoints, newWaypoint]);
  };

  const removeWaypoint = (id) => {
    setWaypoints(waypoints.filter(wp => wp.id !== id));
  };

  const addTask = () => {
    const newTask = {
      id: Date.now(),
      type: 'navigate',
      waypointId: waypoints.length > 0 ? waypoints[waypoints.length - 1]?.id : null,
      description: 'Navigate to waypoint',
      completed: false
    };
    setTasks([...tasks, newTask]);
  };

  const createMission = async () => {
    if (!missionName || waypoints.length === 0) {
      alert('Please provide a mission name and at least one waypoint');
      return;
    }

    const missionData = {
      robotId,
      missionName,
      waypoints,
      tasks
    };

    try {
      const response = await fetch('http://localhost:8000/api/missions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(missionData),
      });

      if (response.ok) {
        alert('Mission created successfully!');
        onMissionCreated(missionData);
        // Reset form
        setWaypoints([]);
        setTasks([]);
        setMissionName('');
      } else {
        throw new Error('Failed to create mission');
      }
    } catch (error) {
      console.error('Error creating mission:', error);
      alert('Error creating mission');
    }
  };

  return (
    <div className="mission-planner">
      <div className="mission-controls">
        <h3>Mission Planner</h3>
        <div className="mission-inputs">
          <input
            type="text"
            placeholder="Mission Name"
            value={missionName}
            onChange={(e) => setMissionName(e.target.value)}
          />
          <button onClick={addTask}>Add Task</button>
          <button onClick={createMission}>Create Mission</button>
        </div>
      </div>

      <div className="mission-map">
        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          style={{ height: '400px', width: '100%' }}
          onClick={handleMapClick}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />

          {waypoints.map((wp) => (
            <Marker key={wp.id} position={[wp.lat, wp.lng]}>
              <Popup>
                Waypoint {wp.sequence + 1}<br />
                <button onClick={() => removeWaypoint(wp.id)}>Remove</button>
              </Popup>
            </Marker>
          ))}

          {waypoints.length > 1 && (
            <Polyline
              positions={waypoints.map(wp => [wp.lat, wp.lng])}
              color="red"
            />
          )}
        </MapContainer>
      </div>

      <div className="mission-waypoints">
        <h4>Waypoints ({waypoints.length})</h4>
        <ul>
          {waypoints.map((wp, index) => (
            <li key={wp.id}>
              Waypoint {index + 1}: ({wp.lat.toFixed(4)}, {wp.lng.toFixed(4)})
              <button onClick={() => removeWaypoint(wp.id)}>X</button>
            </li>
          ))}
        </ul>
      </div>

      <div className="mission-tasks">
        <h4>Tasks ({tasks.length})</h4>
        <ul>
          {tasks.map((task, index) => (
            <li key={task.id}>
              Task {index + 1}: {task.description}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default MissionPlanner;
```

## Best Practices for Fullstack Robotics Development

### 1. Performance Optimization
```python
# backend/app/performance_optimizer.py
import asyncio
import aioredis
from contextlib import asynccontextmanager
import time
from functools import wraps

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

class OptimizedRobotDB:
    def __init__(self, database_url: str, redis_url: str = "redis://localhost"):
        self.database_url = database_url
        self.redis_url = redis_url
        self.db_pool = None
        self.redis = None

    async def initialize(self):
        """Initialize database and Redis connections"""
        # Initialize Redis
        self.redis = aioredis.from_url(self.redis_url)

        # Database pool would be initialized here
        # self.db_pool = await asyncpg.create_pool(self.database_url)

    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection from pool"""
        # conn = await self.db_pool.acquire()
        # try:
        #     yield conn
        # finally:
        #     await self.db_pool.release(conn)
        pass

    @async_timer
    async def get_cached_robot_state(self, robot_id: str):
        """Get robot state with Redis caching"""
        # Try cache first
        cached = await self.redis.get(f"robot_state:{robot_id}")
        if cached:
            return json.loads(cached)

        # If not in cache, get from database and cache it
        # state = await self.get_robot_state_from_db(robot_id)
        # await self.redis.setex(f"robot_state:{robot_id}", 30, json.dumps(state))  # Cache for 30 seconds
        # return state
        pass

class RealTimeOptimizations:
    def __init__(self):
        self.data_batch_size = 10
        self.batch_buffer = []
        self.batch_timer = None

    async def batch_process_sensor_data(self, sensor_data: Dict):
        """Batch process sensor data for efficiency"""
        self.batch_buffer.append(sensor_data)

        if len(self.batch_buffer) >= self.data_batch_size:
            await self.process_batch()
        elif self.batch_timer is None:
            # Start timer to process batch even if not full
            self.batch_timer = asyncio.create_task(self.delayed_batch_process())

    async def delayed_batch_process(self):
        """Process batch after delay if not full"""
        await asyncio.sleep(0.1)  # Wait 100ms
        await self.process_batch()

    async def process_batch(self):
        """Process accumulated batch"""
        if self.batch_buffer:
            batch_data = self.batch_buffer.copy()
            self.batch_buffer.clear()

            # Process batch efficiently
            # This could involve database writes, calculations, etc.
            print(f"Processed batch of {len(batch_data)} sensor readings")
```

### 2. Security and Authentication
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

security = HTTPBearer()

class AuthenticationManager:
    def __init__(self, secret_key: str = "your-secret-key"):
        self.secret_key = secret_key
        self.algorithm = "HS256"

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify JWT token"""
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=[self.algorithm])
            robot_id: str = payload.get("robot_id")
            if robot_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return robot_id
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Apply authentication to robot endpoints
auth_manager = AuthenticationManager()

@app.post("/api/robots/{robot_id}/command")
async def send_command_secure(
    robot_id: str,
    command: RobotCommand,
    current_robot_id: str = Depends(auth_manager.verify_token)
):
    if robot_id != current_robot_id:
        raise HTTPException(status_code=403, detail="Not authorized for this robot")

    # Process command...
    pass
```

## Best Practices
- Implement real-time data streaming between frontend and backend
- Use WebSocket connections for low-latency robot communication
- Apply caching strategies to improve performance
- Implement proper error handling and fallback mechanisms
- Use database connection pooling for efficient resource usage
- Apply security measures including authentication and authorization
- Design responsive interfaces for different device types
- Implement comprehensive logging and monitoring
- Use Docker for consistent deployment environments
- Implement CI/CD pipelines for automated testing and deployment