# Backend Development for Robotics

## Overview
Backend development for robotics involves creating server-side applications that handle robot communication, data processing, API services, and system integration. This includes REST APIs, real-time communication systems, data storage solutions, and integration with ROS and other robotics frameworks.

## Key Technologies
- **Python/FastAPI**: High-performance web APIs
- **Node.js/Express**: Real-time communication and event handling
- **ROS2**: Robot Operating System for communication
- **PostgreSQL/MongoDB**: Data storage for robot data
- **Redis**: Caching and real-time data storage
- **Docker**: Containerization for deployment
- **gRPC**: High-performance RPC communication
- **Message Queues**: Task processing and event handling

## Essential Backend Techniques

### 1. Robot Communication API
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import logging
from datetime import datetime
import redis
from dataclasses import dataclass
from enum import Enum

app = FastAPI(title="Robotics Backend API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for real-time data storage
redis_client = redis.Redis(host='localhost', port=6379, db=0)

class RobotStatus(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    BUSY = "busy"
    ERROR = "error"
    CHARGING = "charging"

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
    status: RobotStatus
    timestamp: datetime

class RobotWebSocketManager:
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
        """Broadcast state update to all connected clients"""
        self.robot_states[robot_id] = state

        # Store in Redis for persistence
        redis_client.set(f"robot:{robot_id}:state", state.json())

        # Broadcast to WebSocket connections
        if robot_id in self.active_connections:
            try:
                await self.active_connections[robot_id].send_text(state.json())
            except WebSocketDisconnect:
                self.disconnect(robot_id)

    async def send_robot_command(self, robot_id: str, command: RobotCommand):
        """Send command to specific robot"""
        if robot_id in self.active_connections:
            try:
                await self.active_connections[robot_id].send_text(
                    json.dumps({"type": "command", "data": command.dict()})
                )
            except WebSocketDisconnect:
                self.disconnect(robot_id)
                raise HTTPException(status_code=404, detail="Robot not connected")

    async def send_robot_state(self, robot_id: str, state: RobotState):
        """Send state to specific robot"""
        if robot_id in self.active_connections:
            try:
                await self.active_connections[robot_id].send_text(
                    json.dumps({"type": "state", "data": state.dict()})
                )
            except WebSocketDisconnect:
                self.disconnect(robot_id)

# Initialize WebSocket manager
websocket_manager = RobotWebSocketManager()

@app.websocket("/ws/robot/{robot_id}")
async def websocket_endpoint(websocket: WebSocket, robot_id: str):
    await websocket_manager.connect(websocket, robot_id)
    try:
        while True:
            data = await websocket.receive_text()
            command_data = json.loads(data)

            # Process incoming commands from robot
            if command_data.get("type") == "state_update":
                state = RobotState(**command_data["data"])
                await websocket_manager.broadcast_robot_state(robot_id, state)
            elif command_data.get("type") == "command_response":
                # Handle command response
                logging.info(f"Command response from {robot_id}: {command_data}")
    except WebSocketDisconnect:
        websocket_manager.disconnect(robot_id)
        logging.info(f"Robot {robot_id} disconnected")

@app.post("/api/robots/{robot_id}/command")
async def send_command(robot_id: str, command: RobotCommand):
    """Send command to robot"""
    try:
        await websocket_manager.send_robot_command(robot_id, command)
        return {"status": "success", "message": "Command sent successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error sending command to {robot_id}: {e}")
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
```

### 2. Data Processing and Storage
```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from datetime import datetime
import pandas as pd
from typing import List, Optional
import numpy as np

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
    sensor_data = Column(JSON)  # Store sensor readings as JSON

class MissionData(Base):
    __tablename__ = "mission_data"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(String, index=True)
    mission_id = Column(String, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String)
    waypoints = Column(JSON)
    completed_tasks = Column(JSON)
    metrics = Column(JSON)

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
                status=robot_state.status.value,
                sensor_data={}  # Add sensor data as needed
            )
            db.add(telemetry)
            db.commit()
        finally:
            db.close()

    def get_telemetry_history(self, robot_id: str, start_time: datetime, end_time: datetime) -> List[RobotTelemetry]:
        """Get telemetry history for a robot"""
        db = self.SessionLocal()
        try:
            telemetry_data = db.query(RobotTelemetry).filter(
                RobotTelemetry.robot_id == robot_id,
                RobotTelemetry.timestamp >= start_time,
                RobotTelemetry.timestamp <= end_time
            ).order_by(RobotTelemetry.timestamp).all()
            return telemetry_data
        finally:
            db.close()

    def analyze_telemetry_patterns(self, robot_id: str, days: int = 7) -> Dict:
        """Analyze telemetry patterns for insights"""
        from datetime import datetime, timedelta

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        telemetry_data = self.get_telemetry_history(robot_id, start_time, end_time)

        if not telemetry_data:
            return {"error": "No data available"}

        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'battery_level': t.battery_level,
                'position_x': t.position_x,
                'position_y': t.position_y,
                'velocity_linear': t.velocity_linear,
                'status': t.status
            } for t in telemetry_data
        ])

        # Calculate statistics
        analysis = {
            'avg_battery': df['battery_level'].mean(),
            'total_distance': self.calculate_total_distance(df),
            'avg_velocity': df['velocity_linear'].mean(),
            'operation_time': self.calculate_operation_time(df),
            'charging_cycles': self.count_charging_cycles(df)
        }

        return analysis

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
        """Calculate total operation time"""
        if len(df) == 0:
            return 0.0

        time_diff = df['timestamp'].max() - df['timestamp'].min()
        return time_diff.total_seconds() / 3600  # Convert to hours

    def count_charging_cycles(self, df) -> int:
        """Count charging cycles based on battery level changes"""
        charging_events = 0
        battery_threshold = 20  # Consider as charging when battery increases significantly

        for i in range(1, len(df)):
            if df.iloc[i]['battery_level'] > df.iloc[i-1]['battery_level'] + 10:
                charging_events += 1

        return charging_events

class MissionService:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_mission(self, robot_id: str, waypoints: List[Dict], mission_id: str = None) -> str:
        """Create a new mission"""
        import uuid
        mission_id = mission_id or str(uuid.uuid4())

        db = self.SessionLocal()
        try:
            mission = MissionData(
                robot_id=robot_id,
                mission_id=mission_id,
                status="active",
                waypoints=waypoints,
                completed_tasks=[],
                metrics={}
            )
            db.add(mission)
            db.commit()
            return mission_id
        finally:
            db.close()

    def update_mission_status(self, mission_id: str, status: str, completed_tasks: List[str] = None):
        """Update mission status"""
        db = self.SessionLocal()
        try:
            mission = db.query(MissionData).filter(MissionData.mission_id == mission_id).first()
            if mission:
                mission.status = status
                if completed_tasks:
                    mission.completed_tasks = completed_tasks
                db.commit()
        finally:
            db.close()
```

### 3. Real-time Communication and Event Handling
```python
import asyncio
import websockets
import json
from typing import Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

class RobotEventType(str, Enum):
    STATE_UPDATE = "state_update"
    COMMAND_RECEIVED = "command_received"
    ERROR_OCCURRED = "error_occurred"
    MISSION_COMPLETED = "mission_completed"
    BATTERY_LOW = "battery_low"
    OBSTACLE_DETECTED = "obstacle_detected"

@dataclass
class RobotEvent:
    robot_id: str
    event_type: RobotEventType
    data: Dict[str, Any]
    timestamp: datetime

class EventBroker:
    def __init__(self):
        self.subscribers: Dict[RobotEventType, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self.is_running = False

    def subscribe(self, event_type: RobotEventType, callback: Callable):
        """Subscribe to specific event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: RobotEventType, callback: Callable):
        """Unsubscribe from specific event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)

    async def publish(self, event: RobotEvent):
        """Publish event to all subscribers"""
        await self.event_queue.put(event)

    async def process_events(self):
        """Process events from queue"""
        self.is_running = True
        while self.is_running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                # Notify subscribers
                if event.event_type in self.subscribers:
                    for callback in self.subscribers[event.event_type]:
                        try:
                            await callback(event)
                        except Exception as e:
                            logging.error(f"Error in event callback: {e}")
            except asyncio.TimeoutError:
                continue  # Continue waiting for events

    def stop(self):
        """Stop event processing"""
        self.is_running = False

class RobotEventService:
    def __init__(self):
        self.event_broker = EventBroker()
        self.robot_connections: Dict[str, asyncio.Queue] = {}

    async def start_event_processing(self):
        """Start event processing loop"""
        await self.event_broker.process_events()

    async def handle_robot_state_update(self, robot_id: str, state_data: Dict):
        """Handle robot state update"""
        event = RobotEvent(
            robot_id=robot_id,
            event_type=RobotEventType.STATE_UPDATE,
            data=state_data,
            timestamp=datetime.utcnow()
        )
        await self.event_broker.publish(event)

        # Check for specific conditions that trigger other events
        await self.check_battery_level(robot_id, state_data.get('battery_level', 100))
        await self.check_obstacles(robot_id, state_data.get('sensor_data', {}))

    async def check_battery_level(self, robot_id: str, battery_level: float):
        """Check if battery level is low"""
        if battery_level < 20:  # Low battery threshold
            event = RobotEvent(
                robot_id=robot_id,
                event_type=RobotEventType.BATTERY_LOW,
                data={'battery_level': battery_level},
                timestamp=datetime.utcnow()
            )
            await self.event_broker.publish(event)

    async def check_obstacles(self, robot_id: str, sensor_data: Dict):
        """Check for obstacle detection"""
        lidar_data = sensor_data.get('lidar', [])
        if lidar_data:
            min_distance = min(lidar_data) if lidar_data else float('inf')
            if min_distance < 0.5:  # 50cm threshold
                event = RobotEvent(
                    robot_id=robot_id,
                    event_type=RobotEventType.OBSTACLE_DETECTED,
                    data={'min_distance': min_distance},
                    timestamp=datetime.utcnow()
                )
                await self.event_broker.publish(event)

    async def register_robot_connection(self, robot_id: str, connection_queue: asyncio.Queue):
        """Register robot connection for direct communication"""
        self.robot_connections[robot_id] = connection_queue

    async def send_command_to_robot(self, robot_id: str, command: Dict):
        """Send command directly to robot"""
        if robot_id in self.robot_connections:
            await self.robot_connections[robot_id].put(command)
        else:
            logging.warning(f"Robot {robot_id} not connected")

# Event handlers
async def battery_low_handler(event: RobotEvent):
    """Handle low battery events"""
    logging.warning(f"Robot {event.robot_id} has low battery: {event.data['battery_level']}%")
    # Send robot to charging station
    # Send notification to operators
    pass

async def obstacle_detected_handler(event: RobotEvent):
    """Handle obstacle detection events"""
    logging.info(f"Robot {event.robot_id} detected obstacle at {event.data['min_distance']}m")
    # Stop robot
    # Plan alternative route
    # Send notification
    pass

# Register event handlers
event_service = RobotEventService()
event_service.event_broker.subscribe(RobotEventType.BATTERY_LOW, battery_low_handler)
event_service.event_broker.subscribe(RobotEventType.OBSTACLE_DETECTED, obstacle_detected_handler)
```

### 4. Task Scheduling and Job Processing
```python
import asyncio
import aiocron
from datetime import datetime, timedelta
from typing import Dict, List, Callable
import aioschedule
import time

class RobotTaskScheduler:
    def __init__(self):
        self.scheduled_tasks: Dict[str, Dict] = {}
        self.active_jobs: List[Dict] = []

    async def schedule_task(self, task_id: str, robot_id: str, task_func: Callable,
                           schedule_time: datetime = None, interval: int = None,
                           params: Dict = None):
        """Schedule a task for a robot"""
        task_info = {
            'task_id': task_id,
            'robot_id': robot_id,
            'task_func': task_func,
            'schedule_time': schedule_time,
            'interval': interval,
            'params': params or {},
            'status': 'scheduled',
            'created_at': datetime.utcnow()
        }

        if interval:
            # Schedule recurring task
            cron_task = aiocron.crontab(f'*/{interval} * * * *', func=lambda: self.execute_task(task_info))
            task_info['cron_task'] = cron_task
        else:
            # Schedule one-time task
            delay = (schedule_time - datetime.utcnow()).total_seconds()
            if delay > 0:
                asyncio.create_task(self._execute_delayed_task(task_info, delay))

        self.scheduled_tasks[task_id] = task_info

    async def _execute_delayed_task(self, task_info: Dict, delay: float):
        """Execute task after delay"""
        await asyncio.sleep(delay)
        await self.execute_task(task_info)

    async def execute_task(self, task_info: Dict):
        """Execute scheduled task"""
        try:
            task_info['status'] = 'executing'
            task_info['start_time'] = datetime.utcnow()

            # Add to active jobs
            self.active_jobs.append(task_info)

            # Execute the task
            result = await task_info['task_func'](task_info['robot_id'], **task_info['params'])

            task_info['status'] = 'completed'
            task_info['result'] = result
            task_info['end_time'] = datetime.utcnow()

            # Remove from active jobs
            self.active_jobs = [job for job in self.active_jobs if job['task_id'] != task_info['task_id']]

        except Exception as e:
            task_info['status'] = 'failed'
            task_info['error'] = str(e)
            task_info['end_time'] = datetime.utcnow()
        finally:
            # Update task status
            self.scheduled_tasks[task_info['task_id']] = task_info

    def cancel_task(self, task_id: str):
        """Cancel scheduled task"""
        if task_id in self.scheduled_tasks:
            task_info = self.scheduled_tasks[task_id]
            if 'cron_task' in task_info:
                task_info['cron_task'].stop()
            task_info['status'] = 'cancelled'
            self.scheduled_tasks[task_id] = task_info

    def get_task_status(self, task_id: str) -> Dict:
        """Get status of specific task"""
        return self.scheduled_tasks.get(task_id, {})

    def get_robot_tasks(self, robot_id: str) -> List[Dict]:
        """Get all tasks for specific robot"""
        return [task for task in self.scheduled_tasks.values() if task['robot_id'] == robot_id]

class MaintenanceScheduler:
    def __init__(self, telemetry_service: TelemetryService):
        self.telemetry_service = telemetry_service
        self.scheduler = RobotTaskScheduler()

    async def schedule_maintenance_tasks(self, robot_id: str):
        """Schedule maintenance tasks based on usage patterns"""
        # Analyze robot usage
        analysis = self.telemetry_service.analyze_telemetry_patterns(robot_id, days=30)

        # Schedule maintenance based on usage
        if analysis.get('operation_time', 0) > 100:  # More than 100 hours of operation
            await self.schedule_battery_maintenance(robot_id)

        if analysis.get('avg_battery', 100) < 80:  # Battery degradation
            await self.schedule_battery_replacement(robot_id)

    async def schedule_battery_maintenance(self, robot_id: str):
        """Schedule battery maintenance task"""
        async def battery_maintenance_task(robot_id):
            # Execute battery maintenance procedures
            logging.info(f"Scheduled battery maintenance for robot {robot_id}")
            # This would call actual maintenance procedures
            return {"status": "completed", "notes": "Battery maintenance performed"}

        schedule_time = datetime.utcnow() + timedelta(hours=24)  # Schedule for tomorrow
        await self.scheduler.schedule_task(
            f"battery_maintenance_{robot_id}",
            robot_id,
            battery_maintenance_task,
            schedule_time=schedule_time
        )

    async def schedule_battery_replacement(self, robot_id: str):
        """Schedule battery replacement task"""
        async def battery_replacement_task(robot_id):
            # Execute battery replacement procedures
            logging.info(f"Scheduled battery replacement for robot {robot_id}")
            # This would call actual replacement procedures
            return {"status": "completed", "notes": "Battery replaced"}

        schedule_time = datetime.utcnow() + timedelta(hours=12)  # Schedule for soon
        await self.scheduler.schedule_task(
            f"battery_replacement_{robot_id}",
            robot_id,
            battery_replacement_task,
            schedule_time=schedule_time
        )
```

## Best Practices for Robotics Backend Development

### 1. Performance Optimization
```python
import asyncio
import asyncpg
from contextlib import asynccontextmanager
from typing import Optional
import time

class OptimizedRobotDB:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None

    async def initialize_pool(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool"""
        if not self.pool:
            await self.initialize_pool()
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)

    async def batch_insert_telemetry(self, robot_states: List[RobotState]):
        """Batch insert multiple telemetry records"""
        async with self.get_connection() as conn:
            # Prepare data for batch insert
            records = []
            for state in robot_states:
                record = (
                    state.robot_id,
                    state.timestamp,
                    state.position.get('x', 0),
                    state.position.get('y', 0),
                    state.position.get('z', 0),
                    state.orientation.get('roll', 0),
                    state.orientation.get('pitch', 0),
                    state.orientation.get('yaw', 0),
                    state.battery_level,
                    state.velocity.get('linear', 0),
                    state.velocity.get('angular', 0),
                    state.status.value
                )
                records.append(record)

            # Execute batch insert
            await conn.executemany("""
                INSERT INTO robot_telemetry
                (robot_id, timestamp, position_x, position_y, position_z,
                 orientation_roll, orientation_pitch, orientation_yaw,
                 battery_level, velocity_linear, velocity_angular, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, records)

    async def get_optimized_robot_state(self, robot_id: str) -> Optional[RobotState]:
        """Get robot state with optimized query"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT robot_id, timestamp, position_x, position_y, position_z,
                       orientation_roll, orientation_pitch, orientation_yaw,
                       battery_level, velocity_linear, velocity_angular, status
                FROM robot_telemetry
                WHERE robot_id = $1
                ORDER BY timestamp DESC
                LIMIT 1
            """, robot_id)

            if row:
                return RobotState(
                    robot_id=row['robot_id'],
                    timestamp=row['timestamp'],
                    position={
                        'x': row['position_x'],
                        'y': row['position_y'],
                        'z': row['position_z']
                    },
                    orientation={
                        'roll': row['orientation_roll'],
                        'pitch': row['orientation_pitch'],
                        'yaw': row['orientation_yaw']
                    },
                    velocity={
                        'linear': row['velocity_linear'],
                        'angular': row['velocity_angular']
                    },
                    battery_level=row['battery_level'],
                    status=RobotStatus(row['status'])
                )
            return None

class CachingService:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}

    def set(self, key: str, value: Any, ttl: int = 300):  # 5 minutes default
        """Set value in cache with TTL"""
        self.cache[key] = value
        self.cache_expiry[key] = time.time() + ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache_expiry:
            if time.time() < self.cache_expiry[key]:
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.cache_expiry[key]
        return None

    def invalidate(self, key: str):
        """Invalidate specific cache key"""
        if key in self.cache:
            del self.cache[key]
        if key in self.cache_expiry:
            del self.cache_expiry[key]
```

### 2. Error Handling and Monitoring
```python
import traceback
from typing import Dict, Any
import sentry_sdk
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RobotError(Exception):
    """Base exception for robot-related errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "ROBOT_ERROR"
        self.details = details or {}

class RobotCommunicationError(RobotError):
    """Raised when robot communication fails"""
    pass

class RobotNotFoundError(RobotError):
    """Raised when robot is not found"""
    pass

class MissionError(RobotError):
    """Raised when mission execution fails"""
    pass

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except RobotError as e:
            # Log robot-specific errors
            logging.error(f"Robot error: {e.message}, Code: {e.error_code}, Details: {e.details}")

            # Report to error tracking
            if sentry_sdk:
                sentry_sdk.capture_exception(e)

            # Return appropriate error response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=400,
                content={
                    "error": e.error_code,
                    "message": e.message,
                    "details": e.details
                }
            )
        except Exception as e:
            # Log unexpected errors
            logging.error(f"Unexpected error: {str(e)}")
            logging.error(traceback.format_exc())

            # Report to error tracking
            if sentry_sdk:
                sentry_sdk.capture_exception(e)

            # Return generic error response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": "An internal error occurred"
                }
            )

# Add middleware to FastAPI app
# app.add_middleware(ErrorHandlingMiddleware)

class RobotHealthMonitor:
    def __init__(self):
        self.uptime = {}
        self.error_counts = {}
        self.performance_metrics = {}

    def record_robot_uptime(self, robot_id: str):
        """Record robot uptime"""
        if robot_id not in self.uptime:
            self.uptime[robot_id] = {"start_time": time.time(), "total_uptime": 0}

        current_uptime = time.time() - self.uptime[robot_id]["start_time"]
        self.uptime[robot_id]["current_uptime"] = current_uptime

    def record_error(self, robot_id: str, error_type: str):
        """Record robot error"""
        if robot_id not in self.error_counts:
            self.error_counts[robot_id] = {}

        if error_type not in self.error_counts[robot_id]:
            self.error_counts[robot_id][error_type] = 0

        self.error_counts[robot_id][error_type] += 1

    def get_health_report(self, robot_id: str) -> Dict[str, Any]:
        """Get health report for robot"""
        report = {
            "robot_id": robot_id,
            "status": "healthy",
            "uptime": self.uptime.get(robot_id, {}).get("current_uptime", 0),
            "error_count": sum(self.error_counts.get(robot_id, {}).values()),
            "error_types": self.error_counts.get(robot_id, {}),
            "timestamp": datetime.utcnow().isoformat()
        }

        # Determine health status based on error count
        error_count = report["error_count"]
        if error_count > 10:
            report["status"] = "unhealthy"
        elif error_count > 5:
            report["status"] = "warning"

        return report
```

## Best Practices
- Implement proper connection pooling for database operations
- Use message queues for handling high-volume robot communications
- Implement circuit breakers for external service calls
- Use caching strategically to reduce database load
- Monitor robot health and performance metrics
- Implement proper error handling and logging
- Use async/await for I/O-bound operations
- Implement rate limiting for API endpoints
- Use proper authentication and authorization
- Implement data retention policies for historical data