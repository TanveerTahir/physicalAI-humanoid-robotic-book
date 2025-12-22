# Frontend Development for Robotics Applications

## Overview
Frontend development for robotics involves creating user interfaces that allow humans to interact with, monitor, and control robotic systems. This includes dashboards, teleoperation interfaces, visualization tools, and configuration panels that bridge the gap between complex robotic systems and human operators.

## Key Technologies
- **React/Vue/Angular**: Modern UI frameworks
- **WebSocket/Socket.io**: Real-time communication with robots
- **Three.js/Babylon.js**: 3D visualization and simulation
- **D3.js/Chart.js**: Data visualization and analytics
- **ROSlibjs**: ROS communication from web browsers
- **WebRTC**: Real-time video streaming
- **Material UI/Tailwind**: UI component libraries

## Essential Frontend Techniques

### 1. Real-time Robot Dashboard
```jsx
import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const RobotDashboard = () => {
  const [robotData, setRobotData] = useState({
    position: { x: 0, y: 0, z: 0 },
    orientation: { roll: 0, pitch: 0, yaw: 0 },
    velocity: { linear: 0, angular: 0 },
    battery: 100,
    status: 'idle'
  });

  const [sensorHistory, setSensorHistory] = useState([]);
  const [socket, setSocket] = useState(null);

  useEffect(() => {
    // Connect to robot backend
    const newSocket = io('http://localhost:3001');
    setSocket(newSocket);

    // Listen for robot data updates
    newSocket.on('robot_data', (data) => {
      setRobotData(data);

      // Update sensor history for charts
      setSensorHistory(prev => {
        const newHistory = [...prev, {
          timestamp: Date.now(),
          battery: data.battery,
          linear_velocity: data.velocity.linear,
          angular_velocity: data.velocity.angular
        }];
        return newHistory.slice(-50); // Keep last 50 data points
      });
    });

    return () => {
      newSocket.close();
    };
  }, []);

  const sendCommand = (command) => {
    if (socket) {
      socket.emit('robot_command', command);
    }
  };

  return (
    <div className="dashboard">
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
        <div className="control-buttons">
          <button onClick={() => sendCommand({ type: 'move', direction: 'forward' })}>
            Forward
          </button>
          <button onClick={() => sendCommand({ type: 'move', direction: 'backward' })}>
            Backward
          </button>
          <button onClick={() => sendCommand({ type: 'rotate', direction: 'left' })}>
            Turn Left
          </button>
          <button onClick={() => sendCommand({ type: 'rotate', direction: 'right' })}>
            Turn Right
          </button>
          <button onClick={() => sendCommand({ type: 'emergency_stop' })} className="emergency">
            Emergency Stop
          </button>
        </div>
      </div>

      <div className="visualization-panel">
        <h2>Sensor Data</h2>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
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
      </div>
    </div>
  );
};

export default RobotDashboard;
```

### 2. 3D Robot Visualization
```jsx
import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const Robot3DViewer = ({ robotState, sensorData }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const robotRef = useRef(null);
  const controlsRef = useRef(null);

  useEffect(() => {
    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    mountRef.current.appendChild(renderer.domElement);

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Create robot model
    const robotGroup = new THREE.Group();
    robotRef.current = robotGroup;

    // Robot body
    const bodyGeometry = new THREE.BoxGeometry(1, 0.5, 0.5);
    const bodyMaterial = new THREE.MeshPhongMaterial({ color: 0x00aaff });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.castShadow = true;
    robotGroup.add(body);

    // Robot wheels
    const wheelGeometry = new THREE.CylinderGeometry(0.2, 0.2, 0.1, 16);
    wheelGeometry.rotateZ(Math.PI / 2);

    const wheelPositions = [
      [0.6, -0.3, 0.3],   // front right
      [0.6, -0.3, -0.3],  // front left
      [-0.6, -0.3, 0.3],  // back right
      [-0.6, -0.3, -0.3]  // back left
    ];

    wheelPositions.forEach(pos => {
      const wheel = new THREE.Mesh(wheelGeometry, bodyMaterial);
      wheel.position.set(...pos);
      wheel.castShadow = true;
      robotGroup.add(wheel);
    });

    scene.add(robotGroup);

    // Add floor
    const floorGeometry = new THREE.PlaneGeometry(20, 20);
    const floorMaterial = new THREE.MeshPhongMaterial({
      color: 0xdddddd,
      side: THREE.DoubleSide
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Add orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);

      // Update robot position based on state
      if (robotState && robotGroup) {
        robotGroup.position.x = robotState.position.x;
        robotGroup.position.y = robotState.position.y;
        robotGroup.position.z = robotState.position.z;

        // Apply rotation based on orientation
        robotGroup.rotation.y = robotState.orientation.yaw;
      }

      renderer.render(scene, camera);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update robot state when props change
  useEffect(() => {
    if (robotRef.current && robotState) {
      robotRef.current.position.set(
        robotState.position.x,
        robotState.position.y,
        robotState.position.z
      );
      robotRef.current.rotation.y = robotState.orientation.yaw;
    }
  }, [robotState]);

  return <div ref={mountRef} style={{ width: '100%', height: '500px' }} />;
};

export default Robot3DViewer;
```

### 3. Teleoperation Interface
```jsx
import React, { useState, useEffect, useRef } from 'react';
import { Joystick } from 'react-joystick-component';

const TeleoperationInterface = ({ onCommand }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [videoStream, setVideoStream] = useState(null);
  const [commandHistory, setCommandHistory] = useState([]);
  const [activeMode, setActiveMode] = useState('manual'); // manual, autonomous, assistive

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    // Initialize video stream if available
    if (navigator.mediaDevices && activeMode !== 'autonomous') {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            setVideoStream(stream);
          }
        })
        .catch(err => {
          console.error('Error accessing video stream:', err);
        });
    }

    return () => {
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [activeMode]);

  const handleJoystickMove = (event) => {
    if (activeMode === 'manual') {
      const command = {
        type: 'teleop',
        linear: event.y * 0.5, // Scale for safety
        angular: event.x * 0.5,
        timestamp: Date.now()
      };

      onCommand(command);
      addToHistory(command);
    }
  };

  const handleKeyboardControl = (event) => {
    if (activeMode !== 'manual') return;

    let linear = 0;
    let angular = 0;

    switch (event.key) {
      case 'ArrowUp':
        linear = 0.5;
        break;
      case 'ArrowDown':
        linear = -0.5;
        break;
      case 'ArrowLeft':
        angular = 0.5;
        break;
      case 'ArrowRight':
        angular = -0.5;
        break;
      default:
        return;
    }

    event.preventDefault();

    const command = {
      type: 'teleop',
      linear,
      angular,
      timestamp: Date.now()
    };

    onCommand(command);
    addToHistory(command);
  };

  const addToHistory = (command) => {
    setCommandHistory(prev => [
      command,
      ...prev.slice(0, 99) // Keep last 100 commands
    ]);
  };

  const executePredefinedAction = (action) => {
    const command = {
      type: 'predefined',
      action,
      timestamp: Date.now()
    };

    onCommand(command);
    addToHistory(command);
  };

  return (
    <div className="teleoperation-interface">
      <div className="control-header">
        <h2>Teleoperation Control</h2>
        <div className="connection-status">
          <span className={`status-indicator ${connectionStatus}`}>
            {connectionStatus}
          </span>
          <select
            value={activeMode}
            onChange={(e) => setActiveMode(e.target.value)}
            className="mode-selector"
          >
            <option value="manual">Manual Control</option>
            <option value="assistive">Assistive Control</option>
            <option value="autonomous">Autonomous Mode</option>
          </select>
        </div>
      </div>

      <div className="main-controls">
        <div className="video-section">
          <h3>Robot Camera Feed</h3>
          <div className="video-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: '100%', height: 'auto' }}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
        </div>

        <div className="joystick-section">
          <h3>Manual Control</h3>
          <div className="joystick-container">
            <Joystick
              size={150}
              throttle={100}
              move={handleJoystickMove}
              stop={() => {
                // Send stop command when joystick returns to center
                onCommand({
                  type: 'teleop',
                  linear: 0,
                  angular: 0,
                  timestamp: Date.now()
                });
              }}
            />
          </div>
        </div>
      </div>

      <div className="action-buttons">
        <h3>Quick Actions</h3>
        <div className="action-grid">
          <button
            onClick={() => executePredefinedAction('home_position')}
            disabled={activeMode !== 'manual'}
          >
            Go Home
          </button>
          <button
            onClick={() => executePredefinedAction('emergency_stop')}
            className="emergency"
          >
            Emergency Stop
          </button>
          <button
            onClick={() => executePredefinedAction('object_detection')}
            disabled={activeMode === 'autonomous'}
          >
            Detect Objects
          </button>
          <button
            onClick={() => executePredefinedAction('mapping')}
            disabled={activeMode !== 'manual'}
          >
            Start Mapping
          </button>
        </div>
      </div>

      <div className="command-history">
        <h3>Command History</h3>
        <div className="history-list">
          {commandHistory.slice(0, 10).map((cmd, index) => (
            <div key={index} className="command-item">
              <span className="command-type">{cmd.type}</span>
              <span className="command-details">
                {cmd.linear ? `L: ${cmd.linear.toFixed(2)}` : ''}
                {cmd.angular ? ` A: ${cmd.angular.toFixed(2)}` : ''}
              </span>
              <span className="command-time">
                {new Date(cmd.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="keyboard-hint">
        <p>Keyboard Controls: Arrow keys for movement (when manual mode active)</p>
      </div>
    </div>
  );
};

export default TeleoperationInterface;
```

### 4. Configuration and Settings Panel
```jsx
import React, { useState, useEffect } from 'react';

const RobotConfigurationPanel = ({ robotId, onSave, onReset }) => {
  const [config, setConfig] = useState({
    navigation: {
      max_velocity: 0.5,
      min_turn_radius: 0.3,
      obstacle_threshold: 0.5,
      inflation_radius: 0.5
    },
    sensors: {
      lidar_range: 10.0,
      camera_resolution: '720p',
      imu_frequency: 100,
      battery_warning_level: 20
    },
    safety: {
      max_current: 10.0,
      max_temperature: 60.0,
      emergency_stop_timeout: 5.0,
      collision_threshold: 50.0
    }
  });

  const [isDirty, setIsDirty] = useState(false);

  const handleConfigChange = (category, key, value) => {
    setConfig(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value
      }
    }));
    setIsDirty(true);
  };

  const handleSave = () => {
    onSave(robotId, config);
    setIsDirty(false);
  };

  const handleReset = () => {
    onReset(robotId);
    setIsDirty(false);
  };

  const renderNumberInput = (category, key, label, min, max, step = 0.1) => (
    <div className="config-row">
      <label>{label}</label>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={config[category][key]}
        onChange={(e) => handleConfigChange(category, key, parseFloat(e.target.value))}
      />
    </div>
  );

  const renderSelectInput = (category, key, label, options) => (
    <div className="config-row">
      <label>{label}</label>
      <select
        value={config[category][key]}
        onChange={(e) => handleConfigChange(category, key, e.target.value)}
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="configuration-panel">
      <div className="panel-header">
        <h2>Robot Configuration</h2>
        <div className="action-buttons">
          <button onClick={handleSave} disabled={!isDirty}>
            Save Changes
          </button>
          <button onClick={handleReset}>
            Reset to Defaults
          </button>
        </div>
      </div>

      <div className="config-tabs">
        <div className="tab-content">
          <h3>Navigation Settings</h3>
          <div className="config-section">
            {renderNumberInput('navigation', 'max_velocity', 'Max Velocity (m/s)', 0, 2, 0.1)}
            {renderNumberInput('navigation', 'min_turn_radius', 'Min Turn Radius (m)', 0, 2, 0.1)}
            {renderNumberInput('navigation', 'obstacle_threshold', 'Obstacle Threshold (m)', 0, 5, 0.1)}
            {renderNumberInput('navigation', 'inflation_radius', 'Inflation Radius (m)', 0, 2, 0.1)}
          </div>

          <h3>Sensor Settings</h3>
          <div className="config-section">
            {renderNumberInput('sensors', 'lidar_range', 'LIDAR Range (m)', 1, 50, 0.5)}
            {renderSelectInput('sensors', 'camera_resolution', 'Camera Resolution', [
              { value: '480p', label: '480p' },
              { value: '720p', label: '720p' },
              { value: '1080p', label: '1080p' }
            ])}
            {renderNumberInput('sensors', 'imu_frequency', 'IMU Frequency (Hz)', 10, 1000, 10)}
            {renderNumberInput('sensors', 'battery_warning_level', 'Battery Warning (%)', 5, 50, 1)}
          </div>

          <h3>Safety Settings</h3>
          <div className="config-section">
            {renderNumberInput('safety', 'max_current', 'Max Current (A)', 1, 50, 0.5)}
            {renderNumberInput('safety', 'max_temperature', 'Max Temperature (°C)', 30, 100, 1)}
            {renderNumberInput('safety', 'emergency_stop_timeout', 'Emergency Stop Timeout (s)', 1, 30, 0.5)}
            {renderNumberInput('safety', 'collision_threshold', 'Collision Threshold (N)', 10, 200, 1)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RobotConfigurationPanel;
```

## Best Practices for Robotics Frontend Development

### 1. Performance Optimization
```jsx
import React, { memo, useMemo, useCallback } from 'react';
import { throttle } from 'lodash';

// Memoized components for performance
const SensorDataDisplay = memo(({ sensorData }) => {
  return (
    <div className="sensor-data">
      {Object.entries(sensorData).map(([key, value]) => (
        <div key={key} className="sensor-item">
          <span className="sensor-label">{key}:</span>
          <span className="sensor-value">{value}</span>
        </div>
      ))}
    </div>
  );
});

// Throttled event handlers to prevent excessive updates
const useThrottledCallback = (callback, delay) => {
  return useCallback(throttle(callback, delay), [callback, delay]);
};

// Efficient data processing
const useProcessedData = (rawData) => {
  return useMemo(() => {
    // Process and format data efficiently
    return rawData.map(item => ({
      ...item,
      processedValue: item.value * 100 // Example processing
    }));
  }, [rawData]);
};

// Virtualized lists for large datasets
const VirtualizedSensorList = ({ items, itemHeight = 50 }) => {
  const [scrollTop, setScrollTop] = useState(0);
  const visibleStart = Math.floor(scrollTop / itemHeight);
  const visibleEnd = Math.min(
    visibleStart + Math.ceil(500 / itemHeight) + 1, // 500px container height
    items.length
  );

  const visibleItems = items.slice(visibleStart, visibleEnd);

  return (
    <div
      className="virtual-list"
      style={{ height: '500px', overflow: 'auto' }}
      onScroll={(e) => setScrollTop(e.target.scrollTop)}
    >
      <div style={{ height: items.length * itemHeight, position: 'relative' }}>
        <div
          style={{
            position: 'absolute',
            top: visibleStart * itemHeight,
            height: visibleItems.length * itemHeight
          }}
        >
          {visibleItems.map((item, index) => (
            <div key={item.id} style={{ height: itemHeight }}>
              {item.name}: {item.value}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
```

### 2. Error Handling and State Management
```jsx
import React, { createContext, useContext, useReducer } from 'react';

// Error boundary for robot UI components
class RobotUIErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Robot UI Error:', error, errorInfo);
    // Log to error reporting service
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong with the robot interface.</h2>
          <button onClick={() => this.setState({ hasError: false, error: null })}>
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Robot state context
const RobotContext = createContext();

const robotReducer = (state, action) => {
  switch (action.type) {
    case 'CONNECT':
      return { ...state, isConnected: true, status: 'connected' };
    case 'DISCONNECT':
      return { ...state, isConnected: false, status: 'disconnected' };
    case 'UPDATE_SENSOR_DATA':
      return { ...state, sensorData: { ...state.sensorData, ...action.payload } };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    default:
      return state;
  }
};

export const RobotProvider = ({ children }) => {
  const [state, dispatch] = useReducer(robotReducer, {
    isConnected: false,
    status: 'disconnected',
    sensorData: {},
    error: null
  });

  return (
    <RobotContext.Provider value={{ state, dispatch }}>
      {children}
    </RobotContext.Provider>
  );
};

export const useRobot = () => {
  const context = useContext(RobotContext);
  if (!context) {
    throw new Error('useRobot must be used within a RobotProvider');
  }
  return context;
};
```

## Best Practices
- Implement real-time data streaming efficiently to prevent UI lag
- Use proper error boundaries to handle robot communication failures
- Provide clear visual feedback for robot states and actions
- Implement keyboard shortcuts for common operations
- Use responsive design for different screen sizes and devices
- Implement proper authentication and authorization for robot access
- Provide offline capability for critical functions
- Use consistent design patterns across all robot interfaces