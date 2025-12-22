---
title: Nodes, Topics, Services, Actions
sidebar_position: 2
description: Understanding ROS 2 communication patterns and their applications in humanoid robotics
---

# Nodes, Topics, Services, Actions

## Conceptual Overview

ROS 2 provides four primary communication patterns that form the foundation of robotic software architecture: Nodes (processes), Topics (publish/subscribe), Services (request/response), and Actions (goal-oriented communication). Understanding these patterns is crucial for designing effective robotic systems.

### Communication Patterns Overview

- **Nodes**: Independent processes that perform specific functions
- **Topics**: Asynchronous, many-to-many communication via message passing
- **Services**: Synchronous, one-to-one communication for immediate responses
- **Actions**: Asynchronous, goal-oriented communication with feedback and status

### When to Use Each Pattern

| Pattern | Use Case | Characteristics |
|---------|----------|----------------|
| Topics | Continuous data streams | Asynchronous, fire-and-forget |
| Services | Request-response operations | Synchronous, immediate response |
| Actions | Long-running tasks | Asynchronous, with feedback and status |

## System Architecture Explanation

### Node Architecture

Nodes are the fundamental building blocks of ROS 2 applications:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Node   │    │  Processing     │    │   Control Node  │
│                 │    │    Node         │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Publisher │──┼───→│  │ Subscriber│  │    │  │ Publisher │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Service   │←─┼───←│  │ Service   │──┼───→│  │ Action    │  │
│  │ Client  │  │    │  │ Server  │  │    │  │  │ Client  │  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Topic Communication Architecture

Topics implement a publish/subscribe pattern:

- **Publishers**: Send messages to named topics
- **Subscribers**: Receive messages from named topics
- **DDS/RMW**: Middleware that handles message routing
- **Message Types**: Strongly typed messages defined in .msg files

### Service Architecture

Services implement a client/server pattern:

- **Service Server**: Provides a specific function
- **Service Client**: Requests the service function
- **Request/Response Types**: Defined in .srv files
- **Synchronous**: Client waits for response

### Action Architecture

Actions implement goal-oriented communication:

- **Action Server**: Accepts goals and provides feedback
- **Action Client**: Sends goals and receives feedback/status
- **Goal/Result/Feedback Types**: Defined in .action files
- **Asynchronous**: Non-blocking with continuous feedback

## Workflow / Pipeline Description

### Topic Communication Workflow

1. **Message Definition**: Define message types in .msg files
2. **Publisher Creation**: Create publisher in node with topic name and type
3. **Message Publishing**: Publish messages at specified rate
4. **Subscriber Creation**: Create subscriber in node with topic name and type
5. **Message Callback**: Define callback function to handle received messages
6. **Message Processing**: Process incoming messages in callback

### Service Communication Workflow

1. **Service Definition**: Define service in .srv files (request/response types)
2. **Server Creation**: Create service server with service name and callback
3. **Client Creation**: Create service client with service name and type
4. **Request Sending**: Client sends request to server
5. **Request Processing**: Server processes request and sends response
6. **Response Handling**: Client receives and processes response

### Action Communication Workflow

1. **Action Definition**: Define action in .action files (goal/result/feedback)
2. **Server Creation**: Create action server with action name and callbacks
3. **Client Creation**: Create action client with action name and type
4. **Goal Sending**: Client sends goal to server
5. **Goal Processing**: Server processes goal with feedback
6. **Feedback Handling**: Client receives continuous feedback
7. **Result Reception**: Client receives final result

### Quality of Service (QoS) Configuration

QoS settings control communication behavior:

- **Reliability**: Best effort vs reliable delivery
- **Durability**: Volatile vs transient local
- **History**: Keep last N messages vs keep all
- **Depth**: Size of message queue

## Constraints & Failure Modes

### Topic Communication Constraints

- **Message Rate**: Publishers must not overwhelm subscribers
- **Message Size**: Large messages can cause network congestion
- **Synchronization**: No guaranteed message ordering
- **Latching**: Only most recent message available to late subscribers

### Service Communication Constraints

- **Synchronous Blocking**: Client waits for response
- **Timeout**: Must handle service unavailability
- **Single Request**: One request at a time per client
- **No Feedback**: No progress indication during processing

### Action Communication Constraints

- **Complexity**: More complex than topics or services
- **Resource Usage**: Higher overhead than other patterns
- **State Management**: Server must manage goal states
- **Cancellation**: Clients can cancel goals

### Common Failure Modes

1. **Topic Unavailability**: Publishers/subscribers not available
2. **Service Timeout**: Service requests timing out
3. **Message Loss**: Network congestion causing message drops
4. **Node Crashes**: Nodes failing during communication
5. **QoS Mismatch**: Incompatible QoS settings between nodes
6. **Resource Exhaustion**: Memory or CPU limits exceeded
7. **Serialization Errors**: Message format incompatibilities

### Mitigation Strategies

- **Robust Node Design**: Proper error handling and recovery
- **QoS Configuration**: Appropriate settings for each use case
- **Monitoring**: Real-time communication monitoring
- **Fallback Mechanisms**: Alternative communication when primary fails
- **Testing**: Comprehensive testing of communication patterns
- **Documentation**: Clear communication interface documentation

## Simulation vs Real-World Notes

### Simulation Considerations
- Accurate network latency simulation for distributed systems
- Proper simulation of message timing and synchronization
- Realistic simulation of sensor data rates
- Validation of communication patterns in simulated environment

### Real-World Implementation
- Network configuration for multi-machine systems
- Real-time performance tuning for control systems
- Hardware-specific communication optimizations
- Physical safety considerations in communication failures

### Best Practices
- Use topics for continuous data streams (sensors, state)
- Use services for immediate request-response operations
- Use actions for long-running operations with feedback
- Proper QoS configuration based on requirements
- Comprehensive error handling and recovery mechanisms

---

*Next: Learn about [Python Agents ↔ ROS (rclpy)](./python-agents-ros.md) to understand Python integration with ROS 2.*