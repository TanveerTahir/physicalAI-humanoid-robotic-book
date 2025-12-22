---
title: URDF for Humanoids
sidebar_position: 4
description: Understanding Unified Robot Description Format for humanoid robot modeling and simulation
---

# URDF for Humanoids

## Conceptual Overview

Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models, including their physical properties, kinematic structure, and visual representation. For humanoid robots, URDF provides the foundation for simulation, visualization, and motion planning.

### What is URDF?

URDF (Unified Robot Description Format) is an XML-based format that describes robot models in terms of:

- **Physical Structure**: Links (rigid bodies) and joints (kinematic constraints)
- **Geometric Properties**: Shape, size, and visual appearance
- **Inertial Properties**: Mass, center of mass, and inertia tensors
- **Kinematic Properties**: Joint limits, types, and motion constraints

### URDF in Humanoid Robotics

For humanoid robots, URDF is particularly important because:

- **Complex Kinematics**: Humanoids have many degrees of freedom
- **Dynamic Simulation**: Accurate physical properties are crucial
- **Collision Avoidance**: Precise geometric models are needed
- **Motion Planning**: Kinematic structure defines possible movements
- **Visualization**: Visual models for debugging and presentation

## System Architecture Explanation

### URDF Structure

```
Robot Model
├── Links (Rigid Bodies)
│   ├── Visual: How the link looks
│   ├── Collision: Collision geometry
│   └── Inertial: Physical properties
├── Joints (Kinematic Constraints)
│   ├── Joint Type: Revolute, prismatic, fixed, etc.
│   ├── Joint Limits: Range of motion constraints
│   ├── Axis: Direction of joint motion
│   └── Origin: Position relative to parent
└── Materials: Visual appearance definitions
```

### URDF Processing Pipeline

```
URDF File → Robot State Publisher → TF2 → Visualization/Planning/Simulation
```

1. **URDF File**: XML description of robot model
2. **Robot State Publisher**: Publishes joint states and transforms
3. **TF2**: Maintains coordinate frame relationships
4. **Applications**: Simulation, visualization, planning tools

### Key URDF Elements

- **`<link>`**: Represents a rigid body with visual, collision, and inertial properties
- **`<joint>`**: Defines connection between links with kinematic constraints
- **`<material>`**: Defines visual appearance properties
- **`<gazebo>`**: Gazebo-specific simulation properties
- **`<transmission>`**: Actuator interface definitions

## Workflow / Pipeline Description

### Creating a Humanoid URDF

1. **Design Robot Structure**: Plan links and joints for humanoid kinematics
2. **Define Links**: Create link elements with visual and collision geometry
3. **Define Joints**: Create joint elements connecting links with proper kinematics
4. **Add Inertial Properties**: Define mass, center of mass, and inertia tensors
5. **Set Joint Limits**: Define range of motion and constraints
6. **Validate URDF**: Check for errors and kinematic consistency
7. **Test in Simulation**: Verify functionality in Gazebo or other simulators

### URDF Development Workflow

```xml
<!-- 1. Define robot root link -->
<link name="base_link">
  <!-- 2. Add visual and collision properties -->
  <visual>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
</link>
```

### Typical Humanoid Joint Hierarchy

```
base_link (torso)
├── left_hip_joint → left_hip_link
│   ├── left_knee_joint → left_knee_link
│   └── left_ankle_joint → left_foot_link
├── right_hip_joint → right_hip_link
│   ├── right_knee_joint → right_knee_link
│   └── right_ankle_joint → right_foot_link
├── left_shoulder_joint → left_shoulder_link
│   ├── left_elbow_joint → left_elbow_link
│   └── left_wrist_joint → left_hand_link
├── right_shoulder_joint → right_shoulder_link
│   ├── right_elbow_joint → right_elbow_link
│   └── right_wrist_joint → right_hand_link
└── neck_joint → head_link
```

## Constraints & Failure Modes

### URDF Constraints

- **Kinematic Consistency**: Joint connections must form a valid kinematic tree
- **Physical Validity**: Inertial properties must be physically realistic
- **Computational Complexity**: Large models may impact simulation performance
- **File Size**: Complex models can result in large URDF files
- **Validation Requirements**: URDF must pass validation checks

### Common Failure Modes

1. **Invalid Kinematics**: Joint loops or disconnected components
2. **Missing Inertial Properties**: Simulation instability
3. **Incorrect Joint Limits**: Unsafe robot motion
4. **Collision Issues**: Poor collision geometry causing simulation errors
5. **Performance Problems**: Too complex models for real-time simulation
6. **Visualization Errors**: Incorrect visual geometry or materials
7. **Transform Issues**: Invalid coordinate frame relationships

### Humanoid-Specific Challenges

- **Complex Kinematics**: Many degrees of freedom requiring careful design
- **Balance Requirements**: Center of mass considerations for stability
- **Collision Detection**: Complex geometries for collision avoidance
- **Motion Planning**: High-dimensional configuration space
- **Actuator Constraints**: Realistic joint torque and velocity limits

### Mitigation Strategies

- **Incremental Development**: Build URDF step by step, testing at each stage
- **Validation Tools**: Use check_urdf and other validation tools
- **Simulation Testing**: Test in simulation before physical deployment
- **Iterative Refinement**: Continuously improve model accuracy
- **Documentation**: Maintain clear documentation of model assumptions
- **Modular Design**: Break complex models into manageable components

## Simulation vs Real-World Notes

### Simulation Considerations
- Accurate inertial properties for realistic dynamics
- Proper collision geometry for contact simulation
- Realistic joint friction and damping parameters
- Gazebo-specific extensions for advanced simulation
- Validation against real robot behavior

### Real-World Implementation
- Physical robot calibration and model refinement
- Integration with hardware drivers
- Safety considerations for joint limits
- Performance optimization for real-time control
- Hardware-specific modifications

### Best Practices
- Start with simple models and add complexity gradually
- Use mesh files for complex geometries
- Include realistic inertial properties
- Define appropriate joint limits and safety constraints
- Test thoroughly in simulation before real-world use
- Maintain consistency between URDF and actual hardware

---

*Next: Explore [Gazebo Physics & Environment Simulation](../module2-simulation/gazebo-physics-environment.md) to understand robotic simulation environments.*