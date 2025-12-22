# ADR-2: Simulation Platform Strategy

Date: 2025-12-22

## Status

Accepted

## Context

The textbook needs to teach Physical AI & Humanoid Robotics concepts that involve simulation environments. We need to choose which simulation platforms to focus on for educational purposes, considering the learning curve, ROS 2 integration, and real-world applicability. Students need to understand both basic simulation concepts and advanced simulation techniques for complex robotics applications.

## Decision

We will use a tiered approach:
- **Gazebo**: For foundational learning due to its tight ROS 2 integration and open-source nature
- **Isaac Sim**: For advanced topics, especially when NVIDIA hardware integration is relevant
- **Unity**: For visualization and cross-platform simulation concepts

This strategy allows students to start with accessible tools and progress to more sophisticated platforms as their understanding deepens.

## Consequences

**Positive:**
- Students can start with accessible, well-documented tools
- Progression from basic to advanced simulation platforms
- Exposure to industry-standard tools used in robotics
- Strong ROS 2 integration for foundational learning
- Advanced capabilities for complex scenarios

**Negative:**
- Students need to learn multiple simulation environments
- Potential confusion when switching between platforms
- Different learning curves for each platform
- More complex documentation and examples to maintain

## Alternatives

**Single Platform Only**: Focus on just one simulation environment (e.g., only Gazebo). Rejected because it would limit students' exposure to industry tools and capabilities.

**Isaac Sim First**: Start with Isaac Sim as the primary platform. Rejected because of its steeper learning curve and hardware requirements for beginners.

**Unity Only**: Use Unity as the primary simulation platform. Rejected because it has weaker ROS 2 integration compared to Gazebo for foundational robotics learning.

## References

- specs/master/plan.md
- specs/master/research.md
- specs/master/spec.md
