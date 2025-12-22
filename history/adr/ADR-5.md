# ADR-5: ROS 2 Distribution Selection

Date: 2025-12-22

## Status

Accepted

## Context

The textbook needs to teach ROS 2 concepts for Physical AI & Humanoid Robotics. We must select which ROS 2 distribution to standardize on for examples and educational content. This choice affects compatibility with hardware, availability of packages, community support, and long-term maintenance. Students need to learn a version that is stable and well-supported for their future careers.

## Decision

We will standardize on ROS 2 Humble Hawksbill (LTS - Long Term Support) with 5 years of support (2022-2027). This distribution provides:
- Long-term support and stability for educational use
- Broad hardware compatibility
- Extensive community resources and documentation
- Maturity for reliable learning experiences
- Sufficient features for advanced robotics concepts

## Consequences

**Positive:**
- Long-term support ensures stability throughout the textbook lifecycle
- Extensive documentation and community resources for students
- Broad hardware compatibility for practical exercises
- Proven stability for educational environments
- Strong ecosystem of packages and tools

**Negative:**
- May miss newer features available in more recent distributions
- Students may need to adapt to newer versions in industry
- Some cutting-edge research may use newer distributions
- Potential for technology lag as newer versions emerge

## Alternatives

**ROS 2 Iron Irwini**: Newer features but shorter support cycle. Rejected because it lacks the long-term stability needed for educational content.

**Multiple Distributions**: Teaching multiple versions simultaneously. Rejected because it would confuse students and increase content complexity.

**Rolling Release**: Using the latest development version. Rejected because of instability and lack of long-term support for educational use.

## References

- specs/master/plan.md
- specs/master/research.md
- specs/master/spec.md
