import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Textbook sidebar with structured modules
  textbookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: Physical AI & Embodied Intelligence',
      items: [
        'module1-foundations/physical-ai-embodied-intelligence',
        'module1-foundations/sensors-physical-constraints',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: ROS 2 - Robotic Nervous System',
      items: [
        'module2-ros2/ros2-architecture-concepts',
        'module2-ros2/nodes-topics-services-actions',
        'module2-ros2/python-agents-ros',
        'module2-ros2/urdf-humanoids',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: Simulation & Digital Twins',
      items: [
        'module2-simulation/gazebo-physics-environment',
        'module2-simulation/sensor-simulation',
        'module2-simulation/unity-visualization-hri',
        'module2-simulation/sim-vs-real-constraints',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac & Sim-to-Real',
      items: [
        'module3-isaac/isaac-sim-synthetic-data',
        'module3-isaac/isaac-ros-hardware',
        'module3-isaac/vslam-navigation',
        'module3-isaac/sim-to-real-transfer',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4-vla/llms-robotics-overview',
        'module4-vla/voice-to-action',
        'module4-vla/cognitive-planning-llms',
        'module4-vla/ros2-action-translation',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Capstone: Autonomous Humanoid System',
      items: [
        'capstone-project/system-architecture',
        'capstone-project/walkthrough',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Hardware & Deployment',
      items: [
        'hardware-deployment/workstation-gpu-requirements',
        'hardware-deployment/jetson-edge-kit-setup',
        'hardware-deployment/physical-cloud-tradeoffs',
        'hardware-deployment/safety-latency-constraints',
      ],
      collapsed: false,
    },
  ],
};

export default sidebars;
