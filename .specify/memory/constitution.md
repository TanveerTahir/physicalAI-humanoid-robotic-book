<!-- SYNC IMPACT REPORT
Version change: 1.0.1 -> 2.0.0
List of modified principles: Mission & Purpose → Technical Accuracy, Audience Definition → Embodied Intelligence First, Pedagogical Philosophy → Learn-by-Building, Style & Tone → AI-Native by Design, Structural Requirements → Clarity for Advanced Learners, Technical Platform Standards updated
Added sections: Content & Knowledge Standards, Sources & References Standards, AI & RAG Constraints, Writing & Structure Standards, Technical Constraints, Ethical & Safety Constraints, Success Criteria sections
Removed sections: None
Templates requiring updates: ✅ Updated /sp.plan template, ⚠ Pending /sp.spec template, ⚠ Pending /sp.tasks template
Follow-up TODOs: None
-->

# AI-Native Textbook for Teaching Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Technical Accuracy
All robotics, AI, simulation, and hardware concepts must be correct, verifiable, and aligned with current industry and academic standards (ROS 2, Gazebo, NVIDIA Isaac, VLA systems). This principle is non-negotiable and all content must be factually accurate and verifiable through official documentation and standards.

### II. Embodied Intelligence First
Emphasize the transition from purely digital AI to AI systems operating under real-world physical constraints (physics, sensors, latency, energy, safety). Content must clearly distinguish between simulation-only workflows and real-world implementations, with explicit attention to physical constraints.

### III. Learn-by-Building
Concepts should be taught through practical workflows: simulations, ROS 2 nodes, perception pipelines, and capstone-style integrations. Every chapter must include hands-on examples that allow learners to build and experiment with the concepts discussed.

### IV. AI-Native by Design
The book is not static documentation. It must be designed for: Retrieval-Augmented Generation (RAG), Agent-assisted learning, Personalization and translation, Continuous extension via agents and subagents. The content structure must support these AI-native features.

### V. Clarity for Advanced Learners
Target audience includes advanced students, engineers, and startup-oriented builders. Explanations must be precise, structured, and free of unnecessary abstraction. Content must be technically rigorous while remaining accessible to the target demographic.

## Standards

### Content & Knowledge Standards
- All factual and technical claims must be verifiable
- Robotics concepts must align with ROS 2 (Humble/Iron) standards
- Simulation content must reflect Gazebo, Unity, and Isaac Sim realities
- Hardware recommendations must be realistic and clearly justified
- Clear separation between:
  - Simulation-only workflows
  - Sim-to-Real workflows
  - Fully physical deployments

### Sources & References Standards
- Prefer official documentation, standards, and vendor references:
  - ROS 2 documentation
  - NVIDIA Isaac / Jetson docs
  - Gazebo / Unity manuals
  - Peer-reviewed robotics and AI research where applicable
- External sources must be clearly attributed
- No unverified claims or "black box" explanations

### AI & RAG Constraints
- The RAG chatbot must:
  - Answer strictly from indexed book content when required
  - Support user-selected text grounding
  - Avoid hallucinations beyond the knowledge base
- Agent behaviors must be deterministic where possible and auditable

### Writing & Structure Standards
- Modular chapter design aligned with course modules
- Each chapter should include:
  - Conceptual overview
  - System architecture explanation
  - Practical workflow or example
  - Common failure modes and constraints
- Language level: clear technical English (approx. Flesch-Kincaid 11–13)
- No filler content; every section must serve learning or implementation

### Technical Constraints
- Documentation framework: Docusaurus
- Deployment: GitHub Pages or Vercel
- Backend stack (for RAG):
  - FastAPI
  - OpenAI Agents / ChatKit SDK
  - Neon Serverless Postgres
  - Qdrant (vector database)
- Code examples must be:
  - Minimal
  - Runnable in principle
  - Clearly scoped and explained

## Ethical & Safety Constraints

- No unsafe robotics instructions without explicit warnings
- Clearly state simulation vs real-world risks
- Avoid instructions that could cause physical harm if misapplied
- Respect licensing of all referenced tools and frameworks

## Development Workflow

### Engineering Process Rules
Spec-driven development must be strictly followed: /sp.specify → /sp.plan → /sp.task → /sp.implement. Code or text must never be generated without a prior specification. All assets must be version-controlled via GitHub. All documentation must be placed inside the repo under /docs.

### Assessment Philosophy
Capstone-style mini-projects conclude every major module. Small exercises guide students toward the final humanoid-capstone. Projects must be reproducible locally or in cloud via recommended configurations.

### RAG and Knowledge Retrieval Quality Rules
Textbook content must be chunked logically for precise retrieval. No chapter may exceed retrieval granularity constraints. The chatbot must always prioritize grounded responses from the book.

## Success Criteria

### Content Quality
- All modules fully documented and internally consistent
- Clear learning progression from Physical AI fundamentals to VLA systems
- Capstone project is coherent, realistic, and technically feasible

### AI Integration
- RAG chatbot retrieves accurate, relevant sections
- No hallucinated explanations beyond indexed content
- Demonstrable usefulness for learning and navigation

### Engineering Quality
- Clean repository structure
- Reproducible build and deployment
- Clear separation of content, infrastructure, and agents

### Hackathon Readiness
- Public GitHub repository
- Live deployed book
- Sub-90-second demo clearly showing:
  - Book
  - RAG chatbot
  - AI-native features

## Governance

Personalization requires stored learner metadata (using BetterAuth at signup). Urdu translation is performed dynamically (caching allowed). Both features must be triggered via button UI in chapters. Deployment rules: Book must be deployed publicly using GitHub Pages or Vercel. Backend must expose documented endpoints for: auth, embeddings, RAG, personalization, translation. Environment variables must be managed securely. Code must be readable and documented. Diagrams, schemas, and URDF snippets must be included openly where possible. Readers should be able to replicate the entire learning process.

**Version**: 2.0.0 | **Ratified**: 2025-12-10 | **Last Amended**: 2025-12-22
