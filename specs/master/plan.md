# Implementation Plan: Physical AI & Humanoid Robotics — AI-Native Textbook

**Branch**: `master` | **Date**: 2025-12-22 | **Spec**: [specs/master/spec.md](../master/spec.md)
**Input**: Feature specification from `/specs/master/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a complete AI-native textbook for Physical AI & Humanoid Robotics using Docusaurus, structured for RAG ingestion and personalized learning experiences. The textbook will cover modules from introduction to Physical AI through to a capstone project integrating all concepts. The content will be optimized for AI retrieval and designed with learn-by-building philosophy.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Markdown, JavaScript/TypeScript, Python 3.11 (for backend services)
**Primary Dependencies**: Docusaurus, React, Node.js, FastAPI, OpenAI SDK, Qdrant (vector database)
**Storage**: Git repository for content, Neon Serverless Postgres for metadata, Qdrant for vector embeddings
**Testing**: pytest for backend services, Jest for frontend components, manual validation for content accuracy
**Target Platform**: Web (GitHub Pages/Vercel deployment), with backend API for RAG and personalization
**Project Type**: Web application with documentation-focused frontend and AI service backend
**Performance Goals**: <200ms p95 response time for RAG queries, <2s page load times for documentation
**Constraints**: Content must be AI-retrievable, RAG must avoid hallucinations, deployment must be public
**Scale/Scope**: Target audience includes advanced students, engineers, and startup-oriented builders; textbook content organized in 5 modules with weekly chapters

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Gate 1: Technical Accuracy
- All robotics, AI, simulation, and hardware concepts must be correct, verifiable, and aligned with current industry and academic standards (ROS 2, Gazebo, NVIDIA Isaac, VLA systems)
- Content must be factually accurate and verifiable through official documentation and standards
- ✅ PASS: Will use official documentation and verified sources for all technical content

### Gate 2: Embodied Intelligence First
- Emphasize transition from purely digital AI to AI systems operating under real-world physical constraints
- Content must clearly distinguish between simulation-only workflows and real-world implementations
- ✅ PASS: Will explicitly mark simulation vs real-world constraints in all relevant sections

### Gate 3: Learn-by-Building
- Concepts should be taught through practical workflows: simulations, ROS 2 nodes, perception pipelines, and capstone-style integrations
- Every chapter must include hands-on examples that allow learners to build and experiment
- ✅ PASS: Will include practical examples and workflows in each chapter

### Gate 4: AI-Native by Design
- Book must be designed for RAG, Agent-assisted learning, Personalization and translation, Continuous extension via agents
- Content structure must support AI-native features
- ✅ PASS: Will structure content with clean chunk boundaries for RAG ingestion

### Gate 5: Clarity for Advanced Learners
- Target audience includes advanced students, engineers, and startup-oriented builders
- Explanations must be precise, structured, and free of unnecessary abstraction
- ✅ PASS: Will maintain technical rigor while remaining accessible to target demographic

### Gate 6: Content & Knowledge Standards
- All factual and technical claims must be verifiable
- Robotics concepts must align with ROS 2 (Humble/Iron) standards
- Clear separation between simulation-only, sim-to-real, and fully physical deployments
- ✅ PASS: Will ensure all claims are verifiable and properly categorized

### Gate 7: AI & RAG Constraints
- RAG chatbot must answer strictly from indexed book content without hallucinations
- Agent behaviors must be deterministic and auditable
- ✅ PASS: Will structure content to prevent hallucinations and ensure auditability

## Project Structure

### Documentation (this feature)

```text
specs/master/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Web application with documentation frontend and AI service backend
backend/
├── src/
│   ├── models/
│   │   ├── user.py         # User authentication and metadata
│   │   ├── content.py      # Textbook content models
│   │   └── rag.py          # RAG interaction models
│   ├── services/
│   │   ├── auth_service.py # Authentication service
│   │   ├── rag_service.py  # RAG query service
│   │   ├── embedding_service.py # Embedding generation
│   │   └── personalization_service.py # Personalization features
│   ├── api/
│   │   ├── v1/
│   │   │   ├── auth.py     # Authentication endpoints
│   │   │   ├── rag.py      # RAG query endpoints
│   │   │   ├── content.py  # Content endpoints
│   │   │   └── personalize.py # Personalization endpoints
│   │   └── main.py         # Main API router
│   └── utils/
│       ├── validators.py   # Request/response validators
│       └── helpers.py      # Utility functions
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── docs/                  # Textbook content in Docusaurus markdown
│   ├── intro/
│   ├── module1-foundations/
│   ├── module2-simulation/
│   ├── module3-perception/
│   ├── module4-integration/
│   └── capstone-project/
├── src/
│   ├── components/
│   │   ├── ChatInterface/  # RAG-powered chat component
│   │   ├── Personalization/ # Personalization controls
│   │   ├── Translation/    # Translation UI
│   │   └── BookNavigation/ # Navigation components
│   ├── pages/
│   ├── css/
│   └── utils/
├── docusaurus.config.js   # Docusaurus configuration
├── sidebars.js            # Navigation sidebar configuration
├── babel.config.js
├── package.json
└── README.md

.history/                 # Prompt History Records
├── prompts/
│   ├── general/
│   └── textbook/
└── adrs/                 # Architecture Decision Records

.skills/                  # Custom skills and subagents
├── quiz_generator/
├── translation_agent/
└── vision_slam_explainer/

.specify/                 # SpecKit Plus configuration
├── memory/               # Memory and constitution
├── templates/            # Template files
└── scripts/              # Automation scripts
```

**Structure Decision**: This is a web application with documentation-focused frontend (Docusaurus) and AI service backend (FastAPI) to support the AI-native textbook requirements. The frontend handles textbook presentation and user interaction while the backend manages RAG, authentication, personalization, and embeddings.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
