---
id: 001
title: Amend Constitution
stage: constitution
date: 2025-12-22
surface: agent
model: Claude Sonnet 4.5
feature: none
branch: master
user: CLIMAX
command: /sp.constitution
labels: [constitution,update,physical-ai]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---
## Prompt

Project: AI-Native Textbook for Teaching Physical AI & Humanoid Robotics

Purpose:
Create a production-grade, AI-native technical textbook that teaches Physical AI & Humanoid Robotics through structured content, executable examples, simulations, and an embedded RAG chatbot. The book must support learning, experimentation, and personalization while remaining technically rigorous and reproducible.

---

Core Principles:

* **Technical Accuracy**
  All robotics, AI, simulation, and hardware concepts must be correct, verifiable, and aligned with current industry and academic standards (ROS 2, Gazebo, NVIDIA Isaac, VLA systems).

* **Embodied Intelligence First**
  Emphasize the transition from purely digital AI to AI systems operating under real-world physical constraints (physics, sensors, latency, energy, safety).

* **Learn-by-Building**
  Concepts should be taught through practical workflows: simulations, ROS 2 nodes, perception pipelines, and capstone-style integrations.

* **AI-Native by Design**
  The book is not static documentation. It must be designed for:

  * Retrieval-Augmented Generation (RAG)
  * Agent-assisted learning
  * Personalization and translation
  * Continuous extension via agents and subagents

* **Clarity for Advanced Learners**
  Target audience includes advanced students, engineers, and startup-oriented builders. Explanations must be precise, structured, and free of unnecessary abstraction.

---

Key Standards:

Content & Knowledge:

* All factual and technical claims must be verifiable
* Robotics concepts must align with ROS 2 (Humble/Iron) standards
* Simulation content must reflect Gazebo, Unity, and Isaac Sim realities
* Hardware recommendations must be realistic and clearly justified
* Clear separation between:

  * Simulation-only workflows
  * Sim-to-Real workflows
  * Fully physical deployments

Sources & References:

* Prefer official documentation, standards, and vendor references:

  * ROS 2 documentation
  * NVIDIA Isaac / Jetson docs
  * Gazebo / Unity manuals
  * Peer-reviewed robotics and AI research where applicable
* External sources must be clearly attributed
* No unverified claims or "black box" explanations

AI & RAG Constraints:

* The RAG chatbot must:

  * Answer strictly from indexed book content when required
  * Support user-selected text grounding
  * Avoid hallucinations beyond the knowledge base
* Agent behaviors must be deterministic where possible and auditable

---

Writing & Structure Standards:

* Modular chapter design aligned with course modules
* Each chapter should include:

  * Conceptual overview
  * System architecture explanation
  * Practical workflow or example
  * Common failure modes and constraints
* Language level: clear technical English (approx. Flesch-Kincaid 11–13)
* No filler content; every section must serve learning or implementation

---

Technical Constraints:

* Documentation framework: Docusaurus
* Deployment: GitHub Pages or Vercel
* Backend stack (for RAG):

  * FastAPI
  * OpenAI Agents / ChatKit SDK
  * Neon Serverless Postgres
  * Qdrant (vector database)
* Code examples must be:

  * Minimal
  * Runnable in principle
  * Clearly scoped and explained

---

Ethical & Safety Constraints:

* No unsafe robotics instructions without explicit warnings
* Clearly state simulation vs real-world risks
* Avoid instructions that could cause physical harm if misapplied
* Respect licensing of all referenced tools and frameworks

---

Success Criteria:

Content Quality:

* All modules fully documented and internally consistent
* Clear learning progression from Physical AI fundamentals to VLA systems
* Capstone project is coherent, realistic, and technically feasible

AI Integration:

* RAG chatbot retrieves accurate, relevant sections
* No hallucinated explanations beyond indexed content
* Demonstrable usefulness for learning and navigation

Engineering Quality:

* Clean repository structure
* Reproducible build and deployment
* Clear separation of content, infrastructure, and agents

Hackathon Readiness:

* Public GitHub repository
* Live deployed book
* Sub-90-second demo clearly showing:

  * Book
  * RAG chatbot
  * AI-native features

## Response snapshot

Constitution updated to version 2.0.0 with the following changes:
- Version change: 1.0.1 -> 2.0.0 (major update due to principle changes)
- List of modified principles: Mission & Purpose → Technical Accuracy, Audience Definition → Embodied Intelligence First, Pedagogical Philosophy → Learn-by-Building, Style & Tone → AI-Native by Design, Structural Requirements → Clarity for Advanced Learners, Technical Platform Standards updated
- Added sections: Content & Knowledge Standards, Sources & References Standards, AI & RAG Constraints, Writing & Structure Standards, Technical Constraints, Ethical & Safety Constraints, Success Criteria sections
- Removed sections: None
- Templates requiring updates: ✅ Updated /sp.plan template, ⚠ Pending /sp.spec template, ⚠ Pending /sp.tasks template
- Follow-up TODOs: None

## Outcome

- ✅ Impact: Major constitution update with new principles and standards
- 🧪 Tests: No tests required for constitution update
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: /sp.specify to create feature specification
- 🧠 Reflection: Constitution now properly reflects the project's core values and technical requirements

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin feature specification process