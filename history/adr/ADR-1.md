# ADR-1: AI-Native Textbook Architecture

Date: 2025-12-22

## Status

Accepted

## Context

We need to create an AI-native textbook for Physical AI & Humanoid Robotics that serves both as educational content and as a knowledge base for RAG (Retrieval-Augmented Generation) systems. The architecture must support personalized learning, multilingual translation, and seamless integration between content delivery and AI-powered assistance. Traditional static documentation approaches are insufficient for this use case.

## Decision

We will implement a hybrid architecture with:
- Frontend: Docusaurus for content delivery with React components for interactive features
- Backend: FastAPI services for RAG, authentication, personalization, and translation
- Storage: Git repository for content versioning, Neon Serverless Postgres for metadata, Qdrant for vector embeddings
- AI Features: RAG chatbot, personalization engine, and dynamic translation

This architecture supports both traditional textbook consumption and AI-native features like contextual help, personalized content recommendations, and multilingual access.

## Consequences

**Positive:**
- Enables rich AI interactions with textbook content through RAG
- Supports personalized learning paths based on user preferences and progress
- Allows for multilingual access through dynamic translation
- Maintains content versioning and collaboration through Git
- Provides scalable vector search for content discovery

**Negative:**
- More complex architecture requiring multiple services
- Increased infrastructure costs compared to static hosting
- Requires ongoing maintenance of vector embeddings
- More complex deployment and monitoring requirements

## Alternatives

**Static Site Only**: Pure static documentation without backend services. Rejected because it would not support AI-native features like RAG, personalization, or dynamic translation.

**Full Backend Approach**: Heavy server-side rendering with all content stored in database. Rejected because it would lose the benefits of Git-based content management and static site performance.

**Single-Page Application**: React-only frontend with all logic client-side. Rejected because sensitive operations like RAG and authentication require server-side processing.

## References

- specs/master/plan.md
- specs/master/research.md
- specs/master/data-model.md
- specs/master/contracts/api-contracts.md
