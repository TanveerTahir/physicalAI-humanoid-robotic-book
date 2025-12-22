# ADR-4: Frontend-Backend Architecture Division

Date: 2025-12-22

## Status

Accepted

## Context

The AI-native textbook requires both content delivery and AI-powered services. We need to determine how to divide responsibilities between frontend and backend to optimize for performance, security, user experience, and maintainability. The system needs to handle authentication, RAG queries, personalization, and translation while providing a responsive educational interface.

## Decision

We will implement a hybrid architecture where:
- **Frontend** (Docusaurus/React): Handles user interface, content presentation, basic interactions, and user experience
- **Backend** (FastAPI): Handles authentication, RAG queries, embedding generation, personalization logic, and translation services

This division ensures sensitive operations and complex processing happen server-side while maintaining a responsive user interface.

## Consequences

**Positive:**
- Sensitive operations (RAG, authentication) are properly secured on backend
- Frontend remains responsive and optimized for content delivery
- Clear separation of concerns between presentation and business logic
- Scalable architecture with backend services that can be scaled independently
- Proper handling of vector embeddings and complex AI operations

**Negative:**
- More complex architecture with additional network calls
- Requires managing multiple services and deployments
- Potential latency for AI-powered features
- More complex error handling across service boundaries

## Alternatives

**Frontend-Heavy**: More processing on client-side for better UX. Rejected because it would expose sensitive operations and AI models to clients.

**Backend-Heavy**: Server-side rendering for all content. Rejected because it would reduce performance and user experience for content consumption.

**Microservices**: Separate services for each function. Rejected because it would add unnecessary complexity for this project scope.

## References

- specs/master/plan.md
- specs/master/research.md
- specs/master/data-model.md
- specs/master/contracts/api-contracts.md
