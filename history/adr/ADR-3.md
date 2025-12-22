# ADR-3: Content Chunking Strategy for RAG

Date: 2025-12-22

## Status

Accepted

## Context

The textbook content needs to be optimized for Retrieval-Augmented Generation (RAG) systems to enable accurate and contextual AI-powered responses. The way content is chunked affects both the precision and recall of the RAG system. Too small chunks may miss context, while too large chunks may dilute relevance. We need to balance retrieval precision with contextual completeness for effective learning support.

## Decision

We will use medium-sized content chunks of 200-500 words. This size provides an optimal balance between:
- Sufficient context for understanding within each chunk
- Precision in retrieval for specific queries
- Manageable processing for the RAG system
- Readable sections for educational purposes

Each chunk will focus on a single concept or idea while maintaining semantic coherence.

## Consequences

**Positive:**
- Better precision in RAG responses compared to larger chunks
- Sufficient context for understanding compared to smaller chunks
- Good balance between retrieval performance and content completeness
- Appropriate size for educational content consumption
- Easier to maintain and update individual concepts

**Negative:**
- May occasionally split related concepts across chunks
- Requires careful editorial planning to maintain flow
- Potential for missing broader context in complex topics
- More chunks to manage and maintain

## Alternatives

**Short Chunks (50-150 words)**: Higher precision but risk of missing context. Rejected because it would fragment educational content and make learning more difficult.

**Long Chunks (500-1000 words)**: Better context but reduced retrieval precision. Rejected because it would dilute relevance and make RAG responses less focused.

**Variable Chunks**: Different sizes based on content type. Rejected because it would add complexity without clear benefits over the medium-sized approach.

## References

- specs/master/plan.md
- specs/master/research.md
- specs/master/spec.md
