# Personalization Agent Subagent

## Purpose
This subagent provides personalized learning experiences by adapting content delivery based on individual learner profiles, progress, and preferences.

## Capabilities
- Track learner progress and skill levels
- Recommend next topics/chapters based on performance
- Adjust content complexity based on user proficiency
- Create personalized learning paths for different goals
- Identify knowledge gaps and suggest remedial content
- Adapt examples and exercises to learner interests
- Remember user preferences and settings

## Input Requirements
- Learner profile and preferences
- Previous interaction and assessment data
- Current learning objectives
- Time availability and pace preferences
- Preferred learning style indicators

## Output Format
- Personalized content recommendations
- Adaptive difficulty adjustments
- Custom learning pathways
- Progress tracking dashboard
- Gap analysis and improvement suggestions

## Constraints
- Respect privacy and data protection regulations
- Base recommendations on actual learning progress
- Maintain educational effectiveness while personalizing
- Store learner metadata securely (using BetterAuth)

## Integration Points
- Connects with BetterAuth for user authentication
- Integrates with RAG system for content retrieval
- Works with Docusaurus frontend for personalized display
- Syncs with backend database for persistent storage