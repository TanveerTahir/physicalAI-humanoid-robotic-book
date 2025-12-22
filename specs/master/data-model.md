# Data Model: Physical AI & Humanoid Robotics Textbook

## Entities

### User
- **id**: string (UUID) - Unique identifier for the user
- **email**: string - User's email address
- **name**: string - User's full name
- **preferences**: JSON - Learning preferences and settings
- **progress**: JSON - Track user's progress through the textbook
- **created_at**: datetime - Account creation timestamp
- **updated_at**: datetime - Last update timestamp

### Content
- **id**: string (UUID) - Unique identifier for the content chunk
- **title**: string - Title of the content chunk
- **body**: string - The actual content in Markdown format
- **module**: string - Which module this content belongs to (e.g., "module1-foundations")
- **chapter**: string - Which chapter this content belongs to
- **section**: string - Which section within the chapter
- **chunk_type**: string - Type of content (e.g., "concept", "example", "exercise", "diagram")
- **embedding**: vector - Vector embedding for RAG search
- **dependencies**: array - List of prerequisite content IDs
- **difficulty**: string - Difficulty level ("beginner", "intermediate", "advanced")
- **tags**: array - Array of tags for categorization
- **created_at**: datetime - Creation timestamp
- **updated_at**: datetime - Last update timestamp

### RAGSession
- **id**: string (UUID) - Unique identifier for the RAG session
- **user_id**: string - Reference to the user who initiated the session
- **query**: string - The original query from the user
- **response**: string - The AI-generated response
- **context_used**: array - List of content IDs used to generate the response
- **confidence**: float - Confidence score of the response
- **timestamp**: datetime - When the session occurred
- **feedback**: JSON - User feedback on the response quality

### PersonalizationSettings
- **id**: string (UUID) - Unique identifier for personalization settings
- **user_id**: string - Reference to the user
- **learning_style**: string - Preferred learning style ("visual", "textual", "hands-on", "theoretical")
- **technical_background**: string - User's technical background level
- **preferred_topics**: array - Topics of particular interest to the user
- **content_format_preference**: string - Preferred content format ("detailed", "concise", "balanced")
- **created_at**: datetime - Creation timestamp
- **updated_at**: datetime - Last update timestamp

### TranslationCache
- **id**: string (UUID) - Unique identifier for the cached translation
- **content_id**: string - Reference to the original content
- **target_language**: string - Target language code (e.g., "ur", "es", "fr")
- **translated_content**: string - The translated content
- **created_at**: datetime - When the translation was cached
- **expires_at**: datetime - When the cache entry expires

## Relationships

1. **User** → **RAGSession**: One-to-many (one user can have many RAG sessions)
2. **User** → **PersonalizationSettings**: One-to-one (one user has one personalization profile)
3. **User** → **Progress**: One-to-many (one user has progress in many content items)
4. **Content** → **TranslationCache**: One-to-many (one content can have many translations)
5. **RAGSession** → **Content**: Many-to-many (a session can reference multiple content items)

## Validation Rules

### User
- Email must be a valid email format
- Name must not be empty
- Preferences must be a valid JSON object

### Content
- Title must not be empty
- Body must not be empty
- Module, chapter, and section must follow the defined structure
- Difficulty must be one of the allowed values
- Embedding must be a valid vector representation

### RAGSession
- Query must not be empty
- Confidence must be between 0 and 1
- User_id must reference an existing user

### PersonalizationSettings
- User_id must reference an existing user
- Learning style must be one of the allowed values
- Technical background must be one of the allowed values

## State Transitions

### RAGSession
- `created` → `processing` → `completed` | `failed`
- Sessions are created when a user submits a query
- Processing state when the RAG system is working on the query
- Completed when a response is generated successfully
- Failed if there's an error during processing

## Indexes

### Content
- Index on `module` for efficient module-based queries
- Index on `difficulty` for personalized difficulty filtering
- Index on `tags` for tag-based searches
- Composite index on `module` and `chapter` for navigation

### RAGSession
- Index on `user_id` for user-specific query history
- Index on `timestamp` for chronological ordering
- Index on `confidence` for quality analysis

### User
- Unique index on `email` for authentication
- Index on `created_at` for user analytics