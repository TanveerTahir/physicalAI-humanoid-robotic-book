# API Contract: Physical AI & Humanoid Robotics Textbook Backend

## Authentication API

### POST /api/v1/auth/register
Register a new user

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "name": "John Doe"
}
```

**Response (201 Created):**
```json
{
  "id": "uuid-string",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2025-12-22T10:00:00Z"
}
```

### POST /api/v1/auth/login
Login user and return authentication token

**Request:**
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response (200 OK):**
```json
{
  "access_token": "jwt-token-string",
  "user": {
    "id": "uuid-string",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

### GET /api/v1/auth/profile
Get current user profile

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response (200 OK):**
```json
{
  "id": "uuid-string",
  "email": "user@example.com",
  "name": "John Doe",
  "preferences": {},
  "progress": {}
}
```

## RAG (Retrieval-Augmented Generation) API

### POST /api/v1/rag/query
Submit a query to the RAG system for textbook content

**Headers:**
```
Authorization: Bearer {access_token} (optional for public content)
Content-Type: application/json
```

**Request:**
```json
{
  "query": "Explain how humanoid robots maintain balance",
  "context": "module3-perception",  // Optional: limit search to specific module
  "max_results": 5,  // Optional: number of results to return (default: 5)
  "include_metadata": true  // Optional: include metadata in response (default: true)
}
```

**Response (200 OK):**
```json
{
  "query": "Explain how humanoid robots maintain balance",
  "response": "Humanoid robots maintain balance through a combination of sensors, control algorithms, and mechanical design...",
  "confidence": 0.85,
  "context_used": [
    {
      "id": "content-uuid",
      "title": "Balance Control in Humanoid Robots",
      "module": "module3-perception",
      "chapter": "stability-systems",
      "relevance_score": 0.92
    }
  ],
  "timestamp": "2025-12-22T10:00:00Z"
}
```

### GET /api/v1/rag/history
Get user's RAG query history

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response (200 OK):**
```json
{
  "queries": [
    {
      "id": "session-uuid",
      "query": "Explain how humanoid robots maintain balance",
      "response": "Humanoid robots maintain balance through...",
      "timestamp": "2025-12-22T10:00:00Z",
      "confidence": 0.85
    }
  ]
}
```

## Content API

### GET /api/v1/content/search
Search textbook content

**Headers:**
```
Authorization: Bearer {access_token} (optional)
```

**Query Parameters:**
- `q`: Search query string
- `module`: Optional module filter
- `difficulty`: Optional difficulty filter (beginner, intermediate, advanced)
- `tags`: Optional comma-separated tags filter
- `limit`: Optional result limit (default: 20)
- `offset`: Optional offset for pagination (default: 0)

**Response (200 OK):**
```json
{
  "results": [
    {
      "id": "content-uuid",
      "title": "Balance Control in Humanoid Robots",
      "module": "module3-perception",
      "chapter": "stability-systems",
      "section": "balance-mechanisms",
      "snippet": "Humanoid robots maintain balance through a combination of...",
      "difficulty": "intermediate",
      "tags": ["balance", "stability", "control"],
      "relevance_score": 0.92
    }
  ],
  "total_count": 1,
  "limit": 20,
  "offset": 0
}
```

### GET /api/v1/content/{id}
Get specific content by ID

**Headers:**
```
Authorization: Bearer {access_token} (optional)
```

**Response (200 OK):**
```json
{
  "id": "content-uuid",
  "title": "Balance Control in Humanoid Robots",
  "body": "# Balance Control in Humanoid Robots\n\nHumanoid robots maintain balance...",
  "module": "module3-perception",
  "chapter": "stability-systems",
  "section": "balance-mechanisms",
  "difficulty": "intermediate",
  "tags": ["balance", "stability", "control"],
  "dependencies": ["content-prereq-uuid"],
  "created_at": "2025-12-22T10:00:00Z",
  "updated_at": "2025-12-22T10:00:00Z"
}
```

## Personalization API

### GET /api/v1/personalize/settings
Get user's personalization settings

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response (200 OK):**
```json
{
  "learning_style": "hands-on",
  "technical_background": "intermediate",
  "preferred_topics": ["control-systems", "perception"],
  "content_format_preference": "balanced"
}
```

### PUT /api/v1/personalize/settings
Update user's personalization settings

**Headers:**
```
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request:**
```json
{
  "learning_style": "theoretical",
  "technical_background": "advanced",
  "preferred_topics": ["control-systems", "perception", "navigation"],
  "content_format_preference": "detailed"
}
```

**Response (200 OK):**
```json
{
  "learning_style": "theoretical",
  "technical_background": "advanced",
  "preferred_topics": ["control-systems", "perception", "navigation"],
  "content_format_preference": "detailed",
  "updated_at": "2025-12-22T10:00:00Z"
}
```

### POST /api/v1/personalize/recommendations
Get personalized content recommendations

**Headers:**
```
Authorization: Bearer {access_token}
```

**Response (200 OK):**
```json
{
  "recommendations": [
    {
      "id": "content-uuid",
      "title": "Advanced Control Systems",
      "module": "module2-simulation",
      "chapter": "control-theory",
      "reason": "Matches your interest in control systems",
      "difficulty": "advanced"
    }
  ]
}
```

## Translation API

### POST /api/v1/translate
Translate content to target language

**Headers:**
```
Authorization: Bearer {access_token} (optional)
Content-Type: application/json
```

**Request:**
```json
{
  "content_id": "content-uuid",  // Optional: if translating specific content
  "text": "Text to translate",   // Required if content_id not provided
  "target_language": "ur"        // Language code (e.g., ur, es, fr)
}
```

**Response (200 OK):**
```json
{
  "original_text": "Text to translate",
  "translated_text": "مترجمہ متن",
  "target_language": "ur",
  "cached": true  // Whether result was from cache
}
```