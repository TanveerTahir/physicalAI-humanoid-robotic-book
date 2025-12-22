# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Development Environment Setup

### Prerequisites
- Node.js 18+ (for Docusaurus frontend)
- Python 3.11+ (for backend services)
- Docker (for local development of backend services)
- Git

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. Run the backend server:
   ```bash
   python -m src.main
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Run the Docusaurus development server:
   ```bash
   npm start
   ```

## Content Creation Workflow

### Adding New Chapters
1. Create a new markdown file in the appropriate module directory:
   ```text
   frontend/docs/module1-foundations/new-chapter.md
   ```

2. Add frontmatter to your chapter:
   ```markdown
   ---
   title: Chapter Title
   sidebar_position: 1
   description: Brief description of the chapter
   module: module1-foundations
   difficulty: beginner
   tags: [tag1, tag2]
   ---
   ```

3. Update the sidebar configuration in `frontend/sidebars.js` to include your new chapter.

### Content Chunking Guidelines
For optimal RAG performance, follow these content chunking guidelines:
- Keep chunks between 200-500 words
- Each chunk should cover a single concept or idea
- Include relevant code examples with explanations
- Use clear headings and subheadings
- Add appropriate tags for categorization

## API Usage Examples

### Querying the RAG System
```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain how humanoid robots maintain balance",
    "context": "module3-perception"
  }'
```

### Searching Content
```bash
curl "http://localhost:8000/api/v1/content/search?q=balance&difficulty=intermediate"
```

## Running Tests

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Deployment

### Frontend Deployment
The frontend can be deployed to GitHub Pages or Vercel:
```bash
cd frontend
npm run build
# The build output will be in the build/ directory
```

### Backend Deployment
The backend can be containerized and deployed to any cloud provider:
```bash
cd backend
docker build -t textbook-backend .
docker run -p 8000:8000 textbook-backend
```

## Development Tips

1. **Content Structure**: Organize content in modules with clear learning progressions
2. **RAG Optimization**: Structure content with clear headings and semantic boundaries
3. **Code Examples**: Include runnable code examples with clear explanations
4. **Cross-References**: Use internal links to connect related concepts
5. **Media**: Include diagrams and images to support learning

## Troubleshooting

### Common Issues
- **RAG queries timing out**: Check that the vector database is properly configured
- **Content not appearing**: Verify that the markdown file has correct frontmatter
- **API authentication failing**: Ensure your JWT tokens are properly formatted

### Getting Help
- Check the API documentation at `/api/docs`
- Review the Docusaurus documentation for content formatting
- Look at existing content examples for formatting patterns