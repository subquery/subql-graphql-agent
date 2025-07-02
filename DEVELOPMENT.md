# Development Guide

This guide explains how to run the SubQuery GraphQL Agent with both backend and frontend.

## Prerequisites

- Python 3.12+
- Node.js 18+
- OpenAI API Key

## Backend Setup

1. **Install Python dependencies:**
```bash
# From project root
uv sync
```

2. **Set environment variables:**
```bash
# Copy and edit environment file
cp .env.example .env

# Edit .env and set:
OPENAI_API_KEY=your-openai-api-key-here
LLM_MODEL=gpt-4o-mini
PORT=8000
```

3. **Start the backend server:**
```bash
# From project root
cd examples
python server.py
```

The backend will start on `http://localhost:8000`

## Frontend Setup

1. **Install Node.js dependencies:**
```bash
# From frontend directory
cd frontend
npm install
```

2. **Start the frontend development server:**
```bash
# From frontend directory
npm run dev
```

The frontend will start on `http://localhost:3000` and automatically proxy API requests to the backend.

## Testing the Setup

1. **Check backend health:**
   - Visit `http://localhost:8000/health`
   - Should return JSON with server status

2. **Check frontend:**
   - Visit `http://localhost:3000`
   - Should see the SubQuery GraphQL Agent interface
   - Top right should show "Online" if backend is connected

## API Endpoints

### Backend (Port 8000)
- `POST /register` - Register new project
- `GET /projects` - List projects
- `GET /projects/{cid}` - Get project config
- `PATCH /projects/{cid}` - Update project config
- `POST /{cid}/chat/completions` - Chat with project
- `GET /health` - Health check

### Frontend (Port 3000)
- Proxies all `/api/*` requests to backend
- Serves React application

## Troubleshooting

### Backend Issues
```bash
# Check if backend is running
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "projects_count": 0,
  "cached_agents": 0,
  "ipfs_gateway": "https://gateway.pinata.cloud/ipfs"
}
```

### Frontend Issues
```bash
# Check if frontend can reach backend
# In browser console, should see:
# "API Base URL: /api"
# API requests to "/api/health", etc.
```

### Common Problems

1. **"Backend offline" in frontend:**
   - Make sure backend is running on port 8000
   - Check CORS settings if needed

2. **API 404 errors:**
   - Verify backend endpoints are working
   - Check proxy configuration in vite.config.ts

3. **OpenAI API errors:**
   - Verify OPENAI_API_KEY is set correctly
   - Check API key has sufficient credits
   - Use gpt-4o for best project analysis (gpt-4o-mini may give generic results)

## Development Workflow

1. Start backend: `cd examples && python server.py`
2. Start frontend: `cd frontend && npm run dev`
3. Register a project via the UI or API
4. Chat with the project using the interface

## Production Build

### Frontend
```bash
cd frontend
npm run build
# Serves dist/ folder with any static hosting
```

### Backend
```bash
# Set production environment variables
export OPENAI_API_KEY=your-key
export LLM_MODEL=gpt-4o
export PORT=8000

# Run the server
python examples/server.py
```