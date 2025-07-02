#!/usr/bin/env python3
"""
Multi-Project SubQuery GraphQL Agent Server

This server can handle multiple SubQuery projects by registering IPFS CIDs
that point to project manifests. Each project gets its own GraphQL agent
with customizable prompts and capabilities.

Features:
- Register projects via IPFS CID manifest fetching
- Project-specific chat completion endpoints: /<cid>/chat/completions
- Agent instance caching with TTL
- Configurable project domains and capabilities
- OpenAI-compatible streaming and non-streaming responses
"""

import asyncio
import json
import os
import sys
import time
import uuid
import yaml
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for development

import httpx
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from graphql_agent import create_graphql_toolkit
from graphql_agent.tools import create_react_agent_prompt

# Configuration
PROJECTS_DIR = Path("./projects")
CACHE_TTL = 3600  # 1 hour agent cache TTL
IPFS_API_URL = os.getenv("IPFS_API_URL", "https://unauthipfs.subquery.network/ipfs/api/v0")

async def fetch_from_ipfs(cid: str, path: str = "") -> str:
    """
    Fetch content from IPFS using multiple methods with fallbacks.
    
    Args:
        cid: IPFS CID
        path: Optional path within the IPFS directory
        
    Returns:
        str: Content of the file
    """
    ipfs_path = f"{cid}/{path}" if path else cid
    
    # Try SubQuery IPFS node first, then gateway fallbacks
    sources = [
        # SubQuery IPFS node (cat API with POST method)
        {
            "name": "SubQuery IPFS Cat API",
            "url": f"{IPFS_API_URL}/cat",
            "method": "post",
            "params": {"arg": ipfs_path}
        },
        # Gateway fallbacks
        {
            "name": "Gateway (gateway.pinata.cloud)",
            "url": f"https://gateway.pinata.cloud/ipfs/{ipfs_path}",
            "method": "get"
        },
        {
            "name": "Gateway (dweb.link)",
            "url": f"https://dweb.link/ipfs/{ipfs_path}",
            "method": "get"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for source in sources:
            try:
                print(f"🔍 Trying {source['name']}: {source['url']}")
                
                if source["method"] == "post":
                    response = await client.post(source["url"], params=source.get("params", {}))
                else:
                    response = await client.get(source["url"])
                
                if response.status_code == 200:
                    content = response.text
                    print(f"✅ Successfully fetched from {source['name']} ({len(content)} chars)")
                    return content
                else:
                    print(f"❌ {source['name']} failed: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                print(f"❌ {source['name']} error: {e}")
                continue
    
    # If all sources fail
    raise HTTPException(
        status_code=400,
        detail=f"Failed to fetch {ipfs_path} from all IPFS sources"
    )


async def analyze_project_with_llm(manifest: dict, schema_content: str, llm=None) -> dict:
    """
    Use LLM to analyze project manifest and schema to generate appropriate prompts.
    
    Args:
        manifest: Project manifest data
        schema_content: GraphQL schema content
        
    Returns:
        dict: Generated domain_name, domain_capabilities, and decline_message
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        
        # Use provided LLM or create one with same config as GraphQLAgent
        if llm is None:
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(
                model=model_name,
                temperature=0  # Same as GraphQLAgent
            )
        
        # Prepare schema content for LLM (truncate if too long)
        schema_preview = schema_content[:3000] if len(schema_content) > 3000 else schema_content
        
        # Get project basics
        project_name = manifest.get('name', 'Unknown Project')
        project_description = manifest.get('description', '')
        
        # Get network/chain info
        network_info = ""
        if 'network' in manifest:
            network = manifest['network']
            if isinstance(network, dict):
                chain_id = network.get('chainId', network.get('endpoint', ''))
                network_info = f"Network: {chain_id}"
        
        # Get datasource info
        datasources_info = ""
        if 'dataSources' in manifest:
            ds_kinds = [ds.get('kind', 'unknown') for ds in manifest['dataSources']]
            datasources_info = f"Data sources: {', '.join(set(ds_kinds))}"
        
        # Create focused analysis prompt
        analysis_prompt = f"""Analyze this SubQuery indexing project and generate specific agent configuration:

PROJECT INFO:
- Name: {project_name}
- Description: {project_description}
- {network_info}
- {datasources_info}

GRAPHQL SCHEMA:
```graphql
{schema_preview}
```

Based on the project info and GraphQL schema entities, generate:

1. A clear domain_name that describes what this project indexes
2. Specific domain_capabilities based on the actual GraphQL entities and what queries users can make
3. A decline_message that mentions the specific domain
4. Suggested questions that users can ask to explore the data

IMPORTANT: Look at the GraphQL types to understand what this project tracks.

Respond ONLY with valid JSON in this exact format (no markdown code blocks):
{{
  "domain_name": "Specific Project Name",
  "domain_capabilities": [
    "Query [specific entity] data and relationships",
    "Analyze [specific metrics] and trends", 
    "Track [specific events/transactions]",
    "Monitor [specific blockchain activities]"
  ],
  "decline_message": "I'm specialized in {project_name} data queries. I can help you with [specific data types], but I cannot assist with [their topic]. Please ask me about {project_name} data instead.",
  "suggested_questions": [
    "Show me recent [specific entity type] transactions",
    "What are the top [entity] by [field]?",
    "How many [events] happened in the last day?",
    "Can you show me a sample GraphQL query for [entity]?"
  ]
}}

Make each capability very specific to the entities found in the schema."""

        print("🤖 Analyzing project with LLM...")
        print(f"📋 Project info - Name: {project_name}, Description: {project_description[:100]}...")
        print(f"🌐 Network: {network_info}")
        print(f"📦 Data sources: {datasources_info}")
        print(f"📄 Schema length: {len(schema_content)} chars (preview: {len(schema_preview)} chars)")
        print(f"📤 Sending prompt to LLM (length: {len(analysis_prompt)} chars)")
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        print(f"🤖 LLM Raw Response: {response.content}")
        
        # Parse JSON response - handle markdown code blocks
        try:
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove closing ```
            
            content = content.strip()
            
            result = json.loads(content)
            
            # Ensure all required fields are present
            if 'suggested_questions' not in result:
                print("⚠️ LLM response missing suggested_questions, adding defaults")
                result['suggested_questions'] = [
                    "What types of data can I query from this project?",
                    "Show me a sample GraphQL query",
                    "What entities are available in this schema?",
                    "How can I filter the data?"
                ]
            
            print(f"✅ LLM analysis completed: {result['domain_name']}")
            print(f"📋 Generated capabilities: {len(result['domain_capabilities'])} items")
            print(f"❓ Generated questions: {len(result['suggested_questions'])} items")
            return result
        except json.JSONDecodeError as e:
            print(f"⚠️ LLM response was not valid JSON: {e}")
            print(f"📄 Full raw response: {response.content}")
            print(f"📄 Cleaned content: {content}")
            raise ValueError("Invalid JSON response from LLM")
            
    except Exception as e:
        print(f"⚠️ LLM analysis failed: {e}, using enhanced fallback")
        
        # Enhanced fallback analysis
        project_name = manifest.get('name', 'SubQuery Project')
        project_description = manifest.get('description', '')
        
        # Generate better domain name
        if project_description and len(project_description) > 10:
            domain_name = f"{project_name} - {project_description[:50]}..."
        else:
            domain_name = project_name
            
        # Generate basic capabilities
        capabilities = [
            "Query blockchain data indexed by this project",
            "Analyze transaction patterns and trends", 
            "Track historical blockchain activities",
            "Monitor smart contract events and state changes"
        ]
            
        return {
            "domain_name": domain_name,
            "domain_capabilities": capabilities,
            "decline_message": f"I'm specialized in {project_name} data queries. I can help you with the indexed blockchain data, but I cannot assist with [their topic]. Please ask me about {project_name} data instead.",
            "suggested_questions": [
                "What types of data can I query from this project?",
                "Show me a sample GraphQL query",
                "What entities are available in this schema?",
                "How can I filter the data?"
            ]
        }

@dataclass
class ProjectConfig:
    """Configuration for a SubQuery project."""
    cid: str
    endpoint: str
    schema_content: str
    domain_name: str = "SubQuery Project"
    domain_capabilities: List[str] = None
    decline_message: str = "I'm specialized in this SubQuery project's data queries. I can help you with the indexed blockchain data, but I cannot assist with [their topic]. Please ask me about this project's data instead."
    suggested_questions: List[str] = None
    
    def __post_init__(self):
        if self.domain_capabilities is None:
            self.domain_capabilities = [
                "Blockchain data indexed by this SubQuery project",
                "Entity relationships and queries",
                "Project-specific metrics and analytics"
            ]
        if self.suggested_questions is None:
            self.suggested_questions = [
                "What types of data can I query from this project?",
                "Show me a sample GraphQL query",
                "What entities are available in this schema?",
                "How can I filter the data?"
            ]

class ProjectManager:
    """Manages SubQuery projects and their configurations."""
    
    def __init__(self):
        self.projects: Dict[str, Dict[str, ProjectConfig]] = {}
        self.agent_cache: Dict[tuple, tuple] = {}  # (user_id, cid) -> (agent, timestamp)
        self._shared_llm = None  # Cached LLM instance for analysis
        PROJECTS_DIR.mkdir(exist_ok=True)
        self._load_projects()
    
    def _get_shared_llm(self):
        """Get or create a shared LLM instance with same config as GraphQLAgent."""
        if self._shared_llm is None:
            from langchain_openai import ChatOpenAI
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            self._shared_llm = ChatOpenAI(
                model=model_name,
                temperature=0
            )
            print(f"🤖 Initialized shared LLM: {model_name}")
        return self._shared_llm
    
    def _get_project_file(self, user_id: str, cid: str) -> Path:
        """Get the file path for a user's project config."""
        user_dir = PROJECTS_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{cid}.json"
    
    def _load_projects(self):
        """Load all projects from disk into memory, grouped by user_id."""
        self.projects.clear()
        if not PROJECTS_DIR.exists():
            return
        for user_dir in PROJECTS_DIR.iterdir():
            if user_dir.is_dir():
                user_id = user_dir.name
                self.projects[user_id] = {}
                for file in user_dir.glob("*.json"):
                    try:
                        with open(file, "r") as f:
                            data = json.load(f)
                            config = ProjectConfig(**data)
                            self.projects[user_id][config.cid] = config
                            print(f"📁 Loaded project: {config.cid} for user {user_id}")
                    except Exception as e:
                        print(f"❌ Failed to load project {file}: {e}")
    
    def _save_project(self, user_id: str, config: ProjectConfig):
        """Save a user's project config to disk."""
        file = self._get_project_file(user_id, config.cid)
        with open(file, "w") as f:
            json.dump(asdict(config), f, indent=2)
    
    async def register_project(self, user_id: str, cid: str, endpoint: str) -> ProjectConfig:
        """Register a new project from IPFS CID for a specific user."""
        if user_id in self.projects and cid in self.projects[user_id]:
            return self.projects[user_id][cid]
        try:
            # Fetch manifest from IPFS using cat API
            print(f"📁 Fetching project manifest for CID: {cid}")
            manifest_content = await fetch_from_ipfs(cid)
            # Parse manifest (YAML or JSON)
            try:
                manifest = yaml.safe_load(manifest_content)
            except yaml.YAMLError:
                manifest = json.loads(manifest_content)
            schema_path = manifest.get('schema', {}).get('file', 'schema.graphql')
            if schema_path.startswith('http'):
                print(f"🔍 Fetching schema from external URL: {schema_path}")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    schema_response = await client.get(schema_path)
                    schema_response.raise_for_status()
                    schema_content = schema_response.text
            elif schema_path.startswith('ipfs://'):
                schema_cid = schema_path.replace('ipfs://', '')
                print(f"📄 Fetching schema from separate IPFS CID: {schema_cid}")
                schema_content = await fetch_from_ipfs(schema_cid)
            else:
                print(f"📄 Fetching schema file: {schema_path}")
                schema_content = await fetch_from_ipfs(cid, schema_path)
            shared_llm = self._get_shared_llm()
            llm_analysis = await analyze_project_with_llm(manifest, schema_content, shared_llm)
            config = ProjectConfig(
                cid=cid,
                endpoint=endpoint,
                schema_content=schema_content,
                domain_name=llm_analysis["domain_name"],
                domain_capabilities=llm_analysis["domain_capabilities"],
                decline_message=llm_analysis["decline_message"],
                suggested_questions=llm_analysis.get("suggested_questions", [])
            )
            # Save to disk and memory
            self._save_project(user_id, config)
            if user_id not in self.projects:
                self.projects[user_id] = {}
            self.projects[user_id][cid] = config
            print(f"✅ Registered project: {llm_analysis['domain_name']} ({cid}) for user {user_id}")
            return config
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to register project {cid}: {str(e)}"
            )
    
    def get_project(self, user_id: str, cid: str) -> Optional[ProjectConfig]:
        """Get a project configuration for a user."""
        return self.projects.get(user_id, {}).get(cid)
    
    def update_project_config(self, cid: str, **updates) -> ProjectConfig:
        """Update project configuration."""
        if cid not in self.projects:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found")
        
        config = self.projects[cid][cid]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(config, field):
                setattr(config, field, value)
        
        # Save changes
        self._save_project(cid, config)
        
        # Invalidate agent cache
        if cid in self.agent_cache:
            del self.agent_cache[cid]
        
        print(f"🔄 Updated project config: {cid}")
        return config
    
    def get_agent(self, user_id: str, cid: str) -> 'GraphQLAgent':
        """Get or create a cached agent for the project (per user)."""
        current_time = time.time()
        cache_key = (user_id, cid)
        # Check cache
        if cache_key in self.agent_cache:
            agent, timestamp = self.agent_cache[cache_key]
            if current_time - timestamp < CACHE_TTL:
                return agent
            else:
                # Cache expired
                del self.agent_cache[cache_key]
        # Create new agent
        config = self.get_project(user_id, cid)
        if not config:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found for user {user_id}")
        agent = GraphQLAgent(config)
        self.agent_cache[cache_key] = (agent, current_time)
        print(f"🤖 Created agent for project: {cid} (user: {user_id})")
        return agent
    
    def delete_project(self, cid: str) -> bool:
        """Delete a project and its configuration."""
        if cid not in self.projects:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found")
        
        # Remove from memory
        del self.projects[cid][cid]
        
        # Remove cached agent
        if cid in self.agent_cache:
            del self.agent_cache[cid]
        
        # Remove from disk
        project_file = self._get_project_file(cid, cid)
        if project_file.exists():
            project_file.unlink()
        
        print(f"🗑️ Deleted project: {cid}")
        return True
    
    def list_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """List all registered projects for a user by reading the user's project directory."""
        user_dir = PROJECTS_DIR / user_id
        projects = []
        if not user_dir.exists() or not user_dir.is_dir():
            return []
        for file in user_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    config = ProjectConfig(**data)
                    projects.append({
                        "cid": config.cid,
                        "domain_name": config.domain_name,
                        "endpoint": config.endpoint,
                        "cached": config.cid in self.agent_cache
                    })
            except Exception as e:
                print(f"❌ Failed to load project {file}: {e}")
        return projects

class GraphQLAgent:
    """GraphQL agent for a specific SubQuery project."""
    
    def __init__(self, config: ProjectConfig):
        """Initialize the agent with project configuration."""
        self.config = config
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize LLM
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
        
        # Create tools
        toolkit = create_graphql_toolkit(config.endpoint, config.schema_content)
        self.tools = toolkit.get_tools()
        
        # Setup agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the LangChain agent with project-specific prompt."""
        prompt_template = create_react_agent_prompt(
            domain_name=self.config.domain_name,
            domain_capabilities=self.config.domain_capabilities,
            decline_message=self.config.decline_message
        )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor with better error handling
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            max_execution_time=180,
            handle_parsing_errors="After calling graphql_execute and getting results, you must provide:\nThought: I have the query results\nFinal Answer: [user-friendly summary of the data]",
            return_intermediate_steps=True
        )
    
    async def query(self, user_input: str):
        """Truly streaming: use self.executor.astream to yield tool observations and final output step by step, chunked."""
        think_started = False
        chunk_size = 60
            
        try:
            print(f"🔍 [astream] Processing query for {self.config.cid}: {user_input}")
            async for event in self.executor.astream({"input": user_input}):
                # Tool step event (chunked streaming)
                if event.get("steps"):
                    if not think_started:
                        yield "<think>\n"
                        think_started = True
                    for step in event["steps"]:
                        action = getattr(step, 'action', None)
                        observation = getattr(step, 'observation', None)
                        if action:
                            tool_name = getattr(action, 'tool', 'tool')
                            log = getattr(action, 'log', '')
                            header = f"[Tool: {tool_name}]\n"
                            yield header
                            # Chunked yield for log
                            idx = 0
                            while idx < len(log):
                                chunk = log[idx:idx+chunk_size]
                                yield chunk
                                idx += chunk_size
                        # Chunked yield for observation (tool output)
                        if observation:
                            yield "\n[Tool Output]:\n"
                            
                            # Truncate graphql_schema_info tool output to reduce token usage
                            display_observation = observation
                            if action and hasattr(action, 'tool') and action.tool == 'graphql_schema_info':
                                max_length = 2000  # Limit schema info output
                                if len(observation) > max_length:
                                    display_observation = observation[:max_length] + f"\n\n... [Output truncated after {max_length} characters to save tokens. Full schema info available but not displayed.]"
                            
                            idx = 0
                            while idx < len(display_observation):
                                chunk = display_observation[idx:idx+chunk_size]
                                yield chunk
                                idx += chunk_size
                        yield "\n\n"
                # Final answer event: 只要有 'output' 字段且非空
                if event.get("output"):
                    # 先关闭 think block（如果还没关）
                    if think_started:
                        yield "</think>\n"
                        think_started = False
                    
                    output = event["output"].strip()
                    
                    idx = 0
                    while idx < len(output):
                        chunk = output[idx:idx+chunk_size]
                        yield chunk
                        idx += chunk_size
            # 兜底：如果有 think block 没关
            if think_started:
                yield "</think>\n"
        except Exception as e:
            print(f"❌ [astream] Query failed for {self.config.cid}: {str(e)}")
            yield f"I encountered an issue processing your query. Error: {str(e)}"
            return

# Global project manager
project_manager = ProjectManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 Multi-Project SubQuery GraphQL Agent Server starting...")
    print(f"📁 Projects directory: {PROJECTS_DIR.absolute()}")
    print(f"🔗 IPFS API: {IPFS_API_URL}")
    
    # Load existing projects
    project_count = sum(len(projects) for projects in project_manager.projects.values())
    if project_count > 0:
        print(f"✅ Loaded {project_count} existing projects")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Multi-Project GraphQL Agent Server...")

app = FastAPI(
    title="Multi-Project SubQuery GraphQL Agent API", 
    description="OpenAI-compatible API server for multiple SubQuery projects",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class RegisterProjectRequest(BaseModel):
    cid: str = Field(..., description="IPFS CID of the SubQuery project manifest")
    endpoint: str = Field(..., description="GraphQL endpoint URL for the SubQuery project")

class RegisterProjectResponse(BaseModel):
    cid: str
    domain_name: str
    endpoint: str
    message: str

class UpdateProjectConfigRequest(BaseModel):
    domain_name: Optional[str] = None
    domain_capabilities: Optional[List[str]] = None
    decline_message: Optional[str] = None
    endpoint: Optional[str] = None
    suggested_questions: Optional[List[str]] = None

class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    messages: List[ChatCompletionMessage] = Field(..., description="List of messages")
    stream: bool = Field(default=False, description="Whether to stream responses")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# API Endpoints

@app.post("/register", response_model=RegisterProjectResponse)
async def register_project(request: RegisterProjectRequest, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    config = await project_manager.register_project(user_id, request.cid, request.endpoint)
    return RegisterProjectResponse(
        cid=config.cid,
        domain_name=config.domain_name,
        endpoint=config.endpoint,
        message=f"Project {config.domain_name} registered successfully"
    )

@app.get("/projects")
async def list_projects(user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    projects = project_manager.list_projects(user_id)
    return {
        "projects": projects,
        "total": len(projects)
    }

@app.get("/projects/{cid}")
async def get_project(cid: str = PathParam(..., description="Project CID"), user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    config = project_manager.get_project(user_id, cid)
    if not config:
        raise HTTPException(status_code=404, detail=f"Project {cid} not found for user {user_id}")
    return {
        "cid": config.cid,
        "domain_name": config.domain_name,
        "domain_capabilities": config.domain_capabilities,
        "decline_message": config.decline_message,
        "endpoint": config.endpoint,
        "suggested_questions": config.suggested_questions,
        "cached": cid in project_manager.agent_cache
    }

@app.patch("/projects/{cid}")
async def update_project_config(cid: str, request: UpdateProjectConfigRequest, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    updates = {k: v for k, v in request.dict().items() if v is not None}
    config = project_manager.update_project_config(user_id, cid, **updates)
    return {
        "cid": config.cid,
        "domain_name": config.domain_name,
        "domain_capabilities": config.domain_capabilities,
        "decline_message": config.decline_message,
        "endpoint": config.endpoint,
        "message": f"Project {cid} configuration updated"
    }

@app.delete("/projects/{cid}")
async def delete_project(cid: str, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    success = project_manager.delete_project(user_id, cid)
    return {
        "cid": cid,
        "deleted": success,
        "message": f"Project {cid} deleted successfully"
    }

@app.post("/{cid}/chat/completions")
async def project_chat_completions(cid: str, request: ChatCompletionRequest, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    agent = project_manager.get_agent(user_id, cid)
    
    # Extract user message (last message)
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    user_input = user_messages[-1].content
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(agent, user_input, request),
            media_type="text/plain"
        )
    else:
        return await non_stream_chat_completion(agent, user_input, request)

async def non_stream_chat_completion(
    agent: GraphQLAgent, 
    user_input: str, 
    request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    try:
        # Process query through GraphQL agent
        response_parts = []
        async for chunk in agent.query(user_input):
            response_parts.append(chunk)
        response_content = "".join(response_parts)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response_content
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(user_input.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_input.split()) + len(response_content.split())
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def stream_chat_completion(
    agent: GraphQLAgent,
    user_input: str, 
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Handle streaming chat completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    try:
        # 直接异步迭代agent.query，分块流式输出think块和output
        async for part in agent.query(user_input):
            chunk_size = 60
            idx = 0
            while idx < len(part):
                chunk_content = part[idx:idx+chunk_size]
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None
                    }]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.05)
                idx += chunk_size
        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {"content": f"Error: {str(e)}"},
                "finish_reason": "error"
            }]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

# Legacy endpoints for backward compatibility
@app.post("/v1/chat/completions")
async def legacy_chat_completions(request: ChatCompletionRequest):
    """Legacy OpenAI compatible chat completions endpoint."""
    # Use the first available project or SubQuery Network as default
    projects = project_manager.list_projects()
    if not projects:
        raise HTTPException(
            status_code=503, 
            detail="No projects registered. Please register a project first using POST /register"
        )
    
    # Use first project as default
    default_cid = projects[0]["cid"]
    return await project_chat_completions(default_cid, request)

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "anymodel",
                "object": "model",
                "created": 1234567890,
                "owned_by": "subql-graphql-agent"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "projects_count": sum(len(projects) for projects in project_manager.projects.values()),
        "cached_agents": len(project_manager.agent_cache),
        "ipfs_api": IPFS_API_URL
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Multi-Project SubQuery GraphQL Agent API server on port {port}")
    print("📖 API endpoints:")
    print(f"  - POST http://localhost:{port}/register")
    print(f"  - GET  http://localhost:{port}/projects")
    print(f"  - POST http://localhost:{port}/<cid>/chat/completions")
    print(f"  - GET  http://localhost:{port}/health")
    print(f"  - POST http://localhost:{port}/v1/chat/completions (legacy)")
    
    uvicorn.run(app, host="0.0.0.0", port=port)