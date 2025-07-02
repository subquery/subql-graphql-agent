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
from fastapi import FastAPI, HTTPException, Path as PathParam
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
        self.projects: Dict[str, ProjectConfig] = {}
        self.agent_cache: Dict[str, tuple] = {}  # cid -> (agent, timestamp)
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
    
    def _get_project_file(self, cid: str) -> Path:
        """Get the file path for a project configuration."""
        return PROJECTS_DIR / f"{cid}.json"
    
    def _load_projects(self):
        """Load all projects from disk."""
        for project_file in PROJECTS_DIR.glob("*.json"):
            try:
                with open(project_file, 'r') as f:
                    data = json.load(f)
                    config = ProjectConfig(**data)
                    self.projects[config.cid] = config
                    print(f"📁 Loaded project: {config.cid}")
            except Exception as e:
                print(f"❌ Failed to load project {project_file}: {e}")
    
    def _save_project(self, config: ProjectConfig):
        """Save a project configuration to disk."""
        project_file = self._get_project_file(config.cid)
        with open(project_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    async def register_project(self, cid: str, endpoint: Optional[str] = None) -> ProjectConfig:
        """Register a new project from IPFS CID."""
        if cid in self.projects:
            return self.projects[cid]
        
        try:
            # Fetch manifest from IPFS using cat API
            print(f"📁 Fetching project manifest for CID: {cid}")
            manifest_content = await fetch_from_ipfs(cid)
            
            # Parse manifest (assuming YAML format)
            try:
                manifest = yaml.safe_load(manifest_content)
            except yaml.YAMLError:
                # Try JSON format as fallback
                manifest = json.loads(manifest_content)
            
            # Extract schema path and endpoint
            schema_path = manifest.get('schema', {}).get('file', 'schema.graphql')
            
            # Use provided endpoint or try to get from manifest or construct default
            if not endpoint:
                if 'network' in manifest and 'endpoint' in manifest['network']:
                    endpoint = manifest['network']['endpoint']
                elif 'endpoint' in manifest:
                    endpoint = manifest['endpoint']
                else:
                    # If no endpoint provided and none in manifest, use SubQuery Network default
                    endpoint = f"https://api.subquery.network/sq/{cid}"
            
            # Fetch schema.graphql from IPFS
            if schema_path.startswith('http'):
                # External schema URL, fetch directly
                print(f"🔍 Fetching schema from external URL: {schema_path}")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    schema_response = await client.get(schema_path)
                    schema_response.raise_for_status()
                    schema_content = schema_response.text
            elif schema_path.startswith('ipfs://'):
                # IPFS URL with separate CID
                schema_cid = schema_path.replace('ipfs://', '')
                print(f"📄 Fetching schema from separate IPFS CID: {schema_cid}")
                schema_content = await fetch_from_ipfs(schema_cid)
            else:
                # Schema file within the same IPFS directory
                print(f"📄 Fetching schema file: {schema_path}")
                schema_content = await fetch_from_ipfs(cid, schema_path)
            
            # Analyze project with LLM to generate intelligent defaults
            shared_llm = self._get_shared_llm()
            llm_analysis = await analyze_project_with_llm(manifest, schema_content, shared_llm)
            
            # Create project configuration with LLM-generated defaults
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
            self._save_project(config)
            self.projects[cid] = config
            
            print(f"✅ Registered project: {llm_analysis['domain_name']} ({cid})")
            return config
            
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to register project {cid}: {str(e)}"
            )
    
    def get_project(self, cid: str) -> Optional[ProjectConfig]:
        """Get a project configuration."""
        return self.projects.get(cid)
    
    def update_project_config(self, cid: str, **updates) -> ProjectConfig:
        """Update project configuration."""
        if cid not in self.projects:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found")
        
        config = self.projects[cid]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(config, field):
                setattr(config, field, value)
        
        # Save changes
        self._save_project(config)
        
        # Invalidate agent cache
        if cid in self.agent_cache:
            del self.agent_cache[cid]
        
        print(f"🔄 Updated project config: {cid}")
        return config
    
    def get_agent(self, cid: str) -> 'GraphQLAgent':
        """Get or create a cached agent for the project."""
        current_time = time.time()
        
        # Check cache
        if cid in self.agent_cache:
            agent, timestamp = self.agent_cache[cid]
            if current_time - timestamp < CACHE_TTL:
                return agent
            else:
                # Cache expired
                del self.agent_cache[cid]
        
        # Create new agent
        config = self.get_project(cid)
        if not config:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found")
        
        agent = GraphQLAgent(config)
        self.agent_cache[cid] = (agent, current_time)
        
        print(f"🤖 Created agent for project: {cid}")
        return agent
    
    def delete_project(self, cid: str) -> bool:
        """Delete a project and its configuration."""
        if cid not in self.projects:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found")
        
        # Remove from memory
        del self.projects[cid]
        
        # Remove cached agent
        if cid in self.agent_cache:
            del self.agent_cache[cid]
        
        # Remove from disk
        project_file = self._get_project_file(cid)
        if project_file.exists():
            project_file.unlink()
        
        print(f"🗑️ Deleted project: {cid}")
        return True
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all registered projects."""
        return [
            {
                "cid": config.cid,
                "domain_name": config.domain_name,
                "endpoint": config.endpoint,
                "cached": config.cid in self.agent_cache
            }
            for config in self.projects.values()
        ]

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
            max_iterations=6,
            max_execution_time=180,
            handle_parsing_errors="Check your output and make sure it conforms! Always respond with Thought: and Action:",
            return_intermediate_steps=True
        )
    
    async def query(self, user_input: str) -> str:
        """Process a user query."""
        intermediate_steps = []
        try:
            print(f"🔍 Processing query for {self.config.cid}: {user_input}")
            result = await self.executor.ainvoke({"input": user_input})
            
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            print(f"🔍 Agent execution result: output={bool(output)}, steps={len(intermediate_steps)}")
            
            # Handle cases where agent stops due to limits or parsing errors
            if (not output or "iteration limit" in output or "time limit" in output or "stopped" in output.lower() or "Invalid Format" in output) and intermediate_steps:
                print("🔧 Agent stopped due to limits, extracting results from intermediate steps...")
                
                # Look for GraphQL execution results
                graphql_results = []
                for i, step in enumerate(intermediate_steps):
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        print(f"🔍 Step {i}: tool={getattr(action, 'tool', 'unknown')}, observation_preview={str(observation)[:100]}...")
                        
                        # Check for GraphQL tool execution
                        if hasattr(action, 'tool') and ('execute' in action.tool.lower() or 'graphql' in action.tool.lower()):
                            if "Query executed successfully" in observation or "data" in observation:
                                graphql_results.append(observation)
                                print(f"✅ Found GraphQL result in step {i}")
                        
                        # Also check if observation contains JSON-like data
                        elif observation and ('{' in observation and '}' in observation):
                            graphql_results.append(observation)
                            print(f"✅ Found JSON data in step {i}")
                
                if graphql_results:
                    # Use LLM to summarize the results
                    last_result = graphql_results[-1]
                    summary_prompt = f"""The user asked: "{user_input}"

A GraphQL query was executed successfully with this result:
{last_result}

Please provide a clear, user-friendly summary of this data that directly answers the user's question. Don't mention technical details about GraphQL or JSON format."""
                    
                    from langchain.schema import HumanMessage
                    summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                    output = summary_response.content
                    print("✅ Generated summary from intermediate results")
            
            print(f"📤 Final result for {self.config.cid}: {output[:200]}...")
            return output if output else "No response generated"
            
        except Exception as e:
            print(f"❌ Query failed for {self.config.cid}: {str(e)}")
            
            # Check if it's a parsing error with successful GraphQL execution
            if "Invalid Format" in str(e) and intermediate_steps:
                print("🔧 Parsing error detected, attempting to extract from cached intermediate results...")
                
                # Use the intermediate_steps we already have
                for i, step in enumerate(intermediate_steps):
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        print(f"🔍 Checking step {i}: {str(observation)[:100]}...")
                        if observation and ('{' in observation and '}' in observation):
                            print(f"✅ Using result from step {i}")
                            
                            # Use LLM to format the result nicely
                            try:
                                summary_prompt = f"""The user asked: "{user_input}"

A GraphQL query was executed successfully with this result:
{observation}

Please provide a clear, user-friendly summary of this data that directly answers the user's question. Don't mention technical details about GraphQL or JSON format."""
                                
                                from langchain.schema import HumanMessage
                                summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                                return summary_response.content
                            except:
                                # If LLM summary fails, return raw data
                                return f"Here's the data from your query:\n\n{observation}"
            
            return f"I encountered an issue processing your query. Error: {str(e)}"

# Global project manager
project_manager = ProjectManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("🚀 Multi-Project SubQuery GraphQL Agent Server starting...")
    print(f"📁 Projects directory: {PROJECTS_DIR.absolute()}")
    print(f"🔗 IPFS API: {IPFS_API_URL}")
    
    # Load existing projects
    project_count = len(project_manager.projects)
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
    endpoint: Optional[str] = Field(None, description="GraphQL endpoint URL (optional, will be auto-detected from manifest if not provided)")

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
async def register_project(request: RegisterProjectRequest):
    """Register a new SubQuery project from IPFS CID."""
    config = await project_manager.register_project(request.cid, request.endpoint)
    
    return RegisterProjectResponse(
        cid=config.cid,
        domain_name=config.domain_name,
        endpoint=config.endpoint,
        message=f"Project {config.domain_name} registered successfully"
    )

@app.get("/projects")
async def list_projects():
    """List all registered projects."""
    return {
        "projects": project_manager.list_projects(),
        "total": len(project_manager.projects)
    }

@app.get("/projects/{cid}")
async def get_project(cid: str = PathParam(..., description="Project CID")):
    """Get project configuration."""
    config = project_manager.get_project(cid)
    if not config:
        raise HTTPException(status_code=404, detail=f"Project {cid} not found")
    
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
async def update_project_config(
    cid: str,
    request: UpdateProjectConfigRequest
):
    """Update project configuration."""
    updates = {k: v for k, v in request.dict().items() if v is not None}
    config = project_manager.update_project_config(cid, **updates)
    
    return {
        "cid": config.cid,
        "domain_name": config.domain_name,
        "domain_capabilities": config.domain_capabilities,
        "decline_message": config.decline_message,
        "endpoint": config.endpoint,
        "message": f"Project {cid} configuration updated"
    }

@app.delete("/projects/{cid}")
async def delete_project(cid: str):
    """Delete a project and its configuration."""
    success = project_manager.delete_project(cid)
    
    return {
        "cid": cid,
        "deleted": success,
        "message": f"Project {cid} deleted successfully"
    }

@app.post("/{cid}/chat/completions")
async def project_chat_completions(
    cid: str,
    request: ChatCompletionRequest
):
    """OpenAI compatible chat completions endpoint for a specific project."""
    # Get agent for this project
    agent = project_manager.get_agent(cid)
    
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
        response_content = await agent.query(user_input)
        
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
        # Process query through GraphQL agent
        response_content = await agent.query(user_input)
        
        # Split response into chunks for streaming
        words = response_content.split()
        chunk_size = 3  # Send 3 words at a time
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " " + " ".join(chunk_words) if i > 0 else " ".join(chunk_words)
            
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
            await asyncio.sleep(0.05)  # Small delay for streaming effect
        
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
        "projects_count": len(project_manager.projects),
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