#!/usr/bin/env python3
"""
FastAPI server with OpenAI-compatible completion API using GraphQL Agent

IMPORTANT: This example is configured for SubQuery Network data.
For your own project, you need to customize:
1. Replace the endpoint URL with your project's GraphQL API
2. Replace schema.graphql with your project's entity schema
3. Update the agent prompt to reflect your project's domain and capabilities

See README.md "Custom Agent Prompts" section for detailed guidance.
"""

import asyncio
import os
import sys
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for development

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from graphql_agent import create_graphql_toolkit
from graphql_agent.tools import create_react_agent_prompt

# Global agent instance
agent_executor = None

class GraphQLAgent:
    """GraphQL agent using LangChain."""
    
    def __init__(self, endpoint: str):
        """Initialize the agent."""
        self.endpoint = endpoint
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize LLM
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
        print(f"🤖 Using LLM model: {model_name}")
        
        # Load entity schema
        # Note: This example uses SubQuery Network's schema - replace with your own project's schema
        schema_file_path = os.path.join(os.path.dirname(__file__), "schema.graphql")
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            entity_schema = f.read()
        
        # Create tools
        toolkit = create_graphql_toolkit(endpoint, entity_schema)
        self.tools = toolkit.get_tools()
        
        # Setup agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the LangChain agent."""
        
        # IMPORTANT: This prompt is specifically tailored for SubQuery Network example
        # For your own project, customize these parameters:
        # - Replace domain_name with your project name
        # - Update domain_capabilities to match your indexed data
        # - Adjust decline_message to be appropriate for your use case
        # See README.md "Custom Agent Prompts" section for guidance
        prompt_template = create_react_agent_prompt(
            domain_name="SubQuery Network",
            domain_capabilities=[
                "Blockchain indexers and their status, performance, and rewards",
                "Projects, deployments, and project metadata",
                "Staking rewards, delegations, and era performance",
                "Network statistics, governance, and protocol data",
                "Withdrawal requests and commission tracking"
            ],
            decline_message="I'm specialized in SubQuery Network data queries. I can help you with indexers, projects, staking rewards, and network statistics, but I cannot assist with [their topic]. Please ask me about SubQuery Network data instead."
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
        
        # Create executor
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=6,
            max_execution_time=180,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    async def query(self, user_input: str) -> str:
        """Process a user query."""
        try:
            print(f"🔍 Processing query: {user_input}")
            result = await self.executor.ainvoke({"input": user_input})
            
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Handle cases where agent stops due to limits
            if (not output or "iteration limit" in output or "time limit" in output or "stopped" in output.lower()) and intermediate_steps:
                print("🔧 Agent stopped due to limits, extracting results from intermediate steps...")
                
                # Look for GraphQL execution results
                graphql_results = []
                for step in intermediate_steps:
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        if hasattr(action, 'tool') and 'execute' in action.tool:
                            if "Query executed successfully" in observation:
                                graphql_results.append(observation)
                
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
            
            print(f"📤 Final result: {output[:200]}...")
            return output if output else "No response generated"
            
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            return f"Error processing query: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent_executor
    
    # Startup: Initialize GraphQL agent
    print("🚀 Initializing GraphQL Agent...")
    try:
        # Note: This example uses SubQuery Network's API - replace with your own project's endpoint
        endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
        agent_executor = GraphQLAgent(endpoint)
        print("✅ GraphQL Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize GraphQL Agent: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 Shutting down GraphQL Agent...")

app = FastAPI(
    title="GraphQL Agent API", 
    description="OpenAI-compatible API with GraphQL agent",
    lifespan=lifespan
)

# OpenAI compatible models
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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI compatible chat completions endpoint."""
    global agent_executor
    
    if agent_executor is None:
        raise HTTPException(status_code=503, detail="GraphQL Agent not initialized")
    
    # Extract user message (last message)
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    user_input = user_messages[-1].content
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(user_input, request),
            media_type="text/plain"
        )
    else:
        return await non_stream_chat_completion(user_input, request)

async def non_stream_chat_completion(user_input: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    global agent_executor
    
    try:
        # Process query through GraphQL agent
        response_content = await agent_executor.query(user_input)
        
        import time
        import uuid
        
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

async def stream_chat_completion(user_input: str, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Handle streaming chat completion."""
    global agent_executor
    
    import time
    import uuid
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    try:
        # Process query through GraphQL agent
        response_content = await agent_executor.query(user_input)
        
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
                "owned_by": "graphql-agent"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global agent_executor
    return {
        "status": "healthy",
        "agent_ready": agent_executor is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting GraphQL Agent API server on port {port}")
    print("📖 OpenAI compatible endpoints:")
    print(f"  - POST http://localhost:{port}/v1/chat/completions")
    print(f"  - GET  http://localhost:{port}/v1/models")
    print(f"  - GET  http://localhost:{port}/health")
    
    uvicorn.run(app, host="0.0.0.0", port=port)