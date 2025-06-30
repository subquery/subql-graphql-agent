# SubQL GraphQL Agent

A specialized GraphQL agent toolkit for LLM interactions with SubQuery Network APIs, featuring natural language query capabilities and OpenAI-compatible API endpoints.

## Overview

This toolkit provides LLM agents with the ability to interact with SubQuery Network GraphQL APIs through natural language, automatically understanding schemas, validating queries, and executing complex GraphQL operations.

### Key Features

- **Natural Language Interface**: Ask questions about SubQuery Network data in plain English
- **Automatic Schema Understanding**: Agents learn PostGraphile v4 patterns and SubQuery entity schemas
- **Query Generation & Validation**: Converts natural language to valid GraphQL queries with built-in validation
- **OpenAI-Compatible API**: FastAPI server with streaming and non-streaming endpoints
- **Specialized for SubQuery Network**: Understands indexers, projects, rewards, delegations, and staking data

## Architecture

### Core Components

1. **GraphQLSource** - Connection wrapper for GraphQL endpoints with entity schema support
2. **GraphQLToolkit** - LangChain-compatible toolkit providing all GraphQL tools
3. **GraphQL Agent Tools** - Individual tools for specific GraphQL operations
4. **FastAPI Server** - OpenAI-compatible API with streaming support

### Available Tools

1. **`graphql_schema_info`** - Get raw entity schema with PostGraphile v4 rules
2. **`graphql_type_detail`** - Get detailed type information (fallback tool)
3. **`graphql_query_validator`** - Validate GraphQL query syntax against schema
4. **`graphql_execute`** - Execute GraphQL queries and return results

## Quick Start

### Prerequisites

1. **Python 3.12+**
2. **OpenAI API Key** (for LLM capabilities)
3. **Dependencies**:

```bash
# Install dependencies
uv sync

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key-here"
export LLM_MODEL="gpt-4o"  # Recommended: gpt-4o or stronger models
export PORT="8000"              # Optional, defaults to 8000
```

### Interactive Mode

Run the agent interactively:

```bash
cd examples
python working_example.py
```

### API Server Mode

Start the OpenAI-compatible API server:

```bash
cd examples
python server.py
```

The server will start on `http://localhost:8000` with endpoints:
- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `GET /v1/models` - List available models
- `GET /health` - Health check

## Usage Examples

### Interactive Agent

```python
from graphql_agent import create_graphql_toolkit
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor

# Load entity schema (learn more: https://subquery.network/doc/indexer/build/graphql.html)
with open("examples/schema.graphql", 'r') as f:
    entity_schema = f.read()

# Create toolkit
endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
toolkit = create_graphql_toolkit(endpoint, entity_schema)

# Create agent  
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Use gpt-4o or stronger for best results
agent = create_react_agent(llm, toolkit.get_tools(), prompt_template)
executor = AgentExecutor(agent=agent, tools=toolkit.get_tools())

# Query with natural language
result = executor.invoke({
    "input": "Show me the top 3 indexers with their project information"
})
```

### OpenAI-Compatible API

```bash
# Non-streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Show me 5 indexers and their rewards"}],
    "stream": false
  }'

# Streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What projects are available?"}],
    "stream": true
  }'
```

### Example Natural Language Queries

The agent is specialized for SubQuery Network data and can handle queries like:

#### Basic Data Retrieval
- "Show me the first 5 indexers and their IDs"
- "What projects are available? Show me their owners"
- "List all indexers with their project information"

#### Staking & Rewards
- "What are my staking rewards for wallet 0x123...?"
- "Show me rewards for the last era"
- "Find delegations for a specific indexer"

#### Performance & Analytics
- "Which indexers have the highest rewards?"
- "Show me project performance metrics"
- "List top performing indexers by era"

#### Schema Exploration
- "What types of data can I query?"
- "Show me available project information"
- "What reward data is tracked?"

## PostGraphile v4 Query Patterns

The agent understands PostGraphile v4 patterns automatically:

### Entity Queries
- **Single**: `entityName(id: ID!)` → Full entity object
- **Collection**: `entityNames(first: Int, filter: EntityFilter)` → Connection with pagination

### Filtering
```graphql
filter: {
  fieldName: { equalTo: "value" }
  amount: { greaterThan: 100 }
  status: { in: ["active", "pending"] }
}
```

### Ordering
```graphql
orderBy: [FIELD_NAME_ASC, CREATED_AT_DESC]
```

### Pagination
```graphql
{
  entities(first: 10, after: "cursor") {
    nodes { id, field }
    pageInfo { hasNextPage, endCursor }
  }
}
```

## Agent Workflow

The agent follows this intelligent workflow:

1. **Relevance Check**: Determines if the question relates to SubQuery Network data
2. **Schema Analysis**: Loads entity schema and PostGraphile rules (once per session)
3. **Query Construction**: Builds GraphQL queries using PostGraphile patterns
4. **Validation**: Validates queries against the live GraphQL schema
5. **Execution**: Executes validated queries to get real data
6. **Summarization**: Provides user-friendly responses based on actual results

### Non-Relevant Query Handling

For questions unrelated to SubQuery Network (e.g., "How to cook pasta?"), the agent politely declines without using any tools:

> "I'm specialized in SubQuery Network data queries. I can help you with indexers, projects, staking rewards, and network statistics, but I cannot assist with cooking. Please ask me about SubQuery Network data instead."

## Tool Details

### GraphQL Schema Info Tool
- **Purpose**: Get raw entity schema with PostGraphile v4 guidance
- **Input**: None
- **Output**: Complete entity schema with query construction rules
- **Usage**: Called once per session to understand data structure

### GraphQL Type Detail Tool
- **Purpose**: Get specific type definitions (fallback when validation fails)
- **Input**: `type_name` (string)
- **Output**: Type definition with minimal token usage (depth=0)
- **Usage**: Only used when validation fails and more type info is needed

### GraphQL Query Validator Tool
- **Purpose**: Validate GraphQL query syntax and schema compatibility
- **Input**: `query` (string) - plain text, auto-cleans formatting
- **Output**: Validation result with detailed error messages
- **Usage**: Always called before query execution

### GraphQL Execute Tool
- **Purpose**: Execute validated GraphQL queries
- **Input**: `query` (string), optional `variables` (dict)
- **Output**: Query results or execution errors
- **Usage**: Called after successful validation to get actual data

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional
export LLM_MODEL="gpt-4o"  # Default model
export PORT="8000"              # Server port
```

### Custom Headers & Authentication

```python
from graphql_agent import create_graphql_toolkit

# With custom headers
headers = {
    "Authorization": "Bearer your-token",
    "X-API-Key": "your-api-key"
}

toolkit = create_graphql_toolkit(
    endpoint="https://your-graphql-endpoint.com/graphql",
    entity_schema=schema_content,
    headers=headers
)
```

### Schema Caching

The toolkit automatically caches GraphQL schemas for performance:

```python
from graphql_agent.base import GraphQLSource

source = GraphQLSource(
    endpoint="https://api.example.com/graphql",
    entity_schema=schema_content,
    schema_cache_ttl=3600  # Cache for 1 hour
)
```

## Development

### Project Structure

```
subql-graphql-agent/
├── graphql_agent/           # Core toolkit package
│   ├── __init__.py         # Package exports
│   ├── base.py             # GraphQLSource and GraphQLToolkit
│   ├── tools.py            # Individual GraphQL tools
│   └── graphql.py          # Schema processing utilities
├── examples/               # Usage examples
│   ├── working_example.py  # Interactive agent demo
│   ├── server.py           # OpenAI-compatible API server
│   └── schema.graphql      # SubQuery entity schema
└── pyproject.toml          # Dependencies and configuration
```

### Dependencies

#### Core
- `python-dotenv>=1.0.0` - Environment variable loading
- `fastapi>=0.109.0` - Web framework for API server
- `uvicorn>=0.27.0` - ASGI server
- `pydantic>=2.6.0` - Data validation
- `httpx>=0.27.0` - HTTP client
- `aiohttp>=3.9.0` - Async HTTP requests
- `graphql-core>=3.2.0` - GraphQL query parsing and validation

#### LangChain Integration
- `langchain>=0.1.0` - Agent framework
- `langchain-core>=0.1.0` - Core components
- `langchain-openai>=0.1.0` - OpenAI integration

#### Development
- `pytest>=8.4.1` - Testing framework

### Testing

Run the test suite:

```bash
pytest tests/ -v
```

### Linting & Formatting

The project uses Ruff for linting and formatting:

```bash
# Lint
ruff check .

# Format
ruff format .
```

## Error Handling

The toolkit includes comprehensive error handling:

### Network Issues
- GraphQL endpoint connectivity problems
- Timeout handling for long-running queries
- Automatic retry for transient failures

### Query Issues
- Invalid GraphQL syntax detection
- Schema validation with detailed error messages
- Field existence verification

### Agent Limitations
- Iteration limits with intelligent fallback
- Time limits with partial result extraction
- Graceful handling of incomplete responses

## Performance Considerations

### Query Optimization
- Always use pagination (`first: N`) for collection queries
- Limit nested relationship depth to avoid expensive queries
- Use specific field selection rather than querying all fields
- Consider using `offset` for simple pagination scenarios

### Caching Strategy
- GraphQL schema introspection results are cached (1 hour TTL)
- Entity schema is loaded once per toolkit instance
- No query result caching (always fresh data)

### Resource Management
- Connection pooling for HTTP requests
- Automatic cleanup of resources
- Memory-efficient schema processing

## Comparison with Alternatives

| Feature | SubQL GraphQL Agent     | Generic GraphQL Tools | SQL Agents |
|---------|-------------------------|----------------------|------------|
| **Domain Specialization** | ✅ SubQuery SDK          | ❌ Generic | ❌ Database only |
| **Natural Language** | ✅ Full support          | ⚠️ Limited | ✅ SQL focused |
| **Schema Understanding** | ✅ PostGraphile + Entity | ⚠️ Basic introspection | ✅ Table schemas |
| **Query Validation** | ✅ Pre-execution         | ⚠️ Runtime only | ✅ SQL validation |
| **Relationship Handling** | ✅ @derivedFrom aware    | ❌ Manual | ✅ Foreign keys |
| **API Compatibility** | ✅ OpenAI compatible     | ❌ Custom only | ❌ Database specific |

## License

This project is licensed under the same terms as the parent project.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run linting and tests
5. Submit a pull request

## Testing & Validation

### Test Suite

Run the comprehensive test suite:

```bash
pytest tests/test_graphql_agent.py -v
```

Test coverage includes:
- ✅ Toolkit creation and configuration
- ✅ Schema info tool functionality
- ✅ Query validation with enhanced schema checking
- ✅ Query execution and error handling
- ✅ Complete workflow testing

### Manual Testing

Test the GraphQL tools directly:

```python
import asyncio
from graphql_agent import create_graphql_toolkit

async def test_tools():
    """Test GraphQL tools directly."""
    endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
    
    # Load entity schema (learn more: https://subquery.network/doc/indexer/build/graphql.html)
    with open("examples/schema.graphql", 'r') as f:
        entity_schema = f.read()
    
    # Create toolkit
    toolkit = create_graphql_toolkit(endpoint, entity_schema)
    tools = toolkit.get_tools()
    
    print(f"Available tools: {len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # Test schema info
    schema_tool = tools[0]
    result = await schema_tool._arun()
    print(f"\nSchema info: {result[:200]}...")

asyncio.run(test_tools())
```

### Endpoint Validation

Test the GraphQL endpoint directly:

```bash
curl -X POST https://index-api.onfinality.io/sq/subquery/subquery-mainnet \
  -H "Content-Type: application/json" \
  -d '{"query": "{ indexers(first: 1) { nodes { id } } }"}'
```

## Troubleshooting

### Common Issues & Solutions

#### 1. Missing Dependencies
```bash
# Error: No module named 'langchain_openai'
uv add langchain-openai

# Error: No module named 'graphql'
uv add graphql-core
```

#### 2. API Key Issues
```bash
# Error: "Invalid API key"
export OPENAI_API_KEY="sk-your-actual-key"

# Verify API key works
python -c "from langchain_openai import ChatOpenAI; print(ChatOpenAI().invoke('Hello'))"
```

#### 3. GraphQL Connection Issues
```bash
# Error: "GraphQL query failed"
# Check internet connection and endpoint
curl -I https://index-api.onfinality.io/sq/subquery/subquery-mainnet
```

#### 4. Agent Issues

**Problem**: Agent validation passes but execution doesn't happen
**Solution**: Updated prompts now emphasize that validation is NOT the final answer

**Problem**: Agent tries to use invalid "skip" action  
**Solution**: Fixed prompt format to go directly to Final Answer for non-relevant queries

**Problem**: Agent reaches iteration limit
**Solution**: Prompt now includes mandatory execution step after validation

#### 5. Import Path Issues
```bash
# Error: "attempted relative import with no known parent package"
# Make sure to run from correct directory
cd examples
python working_example.py
```

### Debug Mode

Enable verbose logging to see agent reasoning:

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows tool selections and reasoning
    max_iterations=10,
    return_intermediate_steps=True
)
```

### Performance Tips

1. **Schema Caching**: Schemas are automatically cached for 1 hour
2. **Query Optimization**: Use pagination and specific field selection
3. **Model Selection**: gpt-4o or stronger models recommended for best performance (gpt-4o-mini works but may have limitations)
4. **Rate Limiting**: Monitor OpenAI API usage to avoid limits

## Production Deployment

### Environment Setup

```bash
# Production environment variables
export OPENAI_API_KEY="your-production-key"
export LLM_MODEL="gpt-4o"
export PORT="8000"

# Optional: Custom endpoint and headers
export GRAPHQL_ENDPOINT="https://your-custom-endpoint.com/graphql"
export GRAPHQL_HEADERS='{"Authorization": "Bearer token"}'
```

### Security Considerations

1. **Input Validation**: All user inputs are validated before processing
2. **Query Sanitization**: GraphQL queries are validated against schema
3. **Rate Limiting**: Implement API rate limits for production use
4. **Error Handling**: Sensitive information is not exposed in error messages

### Monitoring

Key metrics to monitor:
- Query success/failure rates
- Average response times
- OpenAI API usage and costs
- GraphQL endpoint health
- Agent reasoning quality

### Scaling Considerations

1. **Horizontal Scaling**: Multiple server instances with load balancing
2. **Caching Strategy**: Redis for schema and query result caching
3. **Connection Pooling**: Efficient HTTP connection management
4. **Resource Limits**: Memory and CPU limits for agent execution

## Project Achievements

This project demonstrates several key technical achievements:

### 1. Advanced Schema Understanding
- ✅ **Entity Schema Integration**: Combines PostGraphile patterns with custom entity definitions
- ✅ **Intelligent Query Construction**: Automatically generates optimal GraphQL queries
- ✅ **Schema Validation**: Pre-execution validation prevents runtime errors

### 2. Natural Language Interface
- ✅ **Domain Specialization**: Focused on SubQuery Network terminology and concepts
- ✅ **Context Awareness**: Understands relationships between indexers, projects, and rewards
- ✅ **Error Recovery**: Graceful handling of invalid queries with helpful suggestions

### 3. Production-Ready Architecture
- ✅ **OpenAI Compatibility**: Standard API format for easy integration
- ✅ **Streaming Support**: Real-time response streaming for better UX
- ✅ **Comprehensive Error Handling**: Robust error detection and user feedback

### 4. Developer Experience
- ✅ **Easy Integration**: Simple toolkit creation with minimal setup
- ✅ **Flexible Usage**: Both interactive and API modes supported
- ✅ **Extensive Documentation**: Complete examples and troubleshooting guides

## Future Enhancements

### Short-term Improvements
1. **Conversation Memory**: Multi-turn conversation support
2. **Query Optimization**: Automatic performance optimization
3. **Custom Validators**: Domain-specific validation rules
4. **Enhanced Caching**: Intelligent query result caching

### Long-term Vision
1. **Multi-language Support**: Support for additional natural languages
2. **Visual Query Builder**: Web-based query construction interface
3. **Analytics Dashboard**: Query performance and usage analytics
4. **Plugin Architecture**: Extensible tool system for custom domains

## Support

For issues and questions:

1. **Documentation**: Check this README and example code in `examples/`
2. **Troubleshooting**: Review the troubleshooting section above
3. **Testing**: Run the test suite to verify installation
4. **Issues**: Open a GitHub issue with detailed information about your use case

### Getting Help

Include this information when reporting issues:
- Python version and OS
- Error messages and stack traces
- Steps to reproduce the problem
- Expected vs actual behavior

---

**Built for SubQuery Network** - Specialized GraphQL agent toolkit for blockchain indexing and staking data.