# SubQL GraphQL Agent

A specialized GraphQL agent toolkit for LLM interactions with SubQuery SDK-generated APIs, featuring natural language query capabilities and OpenAI-compatible API endpoints.

## Overview

This toolkit provides LLM agents with the ability to interact with any GraphQL API built with SubQuery SDK through natural language, automatically understanding schemas, validating queries, and executing complex GraphQL operations.

### Key Features

- **Natural Language Interface**: Ask questions about blockchain data in plain English
- **Automatic Schema Understanding**: Agents learn PostGraphile v4 patterns and SubQuery entity schemas
- **Query Generation & Validation**: Converts natural language to valid GraphQL queries with built-in validation
- **OpenAI-Compatible API**: FastAPI server with streaming and non-streaming endpoints
- **SubQuery SDK Optimized**: Works with any project built using SubQuery SDK (Ethereum, Polkadot, Cosmos, etc.)

## Design Philosophy

### Solving the GraphQL Schema Size Problem

Traditional GraphQL agents face a fundamental challenge: **schema size exceeds LLM context limits**. Most GraphQL APIs have introspection schemas that are tens of thousands of tokens, making them:

- **Too large** for most commercial LLMs (exceeding context windows)
- **Too expensive** for cost-effective query generation
- **Too noisy** for reliable query construction (low signal-to-noise ratio)

### Our Innovative Approach: Entity Schema + Rules

Instead of using raw GraphQL introspection schemas, we developed a **compressed, high-density schema representation**:

#### 🎯 **Entity Schema as Compressed Knowledge**
- **Compact Format**: 100x smaller than full introspection schemas
- **Domain-Specific**: Contains project-specific entities and relationships
- **High Information Density**: Only essential types, relationships, and patterns
- **Rule-Based**: Combined with PostGraphile v4 patterns for query construction

#### 📊 **Size Comparison**
```
Traditional Approach:
├── Full GraphQL Introspection: ~50,000+ tokens
├── Context Window Usage: 80-95%
└── Result: Often fails or generates invalid queries

Our Approach:
├── Entity Schema: ~500-1,000 tokens  
├── PostGraphile Rules: ~200-300 tokens
├── Context Window Usage: 5-10%
└── Result: Reliable, cost-effective query generation
```

#### 🧠 **How It Works**

1. **Entity Schema Teaching**: LLM learns project's domain model from compressed schema
2. **Pattern Recognition**: PostGraphile v4 rules guide query structure
3. **Intelligent Construction**: Agent builds queries using learned patterns
4. **Validation**: Real-time schema validation ensures correctness

#### ⚡ **Benefits**

- **💰 Cost Effective**: 10-20x lower token usage than traditional approaches
- **🎯 Higher Accuracy**: Domain-specific knowledge reduces errors
- **⚡ Faster Responses**: Smaller context means faster processing
- **🔄 Scalable**: Works consistently across different LLM models

#### 🔧 **Technical Innovation**

```python
# Traditional approach (fails with large schemas)
raw_schema = introspect_graphql_schema()  # 50k+ tokens
context = f"Schema: {raw_schema}\nQuestion: {user_query}"  # Exceeds limits

# Our approach (works reliably)
entity_schema = load_project_entities()   # 500 tokens
rules = get_postgraphile_patterns()       # 300 tokens  
context = f"Entities: {entity_schema}\nRules: {rules}\nQuestion: {user_query}"
```

### Limitations and Extensibility

#### 🎯 **Current Scope**
- **SubQuery SDK Optimized**: Specifically designed for APIs built with SubQuery SDK
- **PostGraphile v4**: Leverages PostGraphile v4 patterns that SubQuery SDK generates
- **Entity-Focused**: Works best with well-defined blockchain entity relationships

#### 🚀 **Extension Potential**
The same philosophy can be applied to other GraphQL ecosystems:

- **Hasura**: Could use Hasura-specific schema compression + rules
- **Apollo Federation**: Could compress federated schemas with service patterns  
- **Custom GraphQL**: Could extract domain models + API patterns
- **Other ORMs**: Could adapt for Prisma, TypeORM, or other ORM-generated schemas

#### 🔮 **Future Directions**
```
SubQuery SDK Agent (Current)
├── Entity Schema: Project-specific domain models
├── Rules: PostGraphile v4 patterns
└── Scope: Any SubQuery SDK-generated API

Generic GraphQL Agent (Future)
├── Schema Compression: Auto-extract domain models
├── Pattern Recognition: Detect API patterns automatically  
├── Multi-Domain: Support multiple GraphQL styles
└── Scope: Any GraphQL API
```

### Why This Matters

This approach represents a **paradigm shift** in GraphQL agent design:

- **From**: "Give LLM everything and hope it works"
- **To**: "Give LLM exactly what it needs to succeed"

The result is a more **reliable**, **cost-effective**, and **performant** GraphQL agent that can actually be deployed in production environments.

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
# Note: This example uses SubQuery Network's schema - replace with your own project's schema
with open("examples/schema.graphql", 'r') as f:
    entity_schema = f.read()

# Create toolkit
# Note: This example uses SubQuery Network's API - replace with your own project's endpoint  
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

**Note**: These examples are from the SubQuery Network demo. For your own project, the queries would be specific to your indexed blockchain data.

The example agent can handle queries like:

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

### Custom Agent Prompts

**Important**: The example prompts are specifically tailored for the SubQuery Network example to help the LLM accurately determine its capabilities. You should customize the prompt for your specific project:

```python
# Customize this prompt for your project's domain
prompt_template = """You are a GraphQL assistant specialized in [YOUR PROJECT] data queries. You can help users find information about:
- [List your project's main entities and use cases]
- [Specific data types your project indexes]
- [Key relationships and metrics available]

Available tools: {tools}
Tool names: {tool_names}

IMPORTANT: Before using any tools, evaluate if the user's question relates to [YOUR PROJECT] data.

IF NOT RELATED to [YOUR PROJECT] (general questions, other projects, personal advice, etc.):
- DO NOT use any tools  
- Politely decline with: "I'm specialized in [YOUR PROJECT] data queries. I can help you with [list key capabilities], but I cannot assist with [their topic]. Please ask me about [YOUR PROJECT] data instead."

IF RELATED to [YOUR PROJECT] data:
[Rest of workflow remains the same]
"""
```

**Why Domain-Specific Prompts Matter:**
- **Better Boundary Recognition**: LLM can accurately determine when it should/shouldn't help
- **Improved User Experience**: Clear communication about capabilities and limitations  
- **Reduced Hallucination**: LLM won't attempt to answer questions outside its domain
- **Professional Responses**: Consistent, helpful decline messages for out-of-scope requests

**Example Customizations:**
- **DeFi Project**: "specialized in DeFi protocol data... trading volumes, liquidity pools, yield farming..."
- **NFT Marketplace**: "specialized in NFT marketplace data... collections, sales, floor prices..."  
- **Gaming Project**: "specialized in blockchain gaming data... players, items, achievements..."

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