#!/usr/bin/env python3
"""
Working GraphQL LangChain Agent Example

This is a complete, working example that demonstrates how to use the GraphQL 
agent toolkit with LangChain to create a natural language interface for GraphQL APIs.

Usage:
    # Interactive mode
    uv run python examples/working_example.py
    
    # Demo mode  
    uv run python examples/working_example.py --demo
"""

import asyncio
import os
import sys

#
# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for development

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from graphql_agent import create_graphql_toolkit


class GraphQLAgent:
    """Simple GraphQL agent using LangChain."""
    
    def __init__(self, endpoint: str):
        """Initialize the agent."""
        self.endpoint = endpoint
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize LLM - read model from environment
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
        print(f"🤖 Using LLM model: {model_name}")
        
        # Load raw schema
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
        
        prompt_template = """You are a GraphQL assistant specialized in SubQuery Network data queries. You can help users find information about:
- Blockchain indexers and their status, performance, and rewards
- Projects, deployments, and project metadata
- Staking rewards, delegations, and era performance
- Network statistics, governance, and protocol data
- Withdrawal requests and commission tracking

Available tools:
{tools}

Tool names: {tool_names}

IMPORTANT: Before using any tools, evaluate if the user's question relates to SubQuery Network data.

IF NOT RELATED to SubQuery Network (general questions, other blockchains, personal advice, programming help, etc.):
- DO NOT use any tools
- Politely decline with: "I'm specialized in SubQuery Network data queries. I can help you with indexers, projects, staking rewards, and network statistics, but I cannot assist with [their topic]. Please ask me about SubQuery Network data instead."

IF RELATED to SubQuery Network data:
WORKFLOW:
1. Call graphql_schema_info ONCE to get raw schema and PostGraphile rules
2. Analyze the raw schema to identify @entity types and available fields  
3. Construct GraphQL queries using PostGraphile patterns - DO NOT call schema tool again
4. MANDATORY: Always validate queries with graphql_query_validator before executing
5. MANDATORY: After successful validation, MUST execute the query with graphql_execute to get actual data
6. If validation fails, you may use graphql_type_detail as FALLBACK to check specific types
7. Provide clear, helpful responses to users based on execution results

CRITICAL: Query validation is NOT the final answer. You MUST execute the validated query to get actual data for the user.

ADDITIONAL RULES:
- When users ask about "my rewards", "my data", etc. without providing addresses/IDs, you MUST ask for the missing information first
- NEVER fabricate or assume wallet addresses, indexer IDs, or other user-specific data

Format:
Question: {input}
Thought: [First: Is this about SubQuery Network data (indexers, projects, rewards, staking, etc.)? If NO, go directly to Final Answer with polite decline. If YES, plan your approach and proceed with workflow]
Action: [tool name - ONLY use if question is SubQuery Network related]
Action Input: [input]
Observation: [result]
Thought: I now have the answer
Final Answer: [user-friendly summary OR polite decline explanation]

Question: {input}
Thought: {agent_scratchpad}"""

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
        
        # Create executor with longer timeout
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=6,  # Reduce iterations to be more focused
            max_execution_time=180,  # 3 minutes timeout
            handle_parsing_errors=True,
            return_intermediate_steps=True  # Help with debugging
        )
    
    async def query(self, user_input: str) -> str:
        """Process a user query."""
        try:
            print(f"🔍 Processing query: {user_input}")
            result = await self.executor.ainvoke({"input": user_input})
            
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # If agent stopped due to limits but we have intermediate results
            if (not output or "iteration limit" in output or "time limit" in output or "stopped" in output.lower()) and intermediate_steps:
                print("🔧 Agent stopped due to limits, extracting results from intermediate steps...")
                
                # Look for GraphQL execution results in intermediate steps
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
                else:
                    # Look for any useful intermediate results
                    for step in intermediate_steps:
                        if len(step) >= 2:
                            action, observation = step[0], step[1]
                            if "successfully" in observation.lower() or len(observation) > 100:
                                summary_prompt = f"The user asked: '{user_input}'. This tool result might be helpful: {observation[:500]}... Please provide a helpful response."
                                from langchain.schema import HumanMessage
                                summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                                output = summary_response.content
                                print("✅ Generated summary from partial results")
                                break
            
            print(f"📤 Final result: {output[:200]}...")
            return output if output else "No response generated"
            
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            return f"Error processing query: {str(e)}"


async def interactive_demo():
    """Run an interactive demo."""
    print("🚀 Starting GraphQL LangChain Agent...")
    
    endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
    
    try:
        agent = GraphQLAgent(endpoint)
        print("✅ Agent ready!")
        
        print("\n📚 Example questions you can ask:")
        print("  • Show me 3 indexers with their IDs")
        print("  • What projects are available?")
        print("  • What types of data can I query?")
        print("  • Find indexers and their controllers")
        
        print("\n" + "="*50)
        print("Type your question or 'quit' to exit")
        print("="*50)
        
        while True:
            try:
                question = input("\n🙋 You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🤖 Agent: Thinking...")
                response = await agent.query(question)
                print(f"\n🤖 Agent: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")


async def simple_demo():
    """Run a simple demonstration."""
    print("🚀 Running simple demo...")
    
    endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
    
    try:
        agent = GraphQLAgent(endpoint)
        
        # Start with a very simple question first
        simple_questions = [
            "What are my boosted projects? my wallet: 0x31E99bdA5939bA2e7528707507b017f43b67F89B"
            # "Show me 2 indexers with their IDs",
            # "List the first 3 projects"
        ]
        
        for i, question in enumerate(simple_questions, 1):
            print(f"\n{'='*80}")
            print(f"🧪 Simple Test {i}/{len(simple_questions)}: {question}")
            
            try:
                # Add timeout for each individual query
                import asyncio
                response = await asyncio.wait_for(agent.query(question), timeout=200)
                print(f"✅ Success!")
                print(f"🤖 Response: {response[:500]}...")
            except asyncio.TimeoutError:
                print(f"⏰ Timeout after 200 seconds")
            except Exception as e:
                print(f"❌ Failed: {e}")
            
            print("-" * 80)
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")


async def test_specific_queries():
    """Test specific GraphQL queries to validate functionality."""
    print("🧪 Testing specific GraphQL functionality...")
    
    endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
    
    try:
        agent = GraphQLAgent(endpoint)
        
        # Test queries that should work
        working_queries = [
            "What is my rewards of last era?",
            # "List the available query types in this GraphQL schema",
            # "What information can I get about projects?"
        ]
        
        # Test queries that might have issues
        edge_case_queries = [
            "Find indexers with invalid field names",  # Should trigger validation
            "Show me data that doesn't exist",  # Should handle gracefully
        ]
        
        print("\n📋 Testing Working Queries:")
        for query in working_queries:
            print(f"\n🔍 Testing: {query}")
            try:
                response = await agent.query(query)
                print(f"✅ Result: {response[:200]}..." if len(response) > 200 else f"✅ Result: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n📋 Testing Edge Cases:")
        for query in edge_case_queries:
            print(f"\n🔍 Testing: {query}")
            try:
                response = await agent.query(query)
                print(f"✅ Handled: {response[:200]}..." if len(response) > 200 else f"✅ Handled: {response}")
            except Exception as e:
                print(f"⚠️ Expected error: {e}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")


async def basic_tool_examples():
    """Show basic usage examples of all tools."""
    print("🔧 Basic GraphQL Tool Examples...")
    
    endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
    
    # Load raw schema
    schema_file_path = os.path.join(os.path.dirname(__file__), "schema.graphql")
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        entity_schema = f.read()
    
    toolkit = create_graphql_toolkit(endpoint, entity_schema)
    tools = toolkit.get_tools()
    
    print(f"Available tools: {[tool.name for tool in tools]}")
    
    try:
        # Example 1: Get schema overview
        schema_tool = next(tool for tool in tools if tool.name == "graphql_schema_info")
        schema_info = await schema_tool._arun()
        print("\n=== Schema Overview ===")
        print(schema_info[:500] + "..." if len(schema_info) > 500 else schema_info)
        
        # Example 2: Validate a simple query
        validator_tool = next(tool for tool in tools if tool.name == "graphql_query_validator")
        test_query = "{ indexers(first: 2) { nodes { id } } }"
        validation_result = await validator_tool._arun(test_query)
        print("\n=== Query Validation ===")
        print(validation_result)
        
        # Example 3: Execute the query (if validation passed)
        if "✅" in validation_result:
            execute_tool = next(tool for tool in tools if tool.name == "graphql_execute")
            execution_result = await execute_tool._arun(test_query)
            print("\n=== Query Execution ===")
            print(execution_result[:300] + "..." if len(execution_result) > 300 else execution_result)
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")


async def cli_tool_test():
    """Command line interface for testing individual tools."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python examples/working_example.py --cli <tool_name> <args...>")
        print("Available tools: schema_info, type_detail, validate, execute")
        return
    
    endpoint = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
    
    # Load raw schema
    schema_file_path = os.path.join(os.path.dirname(__file__), "schema.graphql")
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        entity_schema = f.read()
    
    toolkit = create_graphql_toolkit(endpoint, entity_schema)
    tools = {tool.name.replace("graphql_", ""): tool for tool in toolkit.get_tools()}
    
    tool_name = sys.argv[2]
    if tool_name not in tools:
        print(f"Unknown tool: {tool_name}")
        print(f"Available: {list(tools.keys())}")
        return
    
    tool = tools[tool_name]
    
    try:
        if tool_name == "schema_info":
            result = await tool._arun()
            print(result)
        elif tool_name == "type_detail":
            if len(sys.argv) < 4:
                print("Usage: --cli type_detail <type_name>")
                return
            result = await tool._arun(sys.argv[3])
            print(result)
        elif tool_name == "validate":
            if len(sys.argv) < 4:
                print("Usage: --cli validate <query>")
                return
            result = await tool._arun(sys.argv[3])
            print(result)
        elif tool_name == "execute":
            if len(sys.argv) < 4:
                print("Usage: --cli execute <query>")
                return
            result = await tool._arun(sys.argv[3])
            print(result)
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphQL LangChain Agent Demo")
    parser.add_argument("--demo", action="store_true", help="Run comprehensive demo with all suggested queries")
    parser.add_argument("--test", action="store_true", help="Run specific functionality tests")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode (default)")
    parser.add_argument("--examples", action="store_true", help="Show basic tool usage examples")
    parser.add_argument("--cli", action="store_true", help="Run CLI tool testing mode")
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        print("   Then run: export OPENAI_API_KEY='your-key-here'")
        print("\n💡 Optionally set LLM model:")
        print("   export LLM_MODEL='gpt-4'  # or gpt-4o-mini (default)")
        print("\n💡 For testing without OpenAI, you can run:")
        print("   uv run python -m pytest tests/test_graphql_agent.py -v")
        return
    
    # Determine what to run
    if args.cli:
        print("🔧 Running CLI tool testing mode...")
        asyncio.run(cli_tool_test())
    elif args.examples:
        print("📚 Showing basic tool usage examples...")
        asyncio.run(basic_tool_examples())
    elif args.demo:
        print("🚀 Running comprehensive demo with all suggested natural language queries...")
        asyncio.run(simple_demo())
    elif args.test:
        print("🧪 Running specific functionality tests...")
        asyncio.run(test_specific_queries())
    else:
        print("🚀 Starting interactive mode...")
        asyncio.run(interactive_demo())


if __name__ == "__main__":
    main()