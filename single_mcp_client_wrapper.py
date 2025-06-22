# File: python_src/single_mcp_client_wrapper.py
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List, Any
from contextlib import AsyncExitStack

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# LangChain Imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

# Ensure SCRIPT_DIR is available if this is run standalone or for relative paths
# This assumes it's in python_src
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError: # if __file__ is not defined (e.g. in REPL)
    SCRIPT_DIR = Path.cwd()


class SingleMCPClientWrapper:
    def __init__(self, model: BaseChatModel, server_script_path: str):
        """
        Initializes the Single MCP Client Wrapper.

        Args:
            model: A LangChain compatible chat model instance.
            server_script_path: Relative or absolute path to the single MCP server script.
        """
        self.model: BaseChatModel = model
        self.server_script_full_path: Path = self._resolve_server_path(server_script_path)

        self.session: Optional[ClientSession] = None
        self.tools: List[BaseTool] = []
        self.agent: Optional[Any] = None
        self.exit_stack: Optional[AsyncExitStack] = None # Initialize as None
        self.stdio_transport_tuple = None # To store (stdio, write) tuple

    def _resolve_server_path(self, server_script_path: str) -> Path:
        server_path_obj = Path(server_script_path)
        if not server_path_obj.is_absolute():
            # Assume it's relative to SCRIPT_DIR (python_src)
            return (SCRIPT_DIR / server_script_path).resolve()
        return server_path_obj.resolve()

    async def connect(self):
        """
        Connects to the MCP server, loads tools, and creates the LangChain agent.
        """
        if self.agent: # Already connected and agent created
            print("Already connected and agent initialized.")
            return

        print(f"Wrapper: Attempting to connect to server script: {self.server_script_full_path}")
        if not self.server_script_full_path.exists():
            print(f"Error: Server script not found at {self.server_script_full_path}")
            raise FileNotFoundError(f"Server script not found at {self.server_script_full_path}")

        is_python = self.server_script_full_path.name.endswith('.py')
        is_js = self.server_script_full_path.name.endswith('.js') # Though likely only Python used here
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = sys.executable if is_python else "node" # Use current python for .py

        # Prepare environment for subprocess
        current_env = os.environ.copy()
        existing_pythonpath = current_env.get("PYTHONPATH", "")
        script_dir_str = str(SCRIPT_DIR)
        if script_dir_str not in existing_pythonpath.split(os.pathsep):
            current_env["PYTHONPATH"] = f"{script_dir_str}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else script_dir_str

        server_params = StdioServerParameters(
            command=command,
            args=["-u", str(self.server_script_full_path)], # -u for unbuffered Python output
            cwd=str(self.server_script_full_path.parent),
            env=current_env # Pass the modified environment
        )
        print(f"Wrapper: Starting server with: cmd='{command}', args='{['-u', str(self.server_script_full_path)]}', cwd='{str(self.server_script_full_path.parent)}', PYTHONPATH='{current_env.get('PYTHONPATH')}'")

        self.exit_stack = AsyncExitStack() # Create new stack for this connection attempt
        try:
            print("Wrapper: Initializing stdio transport...")
            self.stdio_transport_tuple = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio_read, stdio_write = self.stdio_transport_tuple
            print("Wrapper: Stdio transport initialized.")

            print("Wrapper: Initializing MCP session...")
            self.session = await self.exit_stack.enter_async_context(ClientSession(stdio_read, stdio_write))
            await self.session.initialize()
            print("Wrapper: MCP Session initialized.")

            print("Wrapper: Loading MCP tools via LangChain adapter...")
            self.tools = await load_mcp_tools(self.session)
            print(f"Wrapper: Tools loaded: {[tool.name for tool in self.tools]}")

            if not self.tools:
                print("Wrapper: Warning: No tools loaded from the MCP server.")
                self.agent = None
                # Consider raising an error if tools are expected but not found
                # For now, let it proceed, process_query will handle no agent.
                return

            print("Wrapper: Creating LangChain agent...")
            self.agent = create_react_agent(self.model, self.tools)
            if self.agent:
                print("Wrapper: LangChain agent created successfully.")
            else:
                print("Wrapper: Error: Failed to create LangChain agent.")
                # This case might be hard to reach if create_react_agent always returns or raises

        except Exception as e:
            print(f"Wrapper: Error during connection/setup: {e}")
            await self.cleanup() # Ensure resources are cleaned up on partial failure
            raise # Re-raise to be handled by the caller

    async def process_query(self, query: str) -> str:
        """
        Processes a query using the configured LangChain agent and MCP tools.
        """
        if not self.agent:
            # Try to connect if not already connected (e.g., first call)
            # Or raise an error if connect should have been called explicitly
            print("Wrapper: Agent not initialized. Attempting to connect first...")
            try:
                await self.connect()
                if not self.agent: # If connection failed to create agent
                    return "Error: Agent not initialized after connection attempt. Please check server and tool loading."
            except Exception as e:
                return f"Error: Failed to connect or initialize agent before query: {e}"

        if not self.session: # Should be set if agent is set from connect()
             return "Error: MCP session is not available, though agent exists. This is an inconsistent state."

        messages = [HumanMessage(content=query)]
        print(f"\nWrapper: Invoking agent with query: '{query}'")
        try:
            response = await self.agent.ainvoke({"messages": messages})

            if isinstance(response, dict) and "messages" in response and response["messages"]:
                # The actual tool output that the agent used might be in earlier messages.
                # The final response is the LLM's synthesis.
                # For `create_react_agent`, the last message is usually the final answer.
                final_answer_message = response["messages"][-1]
                final_response_content = final_answer_message.content

                print(f"Wrapper: Agent raw response dict: {response}") # Log raw for debugging
                print(f"Wrapper: Agent final answer content: {final_response_content}")
                return final_response_content
            else:
                print(f"Wrapper: Unexpected agent response structure: {response}")
                return f"Agent finished, but response format was unexpected: {str(response)}"

        except Exception as e:
            print(f"Wrapper: Error during agent invocation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing query: {str(e)}"

    async def cleanup(self):
        """Cleans up resources."""
        print("\nWrapper: Cleaning up resources...")
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None # Reset stack
        self.session = None
        self.agent = None
        self.tools = []
        self.stdio_transport_tuple = None
        print("Wrapper: Resources cleaned up.")