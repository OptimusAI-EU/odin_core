# File: python_src/app.py
import asyncio
import os
import sys
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import time

# In app.py (or single_server_ui.py)

# ... other imports ...
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq # <--- ADD THIS IMPORT

# ... (dotenv loading) ...

# --- Backend Logic (run_mcp_task) ---
async def run_mcp_task(
    agent_type: str,
    main_input: str,
    agent_model_name: str, # This will now be more important
    # ... other parameters ...
    progress: gr.Progress,
) -> str:
    progress(0, desc="Initializing...")
    # ... (initial prints) ...

    # --- Agent LLM Configuration ---
    try:
        progress(0.1, desc=f"Initializing agent LLM ({agent_model_name})...")
        agent_llm = None # Initialize to None

        # Check for Groq models (example: "groq/llama-3.3-70b-versatile")
        if agent_model_name.startswith("groq/"):
            groq_model_name_actual = agent_model_name.split("/", 1)[1] # Get "llama-3.3-70b-versatile"
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not set in environment for Groq model.")
            agent_llm = ChatGroq(
                model_name=groq_model_name_actual, # Pass the actual model name
                api_key=groq_api_key,
                temperature=0 # Or your desired temperature
            )
            print(f"Using Groq model: {groq_model_name_actual}")
        # Check for OpenRouter models
        elif any(name_part in agent_model_name.lower() for name_part in ["deepseek", "meta-llama/llama-4", "google/gemini"]):
            # Your existing OpenRouter logic
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set in environment for OpenRouter model.")
            agent_llm = ChatOpenAI(model=agent_model_name, api_key=api_key, base_url=base_url)
            print(f"Using OpenRouter model via ChatOpenAI: {agent_model_name}")
        # Default to Ollama for others (like "qwen2.5-coder")
        else:
            # Check if it's an OpenAI model not via OpenRouter (e.g., "gpt-4o")
            if agent_model_name.startswith("gpt-"):
                 openai_api_key = os.getenv("OPENAI_API_KEY")
                 if not openai_api_key:
                     raise ValueError("OPENAI_API_KEY not set in environment for direct OpenAI model.")
                 agent_llm = ChatOpenAI(model=agent_model_name, api_key=openai_api_key)
                 print(f"Using direct OpenAI model: {agent_model_name}")
            else:
                # Fallback to Ollama
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                agent_llm = ChatOllama(model=agent_model_name, base_url=ollama_base_url)
                print(f"Using Ollama model: {agent_model_name} from {ollama_base_url}")

        if agent_llm is None: # Should not happen if logic is correct
            raise ValueError(f"Could not determine LLM type for model name: {agent_model_name}")

        print(f"Agent LLM successfully initialized: {agent_llm.__class__.__name__}")

    except Exception as e:
        error_msg = f"Error initializing agent LLM '{agent_model_name}': {e}"
        print(error_msg, file=sys.stderr)
        return error_msg

    # ... (rest of your run_mcp_task logic, using 'agent_llm' where you previously had 'agent_model')
    # For example, when creating the client wrapper:
    # client_wrapper = SingleMCPClientWrapper(model=agent_llm, server_script_path=...) # For single server
    # OR when creating the agent directly if not using a wrapper in that part:
    # mcp_client_for_sse.model = agent_llm # If you had such a structure for SSE
    # OR for the multi-server stdio:
    # client_wrapper = MultiServerMCPClientWrapper(model=agent_llm)

    # Make sure to pass `agent_llm` to whatever needs the LLM instance.

    # ... (your existing logic for Browser Use or Stdio based servers) ...
    # Ensure that `agent_llm` is the model instance passed to `SingleMCPClientWrapper`,
    # `MultiServerMCPClientWrapper`, or directly to `create_react_agent` if you
    # construct the agent outside those wrappers in some cases (like SSE).

    # Example for the single_server_ui.py test structure:
    if agent_type == "Software Company": # Or if testing a specific path
        software_company_script = "mcp_software_company.py" # Or appropriate script
        client_wrapper = SingleMCPClientWrapper(model=agent_llm, server_script_path=software_company_script)
        # ... rest of the single server logic using client_wrapper ...
        # (the rest of your single_server_ui.py's run_mcp_task logic)
        final_result = "Error: Software company task did not complete."
        try:
            # ... (connect, process_query, cleanup as before) ...
            await client_wrapper.connect()
            if client_wrapper.agent:
                response = await client_wrapper.process_query(message_content) # message_content needs to be defined
                final_result = response
            else:
                final_result = "Error: Agent not initialized in single server wrapper."
        except Exception as e_task:
            final_result = f"Error in Software Company task: {e_task}"
        finally:
            await client_wrapper.cleanup()
        return final_result

    elif agent_type == "Browser Use":
        # ... your Browser Use logic, ensuring agent_llm is used for create_react_agent ...
        # Example:
        # async with SomeSSEClient(server_configs) as sse_tool_client:
        #     tools = await sse_tool_client.get_tools()
        #     agent = create_react_agent(agent_llm, tools) <--- Use agent_llm
        #     response = await agent.ainvoke({"messages": messages})
        #     final_result = response["messages"][-1].content
        #     return final_result
        return "Browser Use logic placeholder - adapt with agent_llm"

    # Add other agent types if you're modifying the full multi-server app.py
    else:
        # Your existing multi-server stdio logic using MultiServerMCPClientWrapper
        # Make sure it receives `agent_llm`
        # client_wrapper = MultiServerMCPClientWrapper(model=agent_llm)
        # ... (connect_to_servers, process_query, cleanup) ...
        return f"Agent type {agent_type} (Multi-Server STDIO) logic placeholder - adapt with agent_llm"

# --- Gradio UI Definition (create_ui) ---
def create_ui():
    # ... (your existing UI blocks) ...
    agent_model_name_dropdown = gr.Dropdown( # Make sure this variable name is what you use
        label="Agent LLM Model",
        choices=[
            # Ollama models (local)
            "qwen2.5-coder",
            "phi4-mini",
            "llama3",
            # Groq models (add the prefix "groq/")
            "llama-3.3-70b-versatile", # <--- ADDED GROQ MODEL
            "groq/llama3-8b-8192",
            "groq/llama3-70b-8192",
            "groq/mixtral-8x7b-32768",
            "groq/gemma-7b-it",
            # OpenRouter models
            "deepseek/deepseek-r1-0528:free", # Ensure your parsing logic handles this structure
            "meta-llama/llama-4-scout:free",  # And this
            "google/gemini-2.5-pro-exp-03-25:free", # And this
            # Direct OpenAI models
            "gpt-4o",
            "gpt-3.5-turbo"
        ],
        value="llama-3.3-70b-versatile" # <--- Default to a working one for now
    )
    # ... (rest of your UI, ensure `agent_model_name_dropdown` is passed to `submit_btn.click`)
    # Ensure inputs for submit_btn.click includes this dropdown:
    # inputs=[..., agent_model_name_dropdown, ...],

    return interface# File: single_server_ui.py (or whatever you named your single-server test UI file)
import asyncio
import os
import sys
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import time

# --- Path Setup & SCRIPT_DIR ---
# This ensures that imports relative to this script's directory work,
# especially for finding SingleMCPClientWrapper and mcp_*.py scripts.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# --- Imports for LLMs and Client Wrapper ---
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq # For Groq models

# Import the SingleMCPClientWrapper (assuming it's in the same directory or SCRIPT_DIR is in sys.path)
from single_mcp_client_wrapper import SingleMCPClientWrapper

# --- Load Environment Variables ---
# Ensure .env is in SCRIPT_DIR (e.g., your python_src or odin_core directory)
dotenv_path = SCRIPT_DIR / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"Loaded .env file from: {dotenv_path}")


# --- Backend Logic: run_mcp_task ---
async def run_mcp_task(
    # Keep all original parameters needed by any agent type,
    # even if this test focuses on Software Company.
    # The UI will send them.
    agent_type: str, # For this test, it will be hardcoded to "Software Company" in the call
    main_input: str,
    agent_model_name: str,
    # MetaGPT specific
    investment: float,
    n_round: int,
    project_name: str,
    code_review: bool,
    run_tests: bool,
    implement: bool,
    inc: bool,
    project_path: str,
    reqa_file: str,
    max_auto_summarize_code: int,
    recover_path: str,
    # Researcher specific (needed for full signature even if not used in this test call)
    report_type: str,
    progress: gr.Progress,
) -> str:
    progress(0, desc="Initializing...")
    print(f"\n--- Received Task (Single Server Test Mode) ---")
    print(f"Intended Agent Type (from UI, may be overridden for test): {agent_type}") # Log what UI sent
    print(f"Main Input: {main_input}")
    print(f"Selected Agent Model: {agent_model_name}")

    # --- Agent LLM Configuration ---
    agent_llm = None
    try:
        progress(0.1, desc=f"Initializing agent LLM ({agent_model_name})...")
        if agent_model_name.startswith("groq/"):
            groq_model_name_actual = agent_model_name.split("/", 1)[1]
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not set for Groq model.")
            agent_llm = ChatGroq(model_name=groq_model_name_actual, api_key=groq_api_key, temperature=0)
            print(f"Using Groq model: {groq_model_name_actual}")
        elif any(name_part in agent_model_name.lower() for name_part in ["deepseek", "meta-llama/llama-4", "google/gemini"]): # OpenRouter
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set for OpenRouter model.")
            agent_llm = ChatOpenAI(model=agent_model_name, api_key=api_key, base_url=base_url, temperature=0)
            print(f"Using OpenRouter model: {agent_model_name}")
        elif agent_model_name.startswith("gpt-"): # Direct OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not set for direct OpenAI model.")
            agent_llm = ChatOpenAI(model=agent_model_name, api_key=openai_api_key, temperature=0)
            print(f"Using direct OpenAI model: {agent_model_name}")
        else: # Fallback to Ollama
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            agent_llm = ChatOllama(model=agent_model_name, base_url=ollama_base_url, temperature=0)
            print(f"Using Ollama model: {agent_model_name} from {ollama_base_url}")

        if agent_llm is None: # Should be caught by specific errors above
            raise ValueError(f"Could not determine LLM type for: {agent_model_name}")
        print(f"Agent LLM initialized: {agent_llm.__class__.__name__}")

    except Exception as e:
        error_msg = f"Error initializing LLM '{agent_model_name}': {e}"
        print(error_msg, file=sys.stderr)
        return error_msg

    # --- Single Server Test for "Software Company" ---
    # Hardcode the script to test and the parameters it needs.
    # The `agent_type` from UI is ignored for this specific test's backend logic.
    current_test_agent_type = "Software Company" # For clarity in this test
    software_company_script_name = "mcp_software_company.py"

    # Construct the tool invocation message for MetaGPT
    # This comes from the UI's main_input and metagpt-specific inputs
    param_details = f"idea='{main_input}'"
    if project_name: param_details += f", project_name='{project_name}'"
    if project_path: param_details += f", project_path='{project_path}'"
    if reqa_file: param_details += f", reqa_file='{reqa_file}'"
    if recover_path: param_details += f", recover_path='{recover_path}'"
    param_details += f", investment={investment}"
    param_details += f", n_round={n_round}"
    param_details += f", code_review={code_review}"
    param_details += f", run_tests={run_tests}"
    param_details += f", implement={implement}"
    param_details += f", inc={inc}"
    param_details += f", max_auto_summarize_code={max_auto_summarize_code}"
    # The tool name expected by the agent is 'execute_metagpt'
    message_content_for_agent = f"Use the execute_metagpt tool with parameters: {param_details}."
    print(f"Constructed message for agent: {message_content_for_agent}")

    client_wrapper = SingleMCPClientWrapper(model=agent_llm, server_script_path=software_company_script_name)
    final_task_result = f"Error: {current_test_agent_type} task did not complete."

    try:
        progress(0.2, desc=f"Connecting to {current_test_agent_type} server...")
        await client_wrapper.connect()
        print("Wrapper connection attempt finished.")

        if client_wrapper.agent:
            progress(0.5, desc="Processing query with agent...")
            print(f"Processing query for {current_test_agent_type} via single wrapper...")
            response = await client_wrapper.process_query(message_content_for_agent)
            final_task_result = response
            print(f"DEBUG [run_mcp_task]: Final result from wrapper: {final_task_result!r}")
        else:
            final_task_result = "Error: Agent could not be initialized in wrapper."
            print(f"DEBUG [run_mcp_task]: Agent not initialized: {final_task_result!r}")

    except FileNotFoundError as e_fnf:
        error_msg = f"Server script not found: {e_fnf}"
        print(error_msg, file=sys.stderr)
        final_task_result = error_msg
    except Exception as e_task:
        error_msg = f"Error during {current_test_agent_type} task execution: {e_task}"
        print(error_msg, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        final_task_result = error_msg
        print(f"DEBUG [run_mcp_task Exception]: {final_task_result!r}")
    finally:
        progress(0.9, desc="Cleaning up client resources...")
        print("Cleaning up single server client wrapper resources...")
        await client_wrapper.cleanup()
        print("Wrapper cleanup complete.")
        progress(1, desc="Task Complete.")

    return final_task_result


# --- Gradio UI Definition ---
def create_ui():
    with gr.Blocks(title="ODIN Core UI (Single Server Test)") as interface:
        gr.Markdown("# ODIN Core (Single Server Test Mode)")
        gr.Markdown("This UI is testing the 'Software Company' agent using the `SingleMCPClientWrapper`.\n"
                    "Select an LLM model. Other agent types from dropdown are ignored by backend for this test.")

        with gr.Row():
            with gr.Column(scale=1):
                # Agent type dropdown is present but its value is overridden in run_mcp_task for this test
                agent_type_dropdown_ui = gr.Dropdown(
                    label="Agent Type (Backend will use 'Software Company')",
                    choices=["Software Company", "Deep Researcher", "Data Interpreter", "Browser Use"],
                    value="Software Company" # UI default
                )
                agent_model_name_dropdown = gr.Dropdown(
                    label="Agent LLM Model",
                    choices=[
                        "qwen2.5-coder", "phi4-mini", "llama3", # Ollama
                        "groq/llama-3.3-70b-versatile", # Groq
                        "groq/llama3-8b-8192", "groq/llama3-70b-8192", "groq/mixtral-8x7b-32768", "groq/gemma-7b-it",
                        "deepseek/deepseek-r1-0528:free", "meta-llama/llama-4-scout:free", "google/gemini-2.5-pro-exp-03-25:free", # OpenRouter
                        "gpt-4o", "gpt-3.5-turbo" # Direct OpenAI
                    ],
                    value="groq/llama-3.3-70b-versatile" # Default to a working one
                )
                main_input_textbox = gr.Textbox(label="Idea for Software Company", lines=3, placeholder="e.g., Create a simple 2048 game")

                # --- MetaGPT Specific Inputs (only ones relevant for this test) ---
                with gr.Group(visible=True) as metagpt_group:
                     gr.Markdown("### Software Company Options")
                     metagpt_project_name = gr.Textbox(label="Project Name (Optional)")
                     metagpt_investment = gr.Slider(label="Investment ($)", minimum=1.0, maximum=100.0, value=3.0, step=0.5)
                     metagpt_n_round = gr.Slider(label="Rounds", minimum=1, maximum=20, value=5, step=1)
                     metagpt_code_review = gr.Checkbox(label="Enable Code Review", value=True)
                     metagpt_run_tests = gr.Checkbox(label="Enable Run Tests", value=False)
                     metagpt_implement = gr.Checkbox(label="Enable Implementation", value=True)
                     metagpt_inc = gr.Checkbox(label="Incremental Mode", value=False)
                     metagpt_project_path = gr.Textbox(label="Project Path (Optional)")
                     metagpt_reqa_file = gr.Textbox(label="Requirements File Path (Optional)")
                     metagpt_max_auto_summarize = gr.Number(label="Max Auto Summarize Code", value=0, precision=0)
                     metagpt_recover_path = gr.Textbox(label="Recover Path (Optional)")

                # Dummy/Placeholder for researcher_report_type to match full signature
                # This group can be hidden as it's not used by the Software Company test
                with gr.Group(visible=False) as researcher_group_hidden:
                    researcher_report_type_dummy = gr.Textbox(label="Report Type (Not Used)", value="research_report")


                submit_btn = gr.Button("Run Software Company Task")

            with gr.Column(scale=2):
                output_textbox = gr.Textbox(label="Output", lines=30, interactive=False) # Increased lines

        # Wrapper function for the submit button click
        async def run_task_for_ui(
            # Parameters must match the order and names of UI components in the inputs list of submit_btn.click
            agent_type_val_from_ui, # Will be ignored by run_mcp_task's hardcoding
            main_input_val,
            agent_model_name_val,
            investment_val, n_round_val, project_name_val, code_review_val, run_tests_val, implement_val,
            inc_val, project_path_val, reqa_file_val, max_auto_summarize_val, recover_path_val,
            report_type_val_from_ui, # From the dummy/hidden researcher input
            progress=gr.Progress(track_tqdm=True)
        ):
            # Line buffering for better real-time logs if subprocesses print a lot
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
            print("Gradio UI: Submit button clicked. Calling run_mcp_task...")

            # Call the main backend logic. For this test, agent_type is effectively hardcoded inside run_mcp_task.
            # We pass all params so the signature matches.
            result = await run_mcp_task(
                agent_type="Software Company", # Hardcode for this test run
                main_input=main_input_val,
                agent_model_name=agent_model_name_val,
                investment=investment_val,
                n_round=n_round_val,
                project_name=project_name_val,
                code_review=code_review_val,
                run_tests=run_tests_val,
                implement=implement_val,
                inc=inc_val,
                project_path=project_path_val,
                reqa_file=reqa_file_val,
                max_auto_summarize_code=max_auto_summarize_val,
                recover_path=recover_path_val,
                report_type=report_type_val_from_ui, # Pass it along
                progress=progress
            )
            print(f"DEBUG [Gradio UI Wrapper]: Final result for UI: {result!r}")
            return result

        submit_btn.click(
            fn=run_task_for_ui,
            inputs=[ # Ensure this list matches the parameters of run_task_for_ui IN ORDER
                agent_type_dropdown_ui, main_input_textbox, agent_model_name_dropdown,
                metagpt_investment, metagpt_n_round, metagpt_project_name, metagpt_code_review, metagpt_run_tests, metagpt_implement,
                metagpt_inc, metagpt_project_path, metagpt_reqa_file, metagpt_max_auto_summarize, metagpt_recover_path,
                researcher_report_type_dummy, # From the hidden group to match signature
            ],
            outputs=output_textbox,
        )
    return interface

# --- Main Application Runner ---
if __name__ == "__main__":
    print(f"Launching Gradio UI for Single Server Test ({Path(__file__).name})...")
    print(f"Script directory (SCRIPT_DIR): {SCRIPT_DIR}")
    # Ensure stdout/stderr are line-buffered for better Gradio/terminal logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print(f"Python executable: {sys.executable}")
    print(f"PYTHONPATH (at app start): {os.environ.get('PYTHONPATH', 'Not set')}")

    # Create and launch the Gradio interface
    gradio_interface = create_ui()
    gradio_interface.launch(server_name="127.0.0.1", server_port=7860, share=False)
    # The script will now block here until Gradio is closed or interrupted.