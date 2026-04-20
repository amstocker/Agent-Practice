"""
A simple tool-using agent built with the Anthropic API.

This demonstrates the core agent loop:
  1. Send a user message (+ tool definitions) to Claude
  2. Claude either responds with text (done!) or requests tool calls
  3. Execute the requested tools
  4. Send the tool results back to Claude
  5. Repeat from step 2

Run it:
  python agent.py "What is 42 * 17?"
  python agent.py  # interactive prompt
"""

import sys
import anthropic
from dotenv import load_dotenv
from tools import TOOL_SCHEMAS, execute_tool

# Load ANTHROPIC_API_KEY from .env file
load_dotenv()

# The Anthropic client automatically reads ANTHROPIC_API_KEY from the environment
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use your tools when they would help answer the user's question.
If you don't need a tool, just respond directly."""


def run_agent(user_message: str) -> str:
    """
    The agent loop. Sends a message to Claude, handles tool calls,
    and returns Claude's final text response.
    """
    messages = [{"role": "user", "content": user_message}]

    while True:
        # --- Step 1: Send the conversation to Claude ---
        # Claude sees the full message history, the system prompt, and the
        # available tools. It decides whether to use a tool or respond directly.
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # --- Step 2: Check why Claude stopped ---
        if response.stop_reason == "end_turn":
            # Claude is done — extract and return the text response
            text_parts = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_parts)

        if response.stop_reason == "tool_use":
            # Claude wants to use one or more tools.
            # First, add Claude's response (including tool_use blocks) to the history.
            # This is required — Claude needs to see its own tool requests.
            messages.append({"role": "assistant", "content": response.content})

            # --- Step 3: Execute each tool call ---
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [Tool call: {block.name}({block.input})]")
                    result = execute_tool(block.name, block.input)
                    print(f"  [Result: {result}]")

                    # Each result must reference the tool_use_id so Claude
                    # can match results to the specific calls it made.
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # --- Step 4: Send tool results back to Claude ---
            # Tool results go in a "user" message. Claude will process them
            # and either make more tool calls or give a final answer.
            messages.append({"role": "user", "content": tool_results})

            # Loop continues — back to Step 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Ask the agent: ")

    print(f"\nYou: {prompt}\n")
    answer = run_agent(prompt)
    print(f"\nAgent: {answer}")
