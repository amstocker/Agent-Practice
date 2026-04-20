"""
A SQL database agent that translates natural language questions into SQL queries.

This builds on the tool-use pattern from project 01, adding:
- Tools that interact with stateful resources (a database)
- The agent writes code (SQL) that our program executes
- Schema discovery — the agent explores the database structure before querying

Run it:
  python agent.py "What genres of music are in the database?"
  python agent.py  # interactive prompt
"""

import sys
import anthropic
from dotenv import load_dotenv
from tools import TOOL_SCHEMAS, execute_tool

# Load ANTHROPIC_API_KEY from .env file.
# Looks in this directory first, then falls back to parent directory.
load_dotenv()
load_dotenv("../.env")

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a helpful data analyst assistant with access to a SQLite database.

You have three tools:
- list_tables: Discover what tables exist in the database.
- describe_table: See the columns and types of a specific table.
- run_sql: Execute a read-only SQL SELECT query.

When answering questions about the data:
1. First explore the schema (list tables, describe relevant tables) so you know what's available.
2. Write SQL queries to answer the question.
3. Present the results in a clear, human-readable way.

If a query returns an error, read the error message and try to fix your SQL.
Only use SELECT queries — the database is read-only."""


def run_agent(user_message: str) -> str:
    """The agent loop — same pattern as project 01."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text_parts = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_parts)

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [Tool call: {block.name}({block.input})]")
                    result = execute_tool(block.name, block.input)
                    print(f"  [Result: {result}]")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Ask the agent: ")

    print(f"\nYou: {prompt}\n")
    answer = run_agent(prompt)
    print(f"\nAgent: {answer}")
