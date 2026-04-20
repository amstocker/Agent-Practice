"""
SQL tools for the database agent.

Three tools that let the agent explore and query a SQLite database:
1. list_tables    — discover what tables exist
2. describe_table — inspect a table's columns and types
3. run_sql        — execute a SELECT query and get results

This progression (discover → inspect → query) teaches the agent to
explore the schema before writing queries — a pattern that matters
in real-world database agents.
"""

import json
import re
from database import get_connection


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "list_tables",
        "description": "List all tables in the database. Call this first to discover what data is available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "describe_table",
        "description": "Show the columns and their types for a given table. Use this to understand a table's structure before querying it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "The name of the table to describe"
                }
            },
            "required": ["table_name"]
        }
    },
    {
        "name": "run_sql",
        "description": "Execute a read-only SQL SELECT query against the database and return the results. Only SELECT statements are allowed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A SQL SELECT query, e.g. 'SELECT * FROM artists LIMIT 5'"
                }
            },
            "required": ["query"]
        }
    }
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def run_list_tables() -> str:
    """Return a list of all table names in the database."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row["name"] for row in cursor.fetchall()]
    return json.dumps({"tables": tables})


def run_describe_table(table_name: str) -> str:
    """Return column names and types for a table using PRAGMA table_info."""
    conn = get_connection()
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [
            {"name": row["name"], "type": row["type"], "nullable": not row["notnull"]}
            for row in cursor.fetchall()
        ]
        if not columns:
            return f"Table '{table_name}' not found."
        return json.dumps({"table": table_name, "columns": columns})
    except Exception as e:
        return f"Error describing table '{table_name}': {e}"


def run_sql(query: str) -> str:
    """Execute a SELECT query and return formatted results."""
    # Safety check: only allow SELECT statements
    stripped = query.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        return "Error: Only SELECT queries are allowed. INSERT, UPDATE, DELETE, and DROP are blocked."

    # Extra guard: reject statements containing dangerous keywords after the first word
    dangerous = re.search(r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b', stripped)
    if dangerous:
        return f"Error: Query contains disallowed keyword '{dangerous.group()}'."

    conn = get_connection()
    try:
        cursor = conn.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return "Query returned 0 rows."

        # Format as a readable table
        columns = [desc[0] for desc in cursor.description]
        result_rows = [dict(row) for row in rows]

        # Build a simple text table
        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))
        for row in result_rows:
            lines.append(" | ".join(str(row[col]) for col in columns))

        return f"{len(result_rows)} row(s) returned:\n" + "\n".join(lines)

    except Exception as e:
        return f"SQL Error: {e}"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_IMPLEMENTATIONS = {
    "list_tables": run_list_tables,
    "describe_table": run_describe_table,
    "run_sql": run_sql,
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Look up a tool by name, call it with the provided input, return the result."""
    func = TOOL_IMPLEMENTATIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        return func(**tool_input)
    except Exception as e:
        return f"Error running tool '{name}': {e}"
