"""
Tool definitions for the agent.

Each tool has two parts:
1. A JSON schema that tells Claude what the tool does and what arguments it accepts.
2. A Python function that actually executes the tool.

The Anthropic API expects tool schemas in a specific format:
{
    "name": "tool_name",
    "description": "What the tool does",
    "input_schema": {
        "type": "object",
        "properties": { ... },
        "required": [ ... ]
    }
}
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Tool schemas (sent to Claude so it knows what tools are available)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /), exponentiation (**), and parentheses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 3 * 7'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature and conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Tokyo'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time, optionally in a specific timezone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name, e.g. 'US/Eastern', 'Asia/Tokyo'. Defaults to UTC."
                }
            },
            "required": []
        }
    }
]


# ---------------------------------------------------------------------------
# Tool implementations (plain Python functions)
# ---------------------------------------------------------------------------

def run_calculator(expression: str) -> str:
    """Evaluate a math expression safely using a restricted eval."""
    # Only allow math-related names — no builtins, no imports
    allowed_names = {"__builtins__": {}}
    try:
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def run_weather(city: str) -> str:
    """Return stubbed weather data. In a real agent, this would call a weather API."""
    # Stubbed data — replace with an actual API call later
    stubbed = {
        "Tokyo":     {"temp_f": 72, "condition": "Partly Cloudy"},
        "Paris":     {"temp_f": 65, "condition": "Rainy"},
        "New York":  {"temp_f": 58, "condition": "Sunny"},
        "London":    {"temp_f": 55, "condition": "Overcast"},
    }
    data = stubbed.get(city, {"temp_f": 70, "condition": "Unknown (stubbed data)"})
    return json.dumps({"city": city, **data})


def run_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in the given timezone."""
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Error with timezone '{timezone}': {e}"


# ---------------------------------------------------------------------------
# Dispatcher — the agent loop calls this single function
# ---------------------------------------------------------------------------

# Maps tool names to their implementation functions
TOOL_IMPLEMENTATIONS = {
    "calculator": run_calculator,
    "get_weather": run_weather,
    "get_current_time": run_current_time,
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Look up a tool by name, call it with the provided input, return the result as a string."""
    func = TOOL_IMPLEMENTATIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        return func(**tool_input)
    except Exception as e:
        return f"Error running tool '{name}': {e}"
