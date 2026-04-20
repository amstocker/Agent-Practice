"""
RAG search tool for the agent.

Provides a tool that searches the ChromaDB vector store for the most
relevant CHAPTERS. Since we chunk at the chapter level, each result
is a full chapter — giving the LLM complete context for its answer.
"""

import os
import json
import chromadb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "retrieve_chapters",
        "description": (
            "Retrieve the most relevant chapters from the Agentic Design Patterns book. "
            "Returns full chapter text with chapter number, title, and page range. "
            "Use the table of contents in your system prompt to guide your query. "
            "Each result is an entire chapter, so request only 1-2 at a time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — describe the topic you're looking for"
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of chapters to return (default 1, max 3)"
                }
            },
            "required": ["query"]
        }
    }
]


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

def run_retrieve_chapters(query: str, n_results: int = 1) -> str:
    """
    Query ChromaDB for the most relevant chapters.

    Returns full chapter text so the LLM has complete context.
    Limited to 3 chapters max to stay within token limits.
    """
    n_results = min(n_results, 3)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("agentic_patterns")
    except Exception:
        return "Error: No document collection found. Run ingest.py first."

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    chapters = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        chapters.append({
            "chapter_number": meta["chapter_number"],
            "title": meta["title"],
            "pages": f"{meta['start_page']}-{meta['end_page']}",
            "text": results["documents"][0][i],
        })

    return json.dumps({"query": query, "chapters": chapters}, indent=2)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_IMPLEMENTATIONS = {
    "retrieve_chapters": run_retrieve_chapters,
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Look up a tool by name, call it, return the result."""
    func = TOOL_IMPLEMENTATIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        return func(**tool_input)
    except Exception as e:
        return f"Error running tool '{name}': {e}"
