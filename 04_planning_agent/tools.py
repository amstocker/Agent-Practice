"""
RAG retrieval tool for the planning agent.

Reuses the ChromaDB vector store from project 03 (03_rag_agent/chroma_db/).
No separate ingestion needed — just point to the existing store.
"""

import os
import json
import chromadb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "..", "03_rag_agent", "chroma_db")


def retrieve_chapters(query: str, n_results: int = 1) -> str:
    """
    Query ChromaDB for the most relevant chapters from the
    Agentic Design Patterns book. Returns full chapter text.
    """
    n_results = min(n_results, 3)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("agentic_patterns")
    except Exception:
        return json.dumps({"error": "No document collection found. Run 03_rag_agent/ingest.py first."})

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

    return json.dumps({"query": query, "chapters": chapters})
