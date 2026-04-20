"""
Document ingestion pipeline for the RAG agent.

This script processes the Agentic Design Patterns PDF and stores its content
in a vector database, chunked by CHAPTER rather than by fixed character count.

Run it once before using the agent:

    python ingest.py

The pipeline:
1. EXTRACT  — Pull text from each page of the PDF (using PyMuPDF)
2. DETECT   — Find chapter boundaries by looking for "Chapter N:" headings
3. CHUNK    — Group pages into chapters (one chunk per chapter)
4. EMBED    — Convert each chapter into a vector (ChromaDB does this automatically)
5. STORE    — Save the vectors + text + metadata in a persistent ChromaDB collection

Why chapter-level chunks?
- Each chapter covers one coherent topic (e.g., "Reflection", "Tool Use")
- The agent gets full context when it retrieves a chapter — no fragmented sentences
- The table of contents in the system prompt lets the agent know which chapters to search for
- Trade-off: larger chunks use more tokens per retrieval, but produce much better answers
"""

import os
import re
import fitz  # PyMuPDF
import chromadb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "..", "resources", "Agentic-Design-Patterns.pdf")
CHROMA_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

# Chapters and their starting PDF pages (1-indexed), determined by scanning
# for "Chapter N:" headings in the PDF. These skip the TOC (page 1) and
# index references at the end.
CHAPTERS = [
    {"number": 1,  "title": "Prompt Chaining",              "start_page": 23},
    {"number": 2,  "title": "Routing",                      "start_page": 36},
    {"number": 3,  "title": "Parallelization",              "start_page": 50},
    {"number": 4,  "title": "Reflection",                   "start_page": 65},
    {"number": 5,  "title": "Tool Use (Function Calling)",  "start_page": 79},
    {"number": 6,  "title": "Planning",                     "start_page": 100},
    {"number": 7,  "title": "Multi-Agent Collaboration",    "start_page": 113},
    {"number": 8,  "title": "Memory Management",            "start_page": 132},
    {"number": 9,  "title": "Learning and Adaptation",      "start_page": 154},
    {"number": 10, "title": "Model Context Protocol",       "start_page": 167},
    {"number": 11, "title": "Goal Setting and Monitoring",  "start_page": 183},
    {"number": 12, "title": "Exception Handling and Recovery", "start_page": 196},
    {"number": 13, "title": "Human-in-the-Loop",            "start_page": 204},
    {"number": 14, "title": "Knowledge Retrieval (RAG)",    "start_page": 213},
    {"number": 15, "title": "Inter-Agent Communication",    "start_page": 231},
    {"number": 16, "title": "Resource-Aware Optimization",  "start_page": 246},
    {"number": 17, "title": "Reasoning Techniques",         "start_page": 262},
    {"number": 18, "title": "Guardrails/Safety Patterns",   "start_page": 286},
    {"number": 19, "title": "Evaluation and Monitoring",    "start_page": 306},
    {"number": 20, "title": "Prioritization",               "start_page": 325},
    {"number": 21, "title": "Exploration and Discovery",    "start_page": 335},
]


def extract_chapters(pdf_path: str) -> list[dict]:
    """
    Extract text for each chapter by reading pages between chapter boundaries.
    Returns a list of {chapter_number, title, start_page, end_page, text} dicts.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    chapters = []

    for i, ch in enumerate(CHAPTERS):
        start = ch["start_page"] - 1  # fitz uses 0-indexed pages
        # End page is one before the next chapter starts (or end of doc)
        if i + 1 < len(CHAPTERS):
            end = CHAPTERS[i + 1]["start_page"] - 1
        else:
            end = total_pages

        # Concatenate all page text for this chapter
        chapter_text = ""
        for page_num in range(start, end):
            chapter_text += doc[page_num].get_text() + "\n"

        chapters.append({
            "chapter_number": ch["number"],
            "title": ch["title"],
            "start_page": ch["start_page"],
            "end_page": ch["start_page"] + (end - start) - 1,
            "text": chapter_text.strip(),
        })

        print(f"  Chapter {ch['number']:2d}: {ch['title']:<40s} "
              f"(pages {ch['start_page']}-{chapters[-1]['end_page']}, "
              f"{len(chapter_text):,} chars)")

    doc.close()
    return chapters


def store_in_chromadb(chapters: list[dict]):
    """
    Store each chapter as a single document in ChromaDB.

    Each chapter gets embedded as one vector. When the agent searches,
    ChromaDB returns the most relevant chapter(s) based on semantic similarity
    to the query.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if re-ingesting
    try:
        client.delete_collection("agentic_patterns")
    except Exception:
        pass

    collection = client.create_collection(
        name="agentic_patterns",
        metadata={"description": "Agentic Design Patterns book — chapter-level chunks"}
    )

    ids = [f"chapter_{ch['chapter_number']}" for ch in chapters]
    documents = [ch["text"] for ch in chapters]
    metadatas = [{
        "chapter_number": ch["chapter_number"],
        "title": ch["title"],
        "start_page": ch["start_page"],
        "end_page": ch["end_page"],
    } for ch in chapters]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"\n  Stored {collection.count()} chapters in collection '{collection.name}'")


if __name__ == "__main__":
    print(f"PDF: {PDF_PATH}\n")

    print("Step 1: Extracting chapters from PDF...")
    chapters = extract_chapters(PDF_PATH)

    print(f"\nStep 2: Embedding and storing in ChromaDB...")
    store_in_chromadb(chapters)

    print(f"\nIngestion complete! Vector store saved to {CHROMA_DIR}/")
