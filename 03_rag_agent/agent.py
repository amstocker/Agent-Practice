"""
A RAG (Retrieval-Augmented Generation) agent that answers questions
about the Agentic Design Patterns book.

This builds on projects 01 and 02 by adding:
- Semantic search over a document corpus (vector similarity)
- Context injection — retrieved chapters are fed to the LLM
- Grounded answers — the agent cites page numbers from the book
- Chapter-level retrieval — the agent sees the TOC and retrieves full chapters

Run it:
  python agent.py "What is the reflection pattern?"
  python agent.py  # interactive prompt
"""

import sys
import json
import anthropic
from dotenv import load_dotenv
from tools import TOOL_SCHEMAS, execute_tool

load_dotenv()
load_dotenv("../.env")

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about the book \
"Agentic Design Patterns" by Antonio Gulli.

Here is the book's table of contents:

Part One — Core Patterns:
  Chapter 1: Prompt Chaining (pages 23-35)
  Chapter 2: Routing (pages 36-49)
  Chapter 3: Parallelization (pages 50-64)
  Chapter 4: Reflection (pages 65-78)
  Chapter 5: Tool Use / Function Calling (pages 79-99)
  Chapter 6: Planning (pages 100-112)
  Chapter 7: Multi-Agent Collaboration (pages 113-131)

Part Two — Advanced Capabilities:
  Chapter 8: Memory Management (pages 132-153)
  Chapter 9: Learning and Adaptation (pages 154-166)
  Chapter 10: Model Context Protocol (pages 167-182)
  Chapter 11: Goal Setting and Monitoring (pages 183-195)

Part Three — Safety and Human Oversight:
  Chapter 12: Exception Handling and Recovery (pages 196-203)
  Chapter 13: Human-in-the-Loop (pages 204-212)
  Chapter 14: Knowledge Retrieval / RAG (pages 213-230)

Part Four — System Design:
  Chapter 15: Inter-Agent Communication (pages 231-245)
  Chapter 16: Resource-Aware Optimization (pages 246-261)
  Chapter 17: Reasoning Techniques (pages 262-285)
  Chapter 18: Guardrails/Safety Patterns (pages 286-305)
  Chapter 19: Evaluation and Monitoring (pages 306-324)
  Chapter 20: Prioritization (pages 325-334)
  Chapter 21: Exploration and Discovery (pages 335-end)

You have a tool to retrieve full chapter content from the book.

When answering questions:
1. Use the table of contents above to identify which chapter(s) are most relevant.
2. Retrieve the relevant chapter(s) using the retrieve_chapters tool.
3. Provide a thorough, well-structured answer based on the chapter content.
4. Cite page numbers when referencing specific content.
5. If the retrieved content doesn't answer the question, say so honestly.
6. Retrieve only 1-2 chapters at a time to keep responses focused."""


def run_agent(user_message: str) -> str:
    """The agent loop — same pattern as projects 01 and 02."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
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
                    print(f"  [Retrieving: {block.input.get('query', '')}]")
                    result = execute_tool(block.name, block.input)
                    # Show which chapters were retrieved
                    parsed = json.loads(result)
                    for ch in parsed.get("chapters", []):
                        print(f"  [Got Chapter {ch['chapter_number']}: "
                              f"{ch['title']} (pages {ch['pages']})]")

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
        prompt = input("Ask about Agentic Design Patterns: ")

    print(f"\nYou: {prompt}\n")
    answer = run_agent(prompt)
    print(f"\nAgent: {answer}")
