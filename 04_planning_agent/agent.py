"""
A planning agent that decomposes complex questions into research steps.

Unlike the reactive agents in projects 01-03 (which let the LLM decide
tool calls on the fly), this agent operates in three explicit phases:

  1. PLAN      — Generate a structured research plan (which chapters to retrieve)
  2. EXECUTE   — Work through the plan step by step, gathering information
  3. SYNTHESIZE — Produce a final answer from all gathered context

This separation of planning from execution is a key agentic design pattern.
The agent thinks about *what to do* before *doing it*.

Run it:
  python agent.py "Compare the reflection and planning patterns"
  python agent.py  # interactive prompt
"""

import sys
import json
import anthropic
from dotenv import load_dotenv
from tools import retrieve_chapters

load_dotenv()
load_dotenv("../.env")

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

BOOK_TOC = """
Part One — Core Patterns:
  Ch 1: Prompt Chaining | Ch 2: Routing | Ch 3: Parallelization
  Ch 4: Reflection | Ch 5: Tool Use | Ch 6: Planning | Ch 7: Multi-Agent Collaboration

Part Two — Advanced Capabilities:
  Ch 8: Memory Management | Ch 9: Learning and Adaptation
  Ch 10: Model Context Protocol | Ch 11: Goal Setting and Monitoring

Part Three — Safety and Human Oversight:
  Ch 12: Exception Handling | Ch 13: Human-in-the-Loop | Ch 14: Knowledge Retrieval (RAG)

Part Four — System Design:
  Ch 15: Inter-Agent Communication | Ch 16: Resource-Aware Optimization
  Ch 17: Reasoning Techniques | Ch 18: Guardrails/Safety Patterns
  Ch 19: Evaluation and Monitoring | Ch 20: Prioritization | Ch 21: Exploration and Discovery
""".strip()


# ---------------------------------------------------------------------------
# Phase 1: Generate a plan
# ---------------------------------------------------------------------------

PLANNING_PROMPT = f"""You are a research planning assistant. Given a user's question about \
the book "Agentic Design Patterns", create a research plan.

Here is the book's table of contents:
{BOOK_TOC}

Respond with a JSON object containing:
- "goal": a one-sentence restatement of what we need to find out
- "steps": a list of steps, where each step has:
  - "step": step number (integer)
  - "action": either "retrieve" (to look up a chapter) or "synthesize" (final step)
  - "query": (for retrieve steps) the search query to find the right chapter
  - "purpose": why this step is needed

Rules:
- Use the table of contents to identify which chapters are relevant
- Only retrieve chapters that are actually needed (typically 1-3)
- The last step should always be "synthesize"
- Keep the plan focused — don't retrieve chapters that aren't relevant

Respond with ONLY the JSON object, no other text."""


def generate_plan(question: str) -> dict:
    """Ask the LLM to create a research plan for the question."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Question: {question}"
        }],
        system=PLANNING_PROMPT,
    )

    plan_text = response.content[0].text

    # Parse the JSON plan — handle markdown code blocks if present
    plan_text = plan_text.strip()
    if plan_text.startswith("```"):
        plan_text = plan_text.split("\n", 1)[1]  # remove ```json line
        plan_text = plan_text.rsplit("```", 1)[0]  # remove closing ```

    return json.loads(plan_text)


# ---------------------------------------------------------------------------
# Phase 2: Execute the plan
# ---------------------------------------------------------------------------

def execute_plan(plan: dict) -> list[dict]:
    """
    Execute each step of the plan, gathering context along the way.
    Returns a list of results from each retrieve step.
    """
    gathered = []

    for step in plan["steps"]:
        if step["action"] == "synthesize":
            print(f"  Step {step['step']}: Synthesize — {step['purpose']}")
            break  # move to phase 3

        if step["action"] == "retrieve":
            print(f"  Step {step['step']}: Retrieve — {step['purpose']}")
            print(f"    Query: \"{step['query']}\"")

            result = retrieve_chapters(step["query"], n_results=1)
            parsed = json.loads(result)

            for ch in parsed.get("chapters", []):
                print(f"    → Got Chapter {ch['chapter_number']}: {ch['title']} "
                      f"(pages {ch['pages']})")
                gathered.append({
                    "step": step["step"],
                    "purpose": step["purpose"],
                    "chapter_number": ch["chapter_number"],
                    "title": ch["title"],
                    "pages": ch["pages"],
                    "text": ch["text"],
                })

    return gathered


# ---------------------------------------------------------------------------
# Phase 3: Synthesize a final answer
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are a knowledgeable assistant answering questions about \
the book "Agentic Design Patterns" by Antonio Gulli.

You have been given relevant chapter content gathered through a research plan.
Use this content to provide a thorough, well-structured answer.

When answering:
- Cite page numbers when referencing specific content
- Organize your answer with clear headings and structure
- Compare and contrast when the question asks for it
- If the gathered content doesn't fully answer the question, say so"""


def synthesize(question: str, gathered: list[dict]) -> str:
    """Final LLM call: combine all gathered context into a comprehensive answer."""

    # Build context from gathered chapter content
    context_parts = []
    for item in gathered:
        context_parts.append(
            f"--- Chapter {item['chapter_number']}: {item['title']} "
            f"(pages {item['pages']}) ---\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYNTHESIS_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Gathered research:\n\n{context}"
            ),
        }],
    )

    return response.content[0].text


# ---------------------------------------------------------------------------
# Main: Plan → Execute → Synthesize
# ---------------------------------------------------------------------------

def run_agent(question: str) -> str:
    """Run the full planning agent pipeline."""

    # Phase 1: Plan
    print("Phase 1: Generating research plan...\n")
    plan = generate_plan(question)
    print(f"  Goal: {plan['goal']}")
    print(f"  Steps: {len(plan['steps'])}")
    for step in plan["steps"]:
        action = step["action"]
        purpose = step["purpose"]
        query = step.get("query", "")
        if query:
            print(f"    {step['step']}. [{action}] \"{query}\" — {purpose}")
        else:
            print(f"    {step['step']}. [{action}] — {purpose}")
    print()

    # Phase 2: Execute
    print("Phase 2: Executing plan...\n")
    gathered = execute_plan(plan)
    print(f"\n  Gathered content from {len(gathered)} chapter(s)\n")

    # Phase 3: Synthesize
    print("Phase 3: Synthesizing answer...\n")
    answer = synthesize(question, gathered)
    return answer


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("Ask a complex question: ")

    print(f"\nQuestion: {prompt}\n")
    print("=" * 60)
    answer = run_agent(prompt)
    print("=" * 60)
    print(f"\n{answer}")
