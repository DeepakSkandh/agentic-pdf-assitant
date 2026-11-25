"""Summarizer agent (map-reduce, so long PDFs fit in an 8K context).
1st run : summarize each chunk -> condense notes -> write summary.
Later runs: revise the summary using the Critic's feedback."""
from config import NOTES_BUDGET_CHARS
from llm import chat

SYSTEM = "You are a precise summarizer. Use only information present in the given text. Never invent facts."

CHUNK_PROMPT = """Summarize this part of a document in 5-8 bullet points.
Keep key facts, numbers, methods and conclusions. Add page numbers like (p. 3) where relevant.

TEXT:
{text}"""

MERGE_PROMPT = """Merge these notes into fewer bullet points. Keep every key fact, number and page reference.

NOTES:
{notes}"""

SUMMARY_PROMPT = """Below are notes from consecutive parts of one document. Write one coherent summary with:
1. Overview (2-3 sentences)
2. Key points (bullets)
3. Main conclusion

NOTES:
{notes}"""

REVISE_PROMPT = """You wrote the SUMMARY below from the NOTES. A fact-checker found these problems:
{issues}

Rewrite the summary fixing every problem. Use only the NOTES. Keep the same structure
(Overview, Key points, Main conclusion). Output only the new summary.

NOTES:
{notes}

SUMMARY:
{summary}"""


def condense(notes: list[str]) -> str:
    """Hierarchically merge notes until they fit in the budget."""
    while len("\n\n".join(notes)) > NOTES_BUDGET_CHARS and len(notes) > 1:
        groups, cur = [], []
        for n in notes:
            if cur and len("\n\n".join(cur + [n])) > NOTES_BUDGET_CHARS:
                groups.append(cur)
                cur = []
            cur.append(n)
        groups.append(cur)
        if len(groups) == len(notes):  # every note is already huge; merge pairs instead
            groups = [notes[i:i + 2] for i in range(0, len(notes), 2)]
        notes = [chat(MERGE_PROMPT.format(notes="\n\n".join(g)), system=SYSTEM) for g in groups]
    return "\n\n".join(notes)


def summarizer_agent(state):
    update = {}
    notes = state.get("notes")
    if notes is None:
        chunk_summaries = [chat(CHUNK_PROMPT.format(text=c), system=SYSTEM) for c in state["chunks"]]
        notes = condense(chunk_summaries)
        update.update(chunk_summaries=chunk_summaries, notes=notes)

    critiques = state.get("critiques") or []
    if state.get("summary") and critiques and not critiques[-1]["approved"]:
        issues = "\n".join(f"- {i}" for i in critiques[-1]["issues"])
        summary = chat(REVISE_PROMPT.format(issues=issues, notes=notes, summary=state["summary"]),
                       system=SYSTEM)
        rounds = state.get("revisions", 0) + 1
        update.update(summary=summary, revisions=rounds,
                      log=[f"Summarizer: revised summary (round {rounds})"])
    else:
        summary = chat(SUMMARY_PROMPT.format(notes=notes), system=SYSTEM)
        update.update(summary=summary, revisions=0,
                      log=[f"Summarizer: summarized {len(state['chunks'])} chunks"])
    return update
