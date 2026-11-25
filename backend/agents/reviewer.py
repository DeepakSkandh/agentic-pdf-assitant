"""Peer Reviewer agent: critiques the DOCUMENT itself (not the summary), like a conference reviewer."""
from llm import chat

PROMPT = """You are an experienced, fair peer reviewer. Based on the NOTES of a document and its SUMMARY,
write a short review in markdown with these sections:

### Strengths
### Weaknesses
(weak assumptions, missing baselines or comparisons, claims the evidence doesn't support)
### Questions for the authors
### Score
(1-10 with one line of justification)

Be specific and refer to page numbers where possible. If the document is not a research paper,
review its clarity, completeness and reliability instead.

NOTES:
{notes}

SUMMARY:
{summary}"""


def reviewer_agent(state):
    review = chat(PROMPT.format(notes=state["notes"], summary=state["summary"]))
    return {"review": review, "log": ["Peer Reviewer: review written"]}
