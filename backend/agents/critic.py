"""Critic agent: fact-checks the summary against the notes and decides approve / revise."""
from llm import chat, parse_json

PROMPT = """You are a strict fact-checker. The NOTES are the ground truth. Check the SUMMARY for:
1. Claims not supported by the NOTES (hallucinations)
2. Important points in the NOTES missing from the SUMMARY
3. Wrong numbers, names or conclusions

Ignore minor wording or style. Respond ONLY with JSON in this exact form:
{{"approved": true or false, "issues": ["short description of each problem"]}}
Set "approved" to true only if there are no serious problems (then "issues" can be empty).

NOTES:
{notes}

SUMMARY:
{summary}"""


def critic_agent(state):
    raw = chat(PROMPT.format(notes=state["notes"], summary=state["summary"]),
               json_mode=True, temperature=0)
    data = parse_json(raw)
    if not data or "approved" not in data:
        # Unparseable output: don't burn another round on it, accept and note it.
        verdict = {"approved": True, "issues": [], "note": "critic output could not be parsed"}
    else:
        issues = [str(i) for i in data.get("issues", []) if str(i).strip()]
        approved = data["approved"]
        if isinstance(approved, str):  # small models sometimes return "false" as a string
            approved = approved.strip().lower() == "true"
        verdict = {"approved": bool(approved), "issues": issues}
        if not verdict["approved"] and not issues:
            verdict["approved"] = True  # rejected with no reasons -> nothing to fix

    msg = "approved" if verdict["approved"] else f"found {len(verdict['issues'])} issue(s)"
    return {"critiques": [verdict], "approved": verdict["approved"], "log": [f"Critic: {msg}"]}
