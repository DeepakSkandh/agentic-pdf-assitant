"""Research Scout agent: finds 2-3 relevant papers and says where to find them. No downloads.

It works on its own (a bounded Reason -> Act -> Observe loop):
  1. DECIDE  : is a search useful for this document at all? If yes, write its own queries.
  2. ACT     : run the queries on arXiv and OpenAlex (relevance + most-cited).
  3. OBSERVE : read the titles/abstracts, pick the best ones, and judge if they're good enough.
  4. REPEAT  : if not good enough, write better queries and search again (max SCOUT_MAX_ROUNDS).

It can also be called directly with a user request ("find the original UKF paper").
"""
import re

from config import SCOUT_AUTO, SCOUT_MAX_PAPERS, SCOUT_MAX_ROUNDS
from llm import chat, parse_json
from tools.search import SOURCES, merge, search_all

MAX_CANDIDATES = 24  # how many candidates the model judges at once (fits the 8K context)

DECIDE_PROMPT = """You are a research assistant who just read a document. Decide on your own whether
finding {k} related papers would genuinely help its reader.

Search when the document is academic or technical and the reader would benefit from:
the original work it builds on, explanations of concepts it assumes, or work addressing
the weaknesses a reviewer noted. Do NOT search for non-technical documents
(receipts, forms, letters, invoices, personal documents).

If you search, write 2-3 short keyword queries (3-8 words each) for an academic search engine.
No quotes, no boolean operators, no word "pdf".

Respond ONLY with JSON: {{"search": true or false, "reason": "one sentence", "queries": ["...", "..."]}}

DOCUMENT SUMMARY:
{summary}

REVIEWER'S WEAKNESSES:
{weaknesses}"""

REQUEST_PROMPT = """Turn the user's request into 2-3 short keyword queries (3-8 words) for an academic search engine.
If the user names a specific paper, the FIRST query must be that paper's exact title.
If the user asks for an original, first, seminal or well-known paper and you know its title or
authors, put them in the FIRST query (e.g. for the original Transformer paper:
"Attention is all you need Vaswani"). The original paper's title often doesn't contain the
method's later name, so use what you know.
No quotes, no boolean operators, no word "pdf".

Respond ONLY with JSON: {{"queries": ["...", "..."]}}

USER REQUEST: {request}
CONTEXT (document the user is reading, may be empty): {context}"""

RANK_PROMPT = """GOAL: {goal}

Below are candidate papers from academic search engines. Pick the {k} that best serve the GOAL,
best first. Prefer papers that are clearly on-topic; use citations and year as tie-breakers.
If the goal names a specific paper, pick the exact match first.
If the goal asks for the original, first or seminal paper, prefer the earliest, most-cited work
that introduced the idea over later extensions or applications.

Then judge: are your picks good enough? If fewer than {k} candidates are truly relevant,
set "good_enough" to false and suggest 1-2 better "new_queries".

Respond ONLY with JSON:
{{"picks": [{{"id": <candidate number>, "why": "one sentence on why it helps"}}],
  "good_enough": true or false, "new_queries": ["..."]}}

CANDIDATES:
{candidates}"""


def format_candidates(cands: list[dict]) -> str:
    lines = []
    for i, p in enumerate(cands, 1):
        meta = ", ".join(x for x in [p["year"], p["venue"],
                                     f"{p['citations']} citations" if p.get("citations") is not None else ""] if x)
        lines.append(f"[{i}] {p['title']} ({meta})\n    {p['abstract'][:300]}")
    return "\n".join(lines)


# Words that describe what the user wants but hurt keyword search ("original paper" etc.)
JUNK_WORDS = re.compile(r"\b(original|earliest|seminal|foundational|first|famous|well[- ]known|"
                        r"paper|papers|article|articles|introducing|introduced|introduces|pdf)\b", re.I)


def clean_queries(qs) -> list[str]:
    if not isinstance(qs, list):
        return []
    out = []
    for q in qs:
        raw = re.sub(r'["\']', "", str(q))
        q = re.sub(r"\s+", " ", JUNK_WORDS.sub(" ", raw)).strip(" ,.-")
        q = q or raw.strip()  # never throw a query away entirely
        if q and q.lower() not in (x.lower() for x in out):
            out.append(q)
    return out[:3]


def build_pool(candidates: list[dict]) -> list[dict]:
    """At most MAX_CANDIDATES for the model to judge. Always keep the most-cited ones,
    so classic papers aren't cut just because they appeared late in the results."""
    if len(candidates) <= MAX_CANDIDATES:
        return candidates
    top_cited = sorted(candidates, key=lambda p: p.get("citations") or 0, reverse=True)[:8]
    keep = {id(p) for p in top_cited}
    for p in candidates:
        if len(keep) >= MAX_CANDIDATES:
            break
        keep.add(id(p))
    return [p for p in candidates if id(p) in keep]  # original order


def scout_loop(goal: str, queries: list[str]) -> tuple[list[dict], list[str]]:
    """The bounded search -> judge -> re-query loop. Returns (papers, log lines)."""
    candidates, picks, log, tried = [], [], [], set()
    for rnd in range(1, SCOUT_MAX_ROUNDS + 1):
        new_q = [q for q in queries if q.lower() not in tried]
        if not new_q:
            break
        tried.update(q.lower() for q in new_q)
        found, errors = search_all(new_q)
        before = len(candidates)
        candidates = merge(candidates, found)
        log.append(f"Scout: round {rnd} searched {new_q} -> {len(candidates) - before} new candidate(s)")
        log.extend(f"Scout: {e}" for e in errors)
        if not candidates:
            if errors and len(errors) >= len(SOURCES) * len(new_q):
                log.append("Scout: all sources failed (offline?)")
                break
            continue

        pool = build_pool(candidates)
        data = parse_json(chat(RANK_PROMPT.format(goal=goal, k=SCOUT_MAX_PAPERS,
                                                  candidates=format_candidates(pool)),
                               json_mode=True, temperature=0)) or {}
        picks = []
        for item in data.get("picks") or []:
            try:
                idx = int(item.get("id")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= idx < len(pool) and all(p["title"] != pool[idx]["title"] for p in picks):
                picks.append({**pool[idx], "why": str(item.get("why", "")).strip()})
            if len(picks) == SCOUT_MAX_PAPERS:
                break
        if len(picks) < SCOUT_MAX_PAPERS:  # top up with the most-cited remaining candidates
            chosen = {p["title"] for p in picks}
            for p in sorted(pool, key=lambda p: p.get("citations") or 0, reverse=True):
                if len(picks) == SCOUT_MAX_PAPERS:
                    break
                if p["title"] not in chosen:
                    picks.append({**p, "why": "Closely related, highly cited work"})
        log.append(f"Scout: picked {len(picks)} paper(s) from {len(pool)} candidates")

        new_queries = clean_queries(data.get("new_queries"))
        if data.get("good_enough", True) is not False or not new_queries:
            break
        queries = new_queries
        log.append("Scout: not satisfied with results, searching again with new queries")

    if not picks and candidates:  # model output unusable: fall back to the top candidates
        picks = [{**p, "why": ""} for p in candidates[:SCOUT_MAX_PAPERS]]
    for p in picks:
        p.pop("abstract", None)  # keep the response small
    return picks, log


def weaknesses_from(review: str) -> str:
    m = re.search(r"#+\s*Weaknesses\s*(.*?)(?:\n#+\s|\Z)", review or "", re.S | re.I)
    return (m.group(1) if m else review or "")[:1500].strip()


def scout_agent(state):
    """Graph node: runs automatically after the Peer Reviewer and decides by itself."""
    if not SCOUT_AUTO:
        return {"papers": [], "scout_note": "Automatic paper search is turned off",
                "log": ["Scout: auto search disabled in config"]}
    try:
        decision = parse_json(chat(DECIDE_PROMPT.format(
            k=SCOUT_MAX_PAPERS, summary=state["summary"][:3000],
            weaknesses=weaknesses_from(state.get("review"))), json_mode=True, temperature=0)) or {}
        queries = clean_queries(decision.get("queries"))
        reason = str(decision.get("reason", "")).strip()
        if not decision.get("search") or not queries:
            return {"papers": [], "scout_note": reason or "No search needed for this document",
                    "log": [f"Scout: decided not to search ({reason or 'not useful here'})"]}

        goal = (f"Find papers that help the reader of this document. Why: {reason}\n"
                f"Document summary: {state['summary'][:1200]}")
        papers, log = scout_loop(goal, queries)
        return {"papers": papers, "scout_note": reason,
                "log": [f"Scout: decided to search ({reason})"] + log}
    except Exception as e:  # network or model trouble must not fail the whole pipeline
        return {"papers": [], "scout_note": f"Search failed: {type(e).__name__}",
                "log": [f"Scout: failed ({type(e).__name__}: {e})"]}


def search_papers(request: str, context: str = "") -> dict:
    """Direct user request, e.g. 'find the original PSO paper by Kennedy and Eberhart'."""
    data = parse_json(chat(REQUEST_PROMPT.format(request=request, context=context[:1500]),
                           json_mode=True, temperature=0)) or {}
    queries = clean_queries(data.get("queries")) or [request]
    goal = f"User request: {request}" + (f"\nThey are reading a document about: {context[:800]}" if context else "")
    papers, log = scout_loop(goal, queries)
    return {"papers": papers, "queries": queries, "log": log}
