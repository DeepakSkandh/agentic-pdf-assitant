"""File Finder agent: finds PDFs on this laptop from a plain-language request.

Loop (bounded, like the Research Scout):
  1. UNDERSTAND : turn "my control systems notes from last month" into keywords, dates, a folder
  2. SEARCH     : query the local full-text index (names + PDF text)
  3. RETRY      : nothing found -> ask the model for synonyms / broader terms and search again
  4. RANK       : the model reads the candidates' names and matching text and picks the best few
Read-only: it never opens, moves or changes your files by itself.
"""
import re
from datetime import date

from config import FINDER_MAX_RESULTS, FINDER_MAX_ROUNDS
from llm import chat, parse_json
from tools.files import STATUS, build_index, indexed_count, search

MAX_CANDIDATES = 12

PARSE_PROMPT = """Today is {today}. The user is looking for a PDF on their own laptop.
Turn the request into search terms:
- "keywords": 2-6 words likely to appear in the file's NAME or TEXT: topics, course names or codes,
  authors, title words. Add useful synonyms or abbreviations (e.g. "control systems" -> "control",
  "laplace", "transfer function"). Don't include words like "pdf", "file", "notes", "my".
- "date_from" / "date_to": only if the request mentions a time ("last week", "in October"),
  as YYYY-MM-DD; otherwise null.
- "folder": only if the user names a folder (Downloads, Desktop, Documents, a course folder); otherwise null.

Respond ONLY with JSON: {{"keywords": ["..."], "date_from": null, "date_to": null, "folder": null}}

REQUEST: {request}"""

RETRY_PROMPT = """No PDFs matched these search terms: {tried}.
The user asked: "{request}"
Suggest 3-5 DIFFERENT keywords that might appear in the file's name or text: synonyms, broader
topics, abbreviations, related terms. Respond ONLY with JSON: {{"keywords": ["..."]}}"""

RANK_PROMPT = """The user is looking for a PDF on their laptop: "{request}"

Candidates (file name, folder, last modified, matching text):
{candidates}

Pick up to {k} files that match the request, best first. Skip files that clearly don't match.
Respond ONLY with JSON: {{"picks": [{{"id": <candidate number>, "why": "short reason"}}]}}"""

STOP = {"pdf", "pdfs", "file", "files", "my", "notes", "document", "documents", "the", "a", "an"}


def clean_keywords(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    out = []
    for k in raw:
        k = str(k).strip().strip('"\'')
        if k and k.lower() not in STOP and k.lower() not in (x.lower() for x in out):
            out.append(k)
    return out[:6]


def valid_date(value) -> str | None:
    try:
        return date.fromisoformat(str(value)).isoformat() if value else None
    except ValueError:
        return None


def describe(files: list[dict]) -> str:
    lines = []
    for i, f in enumerate(files, 1):
        snippet = re.sub(r"\[\[|\]\]", "", f["snippet"])[:220]
        lines.append(f"[{i}] {f['name']} | {f['folder']} | {f['modified']}\n    {snippet}")
    return "\n".join(lines)


def find_files(request: str) -> dict:
    log = []
    if indexed_count() == 0 and not STATUS["running"]:
        log.append("Finder: index was empty, scanning your folders first")
        build_index()

    parsed = parse_json(chat(PARSE_PROMPT.format(today=date.today().isoformat(), request=request),
                             json_mode=True, temperature=0)) or {}
    keywords = clean_keywords(parsed.get("keywords")) or clean_keywords(request.split())
    date_from, date_to = valid_date(parsed.get("date_from")), valid_date(parsed.get("date_to"))
    folder = str(parsed["folder"]).strip() if parsed.get("folder") else None
    filters = ", ".join(x for x in [f"from {date_from}" if date_from else "",
                                    f"to {date_to}" if date_to else "",
                                    f"in '{folder}'" if folder else ""] if x)

    tried, results = [], []
    for rnd in range(1, FINDER_MAX_ROUNDS + 1):
        results = search(keywords, date_from, date_to, folder, limit=MAX_CANDIDATES)
        tried += keywords
        log.append(f"Finder: round {rnd} searched {keywords}{' ' + filters if filters else ''}"
                   f" -> {len(results)} file(s)")
        if results or rnd == FINDER_MAX_ROUNDS:
            break
        retry = parse_json(chat(RETRY_PROMPT.format(tried=tried, request=request),
                                json_mode=True, temperature=0)) or {}
        new = [k for k in clean_keywords(retry.get("keywords")) if k.lower() not in (t.lower() for t in tried)]
        if not new:
            break
        keywords = new
        log.append("Finder: nothing matched, trying different words")

    note = ""
    if not results and filters:  # the date or folder ruled everything out: try without them
        results = search(tried, limit=MAX_CANDIDATES)
        if results:
            note = f"Nothing matched {filters}, so these are matches from any time or folder."
            log.append(f"Finder: nothing {filters}; {len(results)} match(es) without that filter")
    if not results:
        return {"files": [], "keywords": tried, "filters": filters, "note": "", "log": log}

    picks = []
    if len(results) > 1:
        data = parse_json(chat(RANK_PROMPT.format(request=request, k=FINDER_MAX_RESULTS,
                                                  candidates=describe(results)),
                               json_mode=True, temperature=0)) or {}
        for item in data.get("picks") or []:
            try:
                idx = int(item.get("id")) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= idx < len(results) and all(p["path"] != results[idx]["path"] for p in picks):
                picks.append({**results[idx], "why": str(item.get("why", "")).strip()})
            if len(picks) == FINDER_MAX_RESULTS:
                break
    if not picks:  # one result, or the model's answer was unusable: keep the search order
        picks = [{**f, "why": ""} for f in results[:FINDER_MAX_RESULTS]]
    log.append(f"Finder: showing {len(picks)} of {len(results)} matching file(s)")
    return {"files": picks, "keywords": tried, "filters": filters, "note": note, "log": log}
