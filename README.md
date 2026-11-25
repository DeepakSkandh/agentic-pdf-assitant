# Agentic PDF Assistant

A multi-agent system that reads a PDF, summarizes it, fact-checks the summary, reviews the
document like a peer reviewer, and reads the summary aloud. Runs fully locally on
Gemma 4 E4B via Ollama (designed for an 8 GB laptop GPU).

## Agents (v1)

| Agent | Job |
|-------|-----|
| Reader | PDF -> text chunks (PyMuPDF; scanned pages are OCR'd by Gemma's vision) |
| Summarizer | Map-reduce summary: per-chunk notes -> condensed notes -> summary |
| Critic | Fact-checks summary vs notes; sends it back for revision (max 2 rounds) |
| Peer Reviewer | Strengths, weaknesses, questions, score for the document itself |
| Narrator | Rewrites summary for speech -> MP3 (gTTS) |

```
START -> Reader -> Summarizer -> Critic --approved--> Peer Reviewer -> Narrator -> END
                       ^           |
                       +--revise---+
```

Coming next: React frontend, Q&A agent (RAG with citations), Planner agent.

## Setup (Windows, native, no WSL)

1. Install Ollama from https://ollama.com/download, then in PowerShell:
   ```powershell
   ollama --version          # must be 0.20+
   ollama pull gemma4:e4b
   ollama run gemma4:e4b "Say hi"
   ollama ps                 # in a 2nd terminal while it runs: should show 100% GPU
   ```
2. Install Python 3.11 or 3.12 from python.org (tick "Add Python to PATH").
3. Backend:
   ```powershell
   cd pdf-agent\backend
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Run

Test from the terminal first:
```powershell
python run_cli.py C:\path\to\some.pdf
```

Start the API server (used by the frontend):
```powershell
uvicorn main:app --reload --port 8000
```
Then open http://localhost:8000/docs to try the upload endpoint in the browser.

## Web app (React frontend)

Needs two terminals. Terminal 1, the backend:
```powershell
cd pdf-agent\backend
conda activate pdfagent
uvicorn main:app --port 8000
```
Terminal 2, the frontend (first time: `npm install`):
```powershell
cd pdf-agent\frontend
npm install
npm run dev
```
Open http://localhost:5173 and drop in a PDF.

## Tuning (backend/config.py)

- Out of memory or slow: set `MODEL = "gemma4:e2b"`, or lower `NUM_CTX` to 4096.
- Audio uses gTTS, which needs internet. Everything else is offline.
- Close heavy apps first: if VRAM fills up, Ollama spills to system RAM and slows down a lot.
