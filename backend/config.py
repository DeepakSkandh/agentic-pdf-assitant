"""All tunable settings in one place."""
from pathlib import Path

# --- LLM (Ollama) ---
MODEL = "gemma4:e4b"        # fallback if slow / out of memory: "gemma4:e2b"
NUM_CTX = 8192              # context window; bigger = more VRAM (KV cache)
TEMPERATURE = 0.3

# --- Reader ---
MIN_TEXT_CHARS = 50         # a page with less text than this is treated as scanned -> OCR via Gemma vision
OCR_DPI = 150               # resolution used when rendering scanned pages to images
CHUNK_CHARS = 6000          # ~1500 tokens per chunk, fits easily in NUM_CTX

# --- Summarizer / Critic ---
NOTES_BUDGET_CHARS = 14000  # if all chunk notes exceed this, they're merged hierarchically
MAX_REVISIONS = 2           # max Summarizer <-> Critic rounds

# --- Narrator (TTS) ---
TTS_ENGINE = "kokoro"       # "kokoro" = offline, local (CPU) | "gtts" = Google, needs internet
KOKORO_DIR = Path(r"E:\pdf_agent\models\kokoro")
KOKORO_MODEL = KOKORO_DIR / "kokoro-v1.0.onnx"   # or the smaller "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = KOKORO_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "af_heart"   # others: af_bella, af_nicole, am_michael, bf_emma, bm_george ...
KOKORO_SPEED = 1.0
KOKORO_LANG = "en-us"       # "en-gb" for British voices (bf_*, bm_*)
TTS_LANG = "en"             # gTTS fallback settings
TTS_TLD = "co.in"           # Indian English accent in gTTS

# --- Research Scout (paper search; needs internet, nothing is downloaded) ---
SCOUT_AUTO = True           # after summarizing, let the Scout decide by itself whether to search
SCOUT_MAX_PAPERS = 3        # papers returned
SCOUT_MAX_ROUNDS = 2        # search -> judge -> re-query rounds (hard limit)
SCOUT_PER_SOURCE = 6        # results fetched per query from each source
HTTP_TIMEOUT = 15           # seconds per API call
S2_API_KEY = ""             # optional Semantic Scholar key (higher rate limits)
OPENALEX_EMAIL = ""         # optional: your email puts you in OpenAlex's faster "polite pool"
OPENALEX_API_KEY = ""       # optional OpenAlex key, if they ask for one

# --- File Finder (searches PDFs on this laptop; read-only, never changes your files) ---
# ONLY these folders are ever scanned. Add your own, e.g. Path(r"E:\notes") or Path(r"E:\college").
# Folders that don't exist are skipped (the OneDrive ones are there in case Windows moved
# Documents/Desktop into OneDrive).
FINDER_ROOTS = [
    Path.home() / "Downloads",
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "OneDrive" / "Documents",
    Path.home() / "OneDrive" / "Desktop",
]
FINDER_SKIP_DIRS = {"node_modules", "AppData", "site-packages", "venv", ".venv", "__pycache__", "outputs"}
FINDER_MAX_PAGES = 25         # pages of text indexed per PDF (enough to know what it's about)
FINDER_MAX_CHARS = 30000      # characters indexed per PDF
FINDER_MAX_RESULTS = 5        # files shown to the user
FINDER_MAX_ROUNDS = 2         # search -> no match -> retry with synonyms
FINDER_INDEX_PATH = Path(__file__).parent / "finder_index.sqlite"

# --- Storage ---
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
