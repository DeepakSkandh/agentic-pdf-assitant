"""Narrator agent: rewrites the approved summary for listening, then converts it to speech.

Default engine: Kokoro-82M (open source, runs offline on the CPU via ONNX Runtime, so it
doesn't compete with Gemma for VRAM). If Kokoro isn't set up or fails, falls back to gTTS."""
import re
from pathlib import Path

from config import (KOKORO_LANG, KOKORO_MODEL, KOKORO_SPEED, KOKORO_VOICE, KOKORO_VOICES,
                    TTS_ENGINE, TTS_LANG, TTS_TLD)
from llm import chat

PROMPT = """Rewrite this summary as a short script to be read aloud.
Use natural spoken sentences. No bullet points, markdown, headings or symbols.
Write numbers, units and equations in words. Keep it under 300 words. Output only the script.

SUMMARY:
{summary}"""

_kokoro = None  # loaded once, then reused for every job
PAUSE_S = 0.25  # silence between sentences


def get_kokoro():
    global _kokoro
    if _kokoro is None:
        if not KOKORO_MODEL.exists() or not KOKORO_VOICES.exists():
            raise FileNotFoundError(f"Kokoro model files not found in {KOKORO_MODEL.parent}")
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    return _kokoro


def split_sentences(text: str, max_chars: int = 350) -> list[str]:
    """Short pieces keep Kokoro fast and natural; long sentences are split at commas."""
    pieces = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sent) > max_chars:
            cut = sent.rfind(",", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            pieces.append(sent[:cut].strip())
            sent = sent[cut:]
        if sent.strip():
            pieces.append(sent.strip())
    return pieces


def tts_kokoro(text: str, path: Path) -> Path:
    import numpy as np
    import soundfile as sf

    kokoro = get_kokoro()
    chunks, segments, rate, t = [], [], 24000, 0.0
    for piece in split_sentences(text):
        samples, rate = kokoro.create(piece, voice=KOKORO_VOICE, speed=KOKORO_SPEED, lang=KOKORO_LANG)
        dur = len(samples) / rate
        # exact start/end of each sentence in the final audio -> synced subtitles in the UI
        segments.append({"text": piece, "start": round(t, 3), "end": round(t + dur, 3)})
        chunks.append(samples)
        chunks.append(np.zeros(int(rate * PAUSE_S), dtype=samples.dtype))  # short pause between sentences
        t += dur + PAUSE_S
    sf.write(str(path), np.concatenate(chunks), rate)
    return path, segments


def tts_gtts(text: str, path: Path) -> Path:
    from gtts import gTTS
    gTTS(text, lang=TTS_LANG, tld=TTS_TLD).save(str(path))
    # gTTS gives no timings; the UI estimates them from each sentence's length
    return path, [{"text": p, "start": None, "end": None} for p in split_sentences(text)]


def narrator_agent(state):
    spoken = chat(PROMPT.format(summary=state["summary"]))
    spoken = re.sub(r"[*#_`>|]", "", spoken).strip()  # strip any leftover markdown
    job_dir = Path(state["job_dir"])

    errors = []
    engines = [("Kokoro", tts_kokoro, "summary.wav"), ("gTTS", tts_gtts, "summary.mp3")]
    if TTS_ENGINE == "gtts":
        engines.reverse()
    for name, fn, filename in engines:
        try:
            path, segments = fn(spoken, job_dir / filename)
            note = f" (fallback, {errors[0]})" if errors else ""
            return {"spoken_text": spoken, "audio_path": str(path), "audio_segments": segments,
                    "log": [f"Narrator: audio ready via {name}{note}"]}
        except Exception as e:  # try the next engine; never fail the whole job over audio
            errors.append(f"{name} failed: {type(e).__name__}: {e}")
    return {"spoken_text": spoken, "audio_path": None, "audio_segments": [],
            "log": ["Narrator: no audio. " + " | ".join(errors)]}
