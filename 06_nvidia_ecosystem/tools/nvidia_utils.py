"""Shared helpers for Module 06 (NVIDIA ecosystem + Trustworthy AI).

Pure helpers (PII, activation-log) are unit-tested in test_nvidia_utils.py.
NIM/rails wrappers are thin and require external libs at runtime (Colab).
"""
import re

# Order matters: most-specific/longest patterns first so a 16-digit NIK
# is not also captured as a generic account number.
_PII_PATTERNS = [
    ("NIK", re.compile(r"\b\d{16}\b")),                      # KTP: exactly 16 digits
    ("PHONE", re.compile(r"(?:\+62|62|0)8\d{8,12}\b")),      # Indonesian mobile
    ("ACCOUNT", re.compile(r"\b\d{10,15}\b")),               # bank account (10-15 digits)
]
_PLACEHOLDER = {"NIK": "[NIK]", "PHONE": "[PHONE]", "ACCOUNT": "[ACCOUNT]"}


def detect_pii_id(text: str) -> list[dict]:
    """Return [{type, value, start, end}] for Indonesian PII, non-overlapping,
    most-specific pattern winning a span."""
    claimed = []  # (start, end)
    found = []
    for ptype, pat in _PII_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            if any(not (e <= cs or s >= ce) for cs, ce in claimed):
                continue  # overlaps an already-claimed (more specific) span
            claimed.append((s, e))
            found.append({"type": ptype, "value": m.group(), "start": s, "end": e})
    found.sort(key=lambda f: f["start"])
    return found


def mask_pii_id(text: str) -> str:
    """Replace detected PII with [NIK]/[PHONE]/[ACCOUNT] placeholders."""
    spans = detect_pii_id(text)
    for f in sorted(spans, key=lambda f: f["start"], reverse=True):
        text = text[: f["start"]] + _PLACEHOLDER[f["type"]] + text[f["end"] :]
    return text


def summarize_activated_rails(generation_response) -> list[str]:
    """Turn rails.generate(..., options={'log':{'activated_rails':True}}) into a
    readable list like ['self check input (BLOCKED)', ...]. Safe on objects with no log."""
    log = getattr(generation_response, "log", None)
    rails = getattr(log, "activated_rails", None) if log is not None else None
    if not rails:
        return []
    out = []
    for r in rails:
        name = getattr(r, "name", None) or getattr(r, "type", None) or "rail"
        blocked = bool(getattr(r, "stop", False))
        out.append(f"{name} ({'BLOCKED' if blocked else 'passed'})")
    return out


def nim_client(api_key_env: str = "NVIDIA_API_KEY",
               base_url: str = "https://integrate.api.nvidia.com/v1"):
    """openai.OpenAI client pointed at NVIDIA NIM. Reads the key from env (Colab Secrets)."""
    import os
    from openai import OpenAI
    key = os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"{api_key_env} not set — add it to Colab Secrets, then os.environ.")
    return OpenAI(base_url=base_url, api_key=key)


def build_rails(yaml_content: str, colang_content: str = ""):
    """nest_asyncio.apply() + RailsConfig.from_content -> LLMRails. One-call Colab helper."""
    import nest_asyncio
    nest_asyncio.apply()
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_content(yaml_content=yaml_content,
                                      colang_content=colang_content or None)
    return LLMRails(config)
