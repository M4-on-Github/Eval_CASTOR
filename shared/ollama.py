"""
Shared Ollama REST client for CASTOR extraction and judge scripts.
Handles markdown-fence stripping, LaTeX unescape, and JSON parsing of responses.
"""

import json
import re
import time
import urllib.error
import urllib.request


def call_ollama(system: str, user: str, model: str, url: str,
                options: dict = None) -> tuple:
    """POST to Ollama /api/chat. Returns (parsed_dict | None, raw_str, elapsed_s).

    options dict is passed as Ollama model options (e.g. {"temperature": 0, "num_predict": 1024}).
    """
    payload_obj = {
        "model":    model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "format": "json",
    }
    if options:
        payload_obj["options"] = options

    payload = json.dumps(payload_obj).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
        elapsed = time.perf_counter() - t0
    except (urllib.error.URLError, TimeoutError) as e:
        return None, f"HTTP_ERROR: {e}", time.perf_counter() - t0

    try:
        outer   = json.loads(raw)
        content = outer["message"]["content"]
    except (json.JSONDecodeError, KeyError) as e:
        return None, f"RESPONSE_PARSE_ERROR: {e} | raw={raw[:200]}", time.perf_counter() - t0

    # Strip markdown fences that models sometimes add despite format:"json"
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped)
        stripped = re.sub(r'\s*```\s*$', '', stripped).strip()

    # Unescape LaTeX-style escapes (e.g. on\_fire → on_fire) that break JSON
    stripped = re.sub(r'\\([_\-/])', r'\1', stripped)

    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            return None, f"NOT_DICT: {stripped[:200]}", elapsed
        return parsed, stripped, elapsed
    except json.JSONDecodeError as e:
        return None, f"CONTENT_JSON_ERROR: {e} | content={content[:300]}", elapsed
