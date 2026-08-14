"""
Shared Ollama REST client for CASTOR extraction and judge scripts.
Handles markdown-fence stripping, LaTeX unescape, and JSON parsing of responses.
"""

import json
import re
import time
import urllib.error
import urllib.request


class OllamaClient:
    """Talks to a local Ollama server for the pipelines that use one.

    P2 (Gemma field extraction) and P3 (semantic judge) run against Ollama
    rather than the cluster's vLLM, so they can be developed and debugged on a
    laptop without a SLURM allocation. The cluster pipelines (P5-P8) use vLLM
    instead — this client is not involved there.

    Two settings are load-bearing:

      format="json"   asks Ollama to constrain output to valid JSON. It reduces
                      parse failures but does not eliminate them, which is why
                      every caller still handles a None parse.
      timeout 180s    a judge reasoning over a long plan genuinely takes over a
                      minute on CPU. A shorter timeout silently converts slow
                      records into failures and biases the result toward short
                      answers.

    Returns (parsed, raw, elapsed) rather than raising, so one bad record does
    not abort a run over a hundred images. The raw text is returned even on a
    parse failure — it is the only diagnostic left afterwards.
    """

    #: Ollama genuinely needs this long for a judge over a long plan on CPU.
    TIMEOUT_SECONDS = 180


def call_ollama(system: str, user: str, model: str, url: str,
                options: dict = None) -> tuple:
    """POST to Ollama /api/chat. Returns (parsed_dict | None, raw_str, elapsed_s).

    options dict is passed as Ollama model options (e.g. {"temperature": 0, "num_predict": 1024}).

    See OllamaClient for why the timeout and format settings matter.
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


def embed_ollama(text: str, model: str, url: str) -> list:
    """POST to Ollama /api/embeddings. Returns the embedding vector, or None
    on any HTTP/parse error (never raises)."""
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        return None

    try:
        outer = json.loads(raw)
        return outer["embedding"]
    except (json.JSONDecodeError, KeyError):
        return None
