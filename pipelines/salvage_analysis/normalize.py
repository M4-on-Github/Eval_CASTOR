"""
Pipeline 6 Stage 2 — cluster open-vocabulary salvage phrases into canonical
elements.

Stage 1 (extract.py) produces raw phrases ("call a fireboat", "dispatch
fireboat", "fireboat response") that need collapsing into one canonical
label per real-world concept before Stage 3 can build a contingency table.
Phrases are embedded (Ollama) and clustered by cosine distance
(AgglomerativeClustering) — the distance threshold has no default anywhere
in this module; it must be chosen deliberately per run and the resulting
clusters reviewed before Stage 3 is trusted (see SPEC_salvage_analysis.md
Boundaries and ADR-001).

Usage:
  python pipelines/salvage_analysis/normalize.py --run answers_baseline --threshold 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from sklearn.cluster import AgglomerativeClustering

from shared.ollama import embed_ollama

OUT_DIR = EVAL_ROOT / "results" / "p6_salvage_plan"

DEFAULT_MODEL = os.environ.get("CASTOR_SALVAGE_MODEL", "gemma4:31b-cloud")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/embeddings"


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def cluster_phrases(phrase_to_vector: dict, threshold: float) -> dict:
    """Cluster raw phrases by cosine distance between their embeddings.

    Returns {raw_phrase: canonical_label}, where canonical_label is one of
    the original phrases in that cluster (the shortest one, tie-broken
    alphabetically). threshold has no default — callers must pass one
    explicitly (see module docstring)."""
    phrases = list(phrase_to_vector.keys())
    if not phrases:
        return {}
    if len(phrases) == 1:
        return {phrases[0]: phrases[0]}

    vectors = [phrase_to_vector[p] for p in phrases]
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(vectors)

    clusters = {}
    for phrase, label in zip(phrases, labels):
        clusters.setdefault(label, []).append(phrase)

    mapping = {}
    for members in clusters.values():
        canonical = sorted(members, key=lambda p: (len(p), p))[0]
        for phrase in members:
            mapping[phrase] = canonical
    return mapping


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def collect_unique_phrases(raw_elements_path: Path) -> list:
    phrases = set()
    with open(raw_elements_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for phrase in rec.get("raw_elements", []):
                phrases.add(phrase)
    return sorted(phrases)


def run(run_name: str, threshold: float, model: str, url: str):
    raw_elements_path = OUT_DIR / f"raw_elements_{run_name}.jsonl"
    phrases = collect_unique_phrases(raw_elements_path)
    print(f"  {len(phrases)} unique raw phrases to embed.")

    phrase_to_vector = {}
    for phrase in phrases:
        vector = embed_ollama(phrase, model, url)
        if vector is None:
            print(f"  WARNING: embedding failed for phrase: {phrase!r}")
            continue
        phrase_to_vector[phrase] = vector

    mapping = cluster_phrases(phrase_to_vector, threshold)

    out_path = OUT_DIR / f"elements_{run_name}.json"
    out_path.write_text(
        json.dumps({"raw_to_canonical": mapping, "threshold": threshold}, indent=2),
        encoding="utf-8",
    )
    print(f"  {len(set(mapping.values()))} canonical elements from {len(mapping)} phrases.")
    print(f"  Output -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Stage 2: cluster salvage phrases into canonical elements (Pipeline 6)"
    )
    ap.add_argument("--run", required=True, help="Run name (matches extract.py's --run)")
    ap.add_argument("--threshold", type=float, required=True,
                     help="Cosine distance threshold for clustering — no default; "
                          "pick deliberately and review the output before trusting it")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    args = ap.parse_args()

    run(args.run, args.threshold, args.model, args.url)


if __name__ == "__main__":
    main()
