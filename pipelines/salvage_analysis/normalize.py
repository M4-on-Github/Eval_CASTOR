"""
Pipeline 6 Stage 2 — cluster open-vocabulary salvage phrases into canonical
elements.

Stage 1 (extract.py) produces raw phrases ("call a fireboat", "dispatch
fireboat", "fireboat response") that need collapsing into one canonical
label per real-world concept before Stage 3 can build a contingency table.
Phrases are embedded and clustered by cosine distance
(AgglomerativeClustering) — the distance threshold has no default anywhere
in this module; it must be chosen deliberately per run and the resulting
clusters reviewed before Stage 3 is trusted (see SPEC_salvage_analysis.md
Boundaries and ADR-001).

Two backends:
  --backend ollama (default) -- local dev, Ollama's /api/embeddings (needs a
    locally-running Ollama server built with --embeddings support).
  --backend local -- cluster runs, no Ollama available there. Embeds phrases
    with a small local sentence-transformers model (no network at runtime
    once the checkpoint is cached) -- same clustering math either way.

Usage:
  python pipelines/salvage_analysis/normalize.py --run answers_baseline --threshold 0.3
  python pipelines/salvage_analysis/normalize.py --run answers_baseline --threshold 0.3 --backend local
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
from pipelines.salvage_analysis import paths

DEFAULT_MODEL = os.environ.get("CASTOR_SALVAGE_MODEL", "gemma4:31b-cloud")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/embeddings"
DEFAULT_LOCAL_EMBED_MODEL = os.environ.get("CASTOR_SALVAGE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

class PhraseClusterer:
    """Collapses many phrasings of one salvage action into a canonical label.

    Models describe the same action a dozen ways — "deploy tugs", "bring in
    tugboats", "tug assistance" — and every distinct phrasing would otherwise
    count as its own element, scattering the evidence for one concept across
    a dozen near-empty rows and destroying the Stage 4 statistics.

    Clustering is agglomerative over cosine distance between embeddings, with
    average linkage: a phrase joins a cluster on its mean distance to the
    members, so one outlier cannot pull in an unrelated phrase the way single
    linkage would.

    THE THRESHOLD HAS NO DEFAULT, deliberately. It is the single knob that
    decides how aggressively distinct concepts are merged, and a wrong value
    fails silently in both directions — too high fuses unrelated actions into
    one element, too low leaves synonyms scattered. Neither raises; both just
    change the finding. Callers must state a value.
    """

    #: Average linkage over cosine distance — see the class docstring.
    METRIC = "cosine"
    LINKAGE = "average"

    def __init__(self, threshold: float):
        self.threshold = threshold

    @staticmethod
    def canonical_label(members: list) -> str:
        """Pick the cluster's representative: shortest phrase, ties alphabetical.

        Shortest because the briefest phrasing is usually the least
        model-specific ("deploy tugs" over "bring in tugboat assistance from
        the nearest port"). Alphabetical tie-break keeps the choice
        deterministic, so re-running produces the same element names and two
        runs remain comparable.
        """
        return sorted(members, key=lambda p: (len(p), p))[0]

    def cluster(self, phrase_to_vector: dict) -> dict:
        """Return {raw_phrase: canonical_label}."""
        phrases = list(phrase_to_vector.keys())
        if not phrases:
            return {}
        if len(phrases) == 1:
            # AgglomerativeClustering needs at least two samples.
            return {phrases[0]: phrases[0]}

        vectors = [phrase_to_vector[p] for p in phrases]
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.threshold,
            metric=self.METRIC,
            linkage=self.LINKAGE,
        )
        labels = clustering.fit_predict(vectors)

        clusters = {}
        for phrase, label in zip(phrases, labels):
            clusters.setdefault(label, []).append(phrase)

        mapping = {}
        for members in clusters.values():
            canonical = self.canonical_label(members)
            for phrase in members:
                mapping[phrase] = canonical
        return mapping


def cluster_phrases(phrase_to_vector: dict, threshold: float) -> dict:
    """Cluster raw phrases by cosine distance between their embeddings.

    Returns {raw_phrase: canonical_label}, where canonical_label is one of
    the original phrases in that cluster (the shortest one, tie-broken
    alphabetically). threshold has no default — callers must pass one
    explicitly (see module docstring).

    Facade over PhraseClusterer.
    """
    return PhraseClusterer(threshold).cluster(phrase_to_vector)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def embed_local(phrases: list, model_name: str) -> dict:
    """Embed phrases with a local sentence-transformers model -- no network,
    no Ollama. Used on the cluster where no embedding service is available."""
    from sentence_transformers import SentenceTransformer

    print(f"  [local] Loading embedding model {model_name} ...")
    model = SentenceTransformer(model_name)
    vectors = model.encode(phrases, convert_to_numpy=True)
    return {phrase: vector.tolist() for phrase, vector in zip(phrases, vectors)}


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


def run(run_name: str, threshold: float, backend: str = "ollama",
        model: str = DEFAULT_MODEL, url: str = DEFAULT_OLLAMA_URL,
        embed_model: str = DEFAULT_LOCAL_EMBED_MODEL):
    phrases = collect_unique_phrases(paths.raw_elements_path(run_name))
    print(f"  {len(phrases)} unique raw phrases to embed.")

    if backend == "local":
        phrase_to_vector = embed_local(phrases, embed_model)
    else:
        phrase_to_vector = {}
        for phrase in phrases:
            vector = embed_ollama(phrase, model, url)
            if vector is None:
                print(f"  WARNING: embedding failed for phrase: {phrase!r}")
                continue
            phrase_to_vector[phrase] = vector

    mapping = cluster_phrases(phrase_to_vector, threshold)

    out_path = paths.elements_path(run_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--backend", choices=["ollama", "local"], default="ollama",
                    help="'ollama' for local dev (needs Ollama with --embeddings support); "
                         "'local' for cluster runs (sentence-transformers, no network)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama embedding model (--backend ollama only)")
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama embeddings endpoint (--backend ollama only)")
    ap.add_argument("--embed-model", default=DEFAULT_LOCAL_EMBED_MODEL,
                    help="sentence-transformers model name (--backend local only)")
    args = ap.parse_args()

    run(args.run, args.threshold, args.backend, args.model, args.url, args.embed_model)


if __name__ == "__main__":
    main()
