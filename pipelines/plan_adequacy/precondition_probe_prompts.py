"""
precondition_probe_prompts.py -- candidate selection + prompt construction
for Phase 2 and Phase 3 of the revised Direction-4 protocol.

Builds two ready-to-run JSONL files. Neither requires the cluster to
build; both require a short cluster session to actually query the model
(text-only, no image -- see the module docstring section "cluster-side
adaptation needed" below for exactly what run_inference.py needs).

PHASE 2 -- knowledge-vs-application. For each distinct (tool, missing
fact) precondition relationship the registry defines, sample violator
plans (a SEQUENCE_VIOLATION on that exact relationship, from the existing
330-plan corpus) and matched non-violator plans (a plan for the same
casualty that called the same tool WITHOUT violating that precondition).
Each sampled plan becomes one row: an isolated, text-only, no-image
yes/no+justification question about the relationship, built from the
registry's own sourced `requires`/`source` fields -- ground truth is
already in the registry, no new labelling needed.

PHASE 3 -- causal position test. For each relationship with a genuinely
VERIFIED-clean base plan available (inject.py's 4 casualty bases, each
verified clean by test -- see realizable_relationships()), truncate that
real plan text to k=1..max_usable_k steps, where max_usable_k =
gating_step-1 is derived per relationship, not assumed uniform: a
relationship gated at step 3 can only support k=1..2, and forcing k up
to 5 there would require padding with content unrelated to the
relationship, contaminating the "same content, only length varies"
design. Of the 29 distinct relationships the registry defines, 16 have
a verified-clean base to truncate from as of 2026-09-09; the other 13
(tools like lighter_cargo, dredge, remove_impalement that never appear
in any of the 4 verified bases) are recorded as skipped, not fabricated
-- see build_phase3_jsonl's return value. Deterministic/greedy decoding
means each (item, k) is one generation call, not a resampled
distribution -- item is the unit of replication (see
precondition_position_probe.py's Phase-1 finding on why this matters).

Cluster-side generation is already built: precondition_probe.py (this
directory) and submit_precondition_probe.sh (this directory). The actual
generation script for this corpus,
pipelines/plan_coherence/improved/inference/run_inference.py, is
image-driven -- it loads Qwen3-VL-8B with an image tensor on every call,
and Phase 2/3 prompts carry NO image -- so precondition_probe.py copies
its model-loading block verbatim but replaces the encode step, and forces
greedy decoding unconditionally. It lives under plan_adequacy/ (this is
P9's own diagnostic work) but still runs inside castor_qwen.sif, not P9's
usual castor_judge.sif -- see submit_precondition_probe.sh's header for
why that cross-pipeline container dependency is fine (both live under
/data/$USER/, nothing to build).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

REGISTRY_PATH = EVAL_ROOT / "pipelines" / "plan_adequacy" / "registry" / "tools.json"

RUNS = {
    "None":     "answers_qwen3vl8b_baseline_ablation_v2_improved",
    "Vague":    "answers_qwen3vl8b_baseline_control_v2_improved",
    "Specific": "answers_qwen3vl8b_baseline_standard_v2_improved",
}

#: Casualty each tool's family belongs to, for phrasing the question
#: naturally ("For a CASUALTY vessel, ..."). "universal" tools are asked
#: generically ("in a salvage operation").
_CASUALTY_PHRASE = {
    "aground": "a vessel that has run aground",
    "capsized": "a capsized vessel",
    "sunken": "a sunken or submerged vessel",
    "on_fire": "a vessel on fire",
    "universal": "a salvage operation",
    "assessment": "a salvage operation",
    "terminal": "a salvage operation",
}


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def distinct_relationships(registry: dict) -> list:
    """One row per (tool, missing_fact) pair any tool's `requires` names.
    This is the finite, near-census population Phase 2/3 draw from --
    the registry currently defines on the order of 10-13 of these."""
    out = []
    for tool in registry["tools"]:
        if not tool.get("requires"):
            continue
        for fact in tool["requires"]:
            out.append({
                "tool": tool["name"],
                "fact": fact,
                "family": tool["family"],
                "source": tool.get("source", ""),
            })
    return out


def phase2_question(rel: dict) -> str:
    """Isolated, text-only yes/no+justification question. No image, no
    plan-writing context -- this is the whole point of Phase 2."""
    casualty_phrase = _CASUALTY_PHRASE.get(rel["family"], "a salvage operation")
    fact_phrase = rel["fact"].replace("_", " ")
    return (
        f"In salvage operations, for {casualty_phrase}: must \"{fact_phrase}\" "
        f"be established before the action \"{rel['tool'].replace('_', ' ')}\" is "
        f"carried out? Answer yes or no, then explain your reasoning in one "
        f"or two sentences."
    )


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_phase2_candidates(base_dir: Path, relationships: list) -> dict:
    """{(tool, fact): {"violators": [...], "non_violators": [...]}}.
    A violator is a plan whose SEQUENCE_VIOLATION detail names this exact
    fact for this exact tool. A non-violator is a plan that called the
    same tool without that fact missing (detail doesn't name it, or the
    call graded cleanly)."""
    by_rel = {(r["tool"], r["fact"]): {"violators": [], "non_violators": []}
              for r in relationships}

    for arm, run in RUNS.items():
        per_step = _load_csv(base_dir / run / "per_step.csv")
        for row in per_step:
            tool = row["tool"]
            for (rtool, rfact), bucket in by_rel.items():
                if tool != rtool:
                    continue
                entry = {"arm": arm, "image": row["image"],
                        "casualty": row["casualty"], "step_num": row["step_num"]}
                if row["verdict"] == "SEQUENCE_VIOLATION" and rfact in row["detail"]:
                    bucket["violators"].append(entry)
                elif row["verdict"] not in (
                        "SEQUENCE_VIOLATION", "NO_MATCH", "METHOD_ERROR"):
                    bucket["non_violators"].append(entry)
    return by_rel


def build_phase2_jsonl(base_dir: Path, out_path: Path, max_per_group: int = 15):
    registry = load_registry()
    relationships = distinct_relationships(registry)
    candidates = find_phase2_candidates(base_dir, relationships)

    rows = []
    for rel in relationships:
        key = (rel["tool"], rel["fact"])
        bucket = candidates[key]
        for group, entries in (("violator", bucket["violators"]),
                               ("non_violator", bucket["non_violators"])):
            for entry in entries[:max_per_group]:
                rows.append({
                    "relationship": f"{rel['tool']}::{rel['fact']}",
                    "group": group,
                    "question": phase2_question(rel),
                    "registry_source": rel["source"],
                    **entry,
                })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_rel_with_both = sum(
        1 for rel in relationships
        if candidates[(rel["tool"], rel["fact"])]["violators"]
        and candidates[(rel["tool"], rel["fact"])]["non_violators"])
    print(f"Phase 2: {len(relationships)} distinct relationships, "
          f"{n_rel_with_both} have both violator and non-violator cases available, "
          f"{len(rows)} question rows written -> {out_path}")
    return rows


def realizable_relationships(registry: dict) -> dict:
    """{(tool, fact): {casualty, gating_step, max_usable_k}} -- the subset of
    distinct_relationships() that a genuinely verified-clean base plan
    (inject.py's BASES, verified clean by test, not assumed) actually
    exercises. Replays each of the 4 bases through the real worldstate
    (asserting missing_requires() is empty at every step, exactly as
    inject.py itself asserts) and records, for every requires-check that
    passes, which step gated it.

    max_usable_k = gating_step - 1: the longest prefix that still ends
    BEFORE the gated action, which is the only prefix length Phase 3 can
    honestly test -- asking for a k longer than the base plan has before
    that action would require padding with content unrelated to the
    relationship under test, contaminating the "same content, only length
    varies" design principle.

    Deliberately does NOT cover every relationship distinct_relationships()
    returns: registry tools with no requires-check inside any of the 4
    verified bases (e.g. lighter_cargo, dredge, remove_impalement, several
    others) have no verified-clean prefix to truncate, and this function
    will not fabricate one. That is a real, reported scope reduction, not
    a bug -- see build_phase3_jsonl's docstring.
    """
    from pipelines.plan_adequacy.inject import BASES
    from pipelines.plan_adequacy.vocab import ToolRegistry
    from pipelines.plan_adequacy.worldstate import WorldState

    tool_registry = ToolRegistry.load()
    out = {}
    for casualty, base_fn in BASES.items():
        calls = base_fn()
        ws = WorldState()
        for call in calls:
            spec = tool_registry.spec(call.tool)
            missing = ws.missing_requires(call, tool_registry)
            assert not missing, (
                f"base plan no longer clean: {casualty} step {call.step_num} "
                f"missing {missing} -- a registry change broke inject.py's "
                f"verified base; fix that before trusting this module's output")
            for fact in spec.requires:
                key = (call.tool, fact)
                if key not in out:  # first (earliest) gating step wins
                    out[key] = {"casualty": casualty,
                               "gating_step": call.step_num,
                               "max_usable_k": call.step_num - 1,
                               "base_steps": [(c.step_num, c.step_text) for c in calls]}
            ws.apply(call, tool_registry)
    return out


def build_phase3_jsonl(out_path: Path, registry: dict = None):
    """One row per (relationship, k), k ranging 1..max_usable_k PER
    relationship (not a fixed grid -- see realizable_relationships()).
    prefix_plan_text is real, verified-clean plan text truncated from
    inject.py's own base plans, never fabricated.

    Registry relationships with no verified-clean base (13 of 29 as of
    2026-09-09) are recorded in the returned `skipped` list and reported,
    not silently dropped -- this narrows Phase 3's census from 29 to
    whatever verified count is actually available, and that narrowing is
    itself a result worth keeping visible.
    """
    registry = registry or load_registry()
    all_relationships = distinct_relationships(registry)
    realizable = realizable_relationships(registry)

    rows, skipped = [], []
    for rel in all_relationships:
        key = (rel["tool"], rel["fact"])
        info = realizable.get(key)
        if info is None:
            skipped.append(f"{rel['tool']}::{rel['fact']}")
            continue
        base_steps = sorted(info["base_steps"])
        for k in range(1, info["max_usable_k"] + 1):
            prefix_text = "\n".join(f"{n}. {text}" for n, text in base_steps[:k])
            rows.append({
                "relationship": f"{rel['tool']}::{rel['fact']}",
                "tool": rel["tool"], "fact": rel["fact"], "family": rel["family"],
                "casualty": info["casualty"], "gating_step": info["gating_step"],
                "k": k,
                "registry_source": rel["source"],
                "prefix_plan_text": prefix_text,
                "continuation_prompt": (
                    f"Continue this salvage plan. Write only the next step, "
                    f"numbered step {k + 1}."
                ),
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Phase 3: {len(all_relationships)} distinct relationships in the registry; "
          f"{len(realizable)} have a verified-clean base to truncate from "
          f"({len(skipped)} skipped, no verified-clean base: {', '.join(skipped)}); "
          f"{len(rows)} rows written -> {out_path}")
    return rows, skipped


def main():
    ap = argparse.ArgumentParser(description="Phase 2/3 candidate + prompt builder")
    ap.add_argument("--dir", required=True, help="results base directory (for Phase 2 candidates)")
    ap.add_argument("--phase2-out", default="phase2_candidates.jsonl")
    ap.add_argument("--phase3-out", default="phase3_items.jsonl")
    args = ap.parse_args()

    build_phase2_jsonl(Path(args.dir), Path(args.phase2_out))
    build_phase3_jsonl(Path(args.phase3_out))


if __name__ == "__main__":
    main()
