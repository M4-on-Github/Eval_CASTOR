"""
worksheet.py -- generate the Phase 3 hand-coding worksheets.

Two columns of the diagnosis table cannot be filled by any amount of code:
whether a label is CORRECT on real plans, and whether a failure is the
planner's fault or ours. Those need a person reading actual plans, and this
module packages the items for that.

Each worksheet is a single self-contained HTML file -- no server, no build, no
network -- opened in a browser. One item per card, machine labels HIDDEN,
options as buttons, progress kept in localStorage so the pass can be
interrupted, and an export button that writes a JSON the ingest side reads
back.

Blinding is the point of the design, not a nicety. The coder must not see what
the classifier decided, or the exercise measures agreement with a prompt
rather than agreement with the plan. The machine label is carried in the file
(the export has to be joinable) but is never rendered and never revealed by
the UI.

THE THREE TASKS, and why these three. The task list was rebuilt after Phase 1
changed what is actually uncertain:

  1. PERCEPTION (n=40). Is this plan really describing a different casualty?
     The detector was written for this redesign and has never been checked
     against a human, and its one free parameter -- how many
     foreign-DISTINCTIVE tools a plan must call, methods.
     MIN_DISTINCTIVE_FOREIGN_TOOLS -- moves the class from 118 plans (at 1)
     to 33 (at 2). A 3.6x swing on the second-largest class is not a
     parameter anyone should pick by taste.

     So the sample is drawn deliberately from BOTH sides of that boundary:
     half from plans both thresholds flag (CONFIRMED) and half from plans
     only the permissive threshold flags (DISPUTED). Coding the two together
     estimates precision at each setting from one pass, which is the only
     reason to spend 40 items here rather than 20. The coder is not told
     which group an item is in.

     (The original plan budgeted 30 no-route plans here; NO_PROCEDURE
     collapsed once the cross-casualty match landed, so that budget moved to
     where the uncertainty actually went.)

  2. MAGNITUDE (n=60). Does this step commit to a quantity? Validates the
     digit-and-unit proxy that decides the COMMITMENT class outright.

  3. ATTRIBUTION (n=40, stratified across classes). Planner, registry,
     extractor, or ambiguous -- the % planner column, and the only thing that
     stops an instrument artefact being reported as a planner finding.

Sampling is deterministic: sort by image id, take every k-th item, so the same
corpus always yields the same worksheet and a rerun cannot quietly reshuffle
which plans got looked at.
"""

import csv
import html
import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

TASKS = {
    "perception": {
        "title": "Is this plan describing the wrong casualty?",
        "question": "The photo shows a <b>{casualty}</b> vessel. Reading only the plan, "
                    "what kind of casualty is it actually written for?",
        "options": [
            ("same", "The plan matches the real casualty ({casualty})"),
            ("aground", "It reads as an <b>aground</b> plan"),
            ("capsized", "It reads as a <b>capsized</b> plan"),
            ("on_fire", "It reads as an <b>on_fire</b> plan"),
            ("sunken", "It reads as a <b>sunken</b> plan"),
            ("generic", "Too generic to tell -- no casualty-specific technique"),
        ],
    },
    "magnitude": {
        "title": "Does this step commit to a quantity?",
        "question": "Does this step state a magnitude an executor could act on "
                    "without asking anyone a further question?",
        "options": [
            ("yes", "Yes -- a usable quantity is stated"),
            ("no", "No -- the quantity is deferred, implied, or absent"),
            ("na", "No quantity is needed for this action"),
        ],
    },
    "attribution": {
        "title": "Whose failure is this?",
        "question": "This plan was graded as failing. Reading the steps, where does "
                    "the fault actually lie?",
        "options": [
            ("planner", "The <b>planner</b> -- the plan really is wrong"),
            ("registry", "Our <b>registry</b> -- the action is sound but we have no tool/route for it"),
            ("extractor", "Our <b>extractor</b> -- the step says the right thing, it was parsed wrong"),
            ("ambiguous", "Genuinely unclear"),
        ],
    },
}

_CSS = """
:root{--ink:#1a2226;--faint:#6F7C83;--rule:#9FADA7;--bg:#fbfaf8;--card:#fff;
--blue:#1C5D78;--green:#3C6E52;--sunk:#E7ECEA}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--rule);
padding:14px 22px;display:flex;gap:18px;align-items:baseline;z-index:10}
h1{font-size:16px;margin:0;font-weight:650}
.meta{color:var(--faint);font-size:13px}
.bar{flex:1;height:5px;background:var(--sunk);border-radius:3px;overflow:hidden;max-width:280px}
.bar>i{display:block;height:100%;background:var(--green);width:0}
button{font:inherit;cursor:pointer}
.export{border:1px solid var(--blue);background:var(--blue);color:#fff;
padding:7px 15px;border-radius:5px}
.export[disabled]{opacity:.4;cursor:not-allowed}
main{max-width:820px;margin:0 auto;padding:22px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:7px;
padding:18px 20px;margin-bottom:18px}
.card.done{opacity:.5}
.id{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--faint);
margin-bottom:10px;display:flex;justify-content:space-between}
.q{font-size:14px;color:var(--faint);margin:14px 0 9px}
ol.steps{margin:0;padding-left:22px}
ol.steps li{margin-bottom:7px}
.steptext{white-space:pre-wrap}
.single{background:var(--sunk);padding:12px 14px;border-radius:5px;white-space:pre-wrap}
.opts{display:flex;flex-direction:column;gap:6px}
.opts button{text-align:left;border:1px solid var(--rule);background:var(--card);
padding:8px 12px;border-radius:5px}
.opts button:hover{border-color:var(--blue)}
.opts button[aria-pressed="true"]{background:var(--green);border-color:var(--green);color:#fff}
.opts button[aria-pressed="true"] b{color:#fff}
.note{width:100%;margin-top:8px;font:inherit;padding:7px 10px;border:1px solid var(--rule);
border-radius:5px;background:var(--card);color:var(--ink)}
footer{max-width:820px;margin:0 auto;padding:0 22px 60px;color:var(--faint);font-size:13px}
"""

_JS = """
const KEY = 'p9ws:' + TASK;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { saved = {}; }

function paint() {
  let n = 0;
  document.querySelectorAll('.card').forEach(card => {
    const id = card.dataset.id, rec = saved[id];
    card.querySelectorAll('.opts button').forEach(b =>
      b.setAttribute('aria-pressed', String(!!rec && rec.choice === b.dataset.val)));
    const note = card.querySelector('.note');
    if (note && rec && rec.note != null && note.value !== rec.note) note.value = rec.note;
    if (rec) { n++; card.classList.add('done'); } else { card.classList.remove('done'); }
  });
  const total = document.querySelectorAll('.card').length;
  document.getElementById('count').textContent = n + ' / ' + total;
  document.querySelector('.bar>i').style.width = (100 * n / total) + '%';
  document.querySelector('.export').disabled = n === 0;
}

function persist() {
  try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {}
  paint();
}

document.addEventListener('click', e => {
  const b = e.target.closest('.opts button');
  if (!b) return;
  const card = b.closest('.card'), id = card.dataset.id;
  const note = card.querySelector('.note');
  if (saved[id] && saved[id].choice === b.dataset.val) delete saved[id];
  else saved[id] = { choice: b.dataset.val, note: note ? note.value : '' };
  persist();
});

document.addEventListener('input', e => {
  if (!e.target.classList.contains('note')) return;
  const id = e.target.closest('.card').dataset.id;
  if (saved[id]) { saved[id].note = e.target.value; persist(); }
});

document.querySelector('.export').addEventListener('click', () => {
  const out = { task: TASK, coded_at: new Date().toISOString(), labels: saved };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url; a.download = 'p9_coded_' + TASK + '.json'; a.click();
  URL.revokeObjectURL(url);
});

paint();
"""


def _read(path, run=None):
    """Rows, each tagged with the run it came from.

    The tag is load-bearing, not bookkeeping. per_step.csv and per_image.csv
    carry `image` but no arm column, and the SAME image appears in all three
    arms -- so keying anything by image alone silently merges three different
    plans. That produced 18-step cards (3 arms x 6 steps) in the first
    generated worksheet, which would have made every perception judgement
    meaningless. Item ids are therefore (run, image), rendered "arm | image".
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    run = run or Path(path).parent.name
    for r in rows:
        r["_run"] = run
        r["_key"] = f'{run}|{r["image"]}'
    return rows


def _arm(run: str) -> str:
    """Short arm label for display: the worksheet shows which arm a plan came
    from, since two arms' plans for the same image are different plans."""
    for name in ("ablation", "standard", "control"):
        if name in run:
            return name
    return run


def _every_kth(items, n, key):
    """Deterministic subsample: sort by `key`, take every k-th. No seed, no
    sampling -- the same corpus always yields the same worksheet, so a rerun
    cannot quietly change which items were looked at."""
    items = sorted(items, key=key)
    if len(items) <= n:
        return items
    k = len(items) / n
    return [items[int(i * k)] for i in range(n)]


def _render(task, items, out_path):
    spec = TASKS[task]
    cards = []
    for it in items:
        opts = "".join(
            f'<button data-val="{html.escape(v)}">'
            f'{lbl.format(casualty=html.escape(it.get("casualty", "")))}</button>'
            for v, lbl in spec["options"])
        if it.get("steps"):
            body = "<ol class='steps'>" + "".join(
                f"<li><span class='steptext'>{html.escape(s)}</span></li>"
                for s in it["steps"]) + "</ol>"
        else:
            body = f"<div class='single'>{html.escape(it['text'])}</div>"
        cards.append(
            f'<div class="card" data-id="{html.escape(it["id"])}">'
            f'<div class="id"><span>{html.escape(it["id"])}</span>'
            f'<span>{html.escape(it.get("hint", ""))}</span></div>'
            f'{body}'
            f'<div class="q">{spec["question"].format(casualty=html.escape(it.get("casualty", "")))}</div>'
            f'<div class="opts">{opts}</div>'
            f'<input class="note" placeholder="optional note">'
            f'</div>')

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P9 coding &middot; {html.escape(spec['title'])}</title>
<style>{_CSS}</style></head><body>
<header>
  <h1>{html.escape(spec['title'])}</h1>
  <span class="meta" id="count">0 / 0</span>
  <span class="bar"><i></i></span>
  <button class="export" disabled>Export JSON</button>
</header>
<main>{''.join(cards)}</main>
<footer>Progress is saved in this browser as you go, so you can stop and come back.
Machine labels are deliberately not shown. Export when finished.</footer>
<script>const TASK = {json.dumps(task)};{_JS}</script>
</body></html>"""
    Path(out_path).write_text(doc, encoding="utf-8")
    return len(items)


# ── item builders ────────────────────────────────────────────────────────────

def _perception_groups(run_dirs):
    """{image: "confirmed"|"disputed"} over every plan either threshold flags.

    confirmed = flagged at MIN_DISTINCTIVE_FOREIGN_TOOLS (the shipped value);
    disputed  = flagged only by the permissive threshold of 1.
    """
    from pipelines.plan_adequacy.methods import (MIN_DISTINCTIVE_FOREIGN_TOOLS,
                                                 RouteRegistry,
                                                 detect_perception_mismatch)
    from pipelines.plan_adequacy.run_executor import group_tool_calls
    from pipelines.plan_adequacy.vocab import ToolRegistry

    rr, tr = RouteRegistry.load(), ToolRegistry.load()
    groups = {}
    for d in run_dirs:
        calls_path = Path(d) / "tool_calls.jsonl"
        if not calls_path.exists():
            continue
        for image, calls in group_tool_calls(calls_path).items():
            casualty = image.split("/")[0]
            if casualty not in rr.all_casualties():
                continue
            names = {c.tool for c in calls if c.tool != "no_match" and tr.has(c.tool)}
            strict = detect_perception_mismatch(names, casualty, rr,
                                                MIN_DISTINCTIVE_FOREIGN_TOOLS)
            loose = detect_perception_mismatch(names, casualty, rr, 1)
            key = f"{Path(d).name}|{image}"
            if strict:
                groups[key] = "confirmed"
            elif loose:
                groups[key] = "disputed"
    return groups


def _steps_by_key(per_step):
    steps = {}
    for r in per_step:
        steps.setdefault(r["_key"], []).append((int(r["step_num"]), r["step_text"]))
    return {k: [t for _, t in sorted(v)] for k, v in steps.items()}


def items_perception(per_image, per_step, n=40, run_dirs=()):
    steps = _steps_by_key(per_step)
    by_key = {r["_key"]: r for r in per_image}

    groups = _perception_groups(run_dirs) if run_dirs else {}
    if not groups:      # no tool_calls available -- fall back to the class label
        groups = {r["_key"]: "confirmed" for r in per_image
                  if r["failure_class"] == "STRATEGY_PERCEPTION"}

    out = []
    for group, share in (("confirmed", n // 2), ("disputed", n - n // 2)):
        keys = [k for k, g in groups.items() if g == group and k in by_key]
        for key in _every_kth(keys, share, key=lambda k: k):
            run, image = key.split("|", 1)
            out.append({"id": key, "casualty": by_key[key]["casualty"],
                        "hint": f"{_arm(run)} arm",
                        "steps": steps.get(key, [])})
    # Interleave so the two groups are not visually separated in the file --
    # a coder who notices a block boundary has been partially unblinded.
    out.sort(key=lambda it: it["id"])
    return out


def _wants_numeric(tool, tool_registry):
    if not tool_registry.has(tool):
        return False
    return any(t.startswith(("int", "float"))
               for t in tool_registry.spec(tool).params.values())


def items_magnitude(per_step, n=60, tool_registry=None):
    """Half from steps the checker called UNSPECIFIED, half from
    SPECIFIED_UNGRADED. Both halves, because the proxy can fail in either
    direction and a one-sided sample measures recall or precision but never
    both: "at least two tugboats" states a count the digit rule misses, while
    a step whose only digit is ">50 m" of vessel size gets credited with an
    action magnitude it never stated.

    BOTH halves are restricted to tools that actually declare a numeric
    parameter. Without that restriction the SPECIFIED half is useless: a first
    version drew it from all tools and 29 of 30 items landed on assessment and
    terminal tools (survey_hull, post_operation_assessment, ...) which take no
    magnitude at all, so the question "does this commit a quantity" was vacuous
    for all but one item. UNSPECIFIED can only fire on a numeric-param tool, so
    that half was always in scope; only the SPECIFIED side was leaking.
    """
    from pipelines.plan_adequacy.vocab import ToolRegistry
    tool_registry = tool_registry or ToolRegistry.load()
    out = []
    for verdict, share in (("UNSPECIFIED", n // 2), ("SPECIFIED_UNGRADED", n - n // 2)):
        rows = [r for r in per_step
                if r["verdict"] == verdict and r["tool"] != "no_match"
                and _wants_numeric(r["tool"], tool_registry)]
        for r in _every_kth(rows, share, key=lambda r: (r["_key"], int(r["step_num"]))):
            out.append({"id": f'{r["_key"]}#{r["step_num"]}',
                        "text": r["step_text"],
                        "hint": f'{r["tool"]}  ({_arm(r["_run"])})'})
    return out


def items_attribution(per_image, per_step, n=40):
    """Stratified across failing classes so no class's attribution rests on a
    handful of plans."""
    steps = _steps_by_key(per_step)
    classes = sorted({r["failure_class"] for r in per_image
                      if r["failure_class"] not in ("VALID", "INCOMPLETE")})
    per_class = max(1, n // max(1, len(classes)))
    out = []
    for cls in classes:
        rows = [r for r in per_image if r["failure_class"] == cls]
        for r in _every_kth(rows, per_class, key=lambda r: r["_key"]):
            fs = r["failure_step"]
            where = f"first problem at step {fs}" if fs else "no step executed"
            out.append({"id": r["_key"], "casualty": r["casualty"],
                        "steps": steps.get(r["_key"], []),
                        "hint": f'{where}  ({_arm(r["_run"])})'})
    return out


def generate(run_dirs, out_dir, n_perception=40, n_magnitude=60, n_attribution=40):
    per_image, per_step = [], []
    for d in run_dirs:
        per_image += _read(Path(d) / "per_image.csv")
        per_step += _read(Path(d) / "per_step.csv")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = {}
    made["perception"] = _render("perception",
                                  items_perception(per_image, per_step, n_perception, run_dirs),
                                  out_dir / "p9_code_perception.html")
    made["magnitude"] = _render("magnitude", items_magnitude(per_step, n_magnitude),
                                 out_dir / "p9_code_magnitude.html")
    made["attribution"] = _render("attribution", items_attribution(per_image, per_step, n_attribution),
                                   out_dir / "p9_code_attribution.html")
    return made


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate P9 hand-coding worksheets")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    made = generate([Path(args.dir) / r for r in args.runs], args.out)
    for task, n in made.items():
        print(f"  {task:<14} {n:>3} items -> {Path(args.out) / f'p9_code_{task}.html'}")
    print(f"\nTotal {sum(made.values())} items. Open each file in a browser; "
          f"progress saves as you go.")


if __name__ == "__main__":
    main()
