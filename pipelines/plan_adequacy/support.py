"""
Support steps: what the registry cannot express, and which of it is scaffolding.

NO_MATCH was 385 of 1,980 steps -- 19% of everything the v2 plans say. Reading
them, 74% fall into six recurring capabilities the registry has no vocabulary
for: its 47 tools are technical actions plus assessments, and a real salvage
plan is substantially operational scaffolding. "Establish a safety perimeter"
alone accounts for 83 steps.

Adding six effect-free tools to close the gap was considered and REJECTED:
those steps would become SPECIFIED_UNGRADED -- "fine" -- and lift EPL without
a single plan becoming one step more executable. Instead the executor gives a
step that is PURELY scaffolding its own verdict, SUPPORT, which is neither
graded nor counted: it does not stop a plan, does not count towards EPL, and
does not appear among a plan's errors. Everything else the extractor could not
map stays NO_MATCH and still stops the plan.

"Purely" is the load-bearing word. The six category patterns alone were
hand-audited on 40 corpus steps and about half of them also carried a real
salvage action (a tow, a hull inspection by divers, mustering personnel) or an
undecided technique ("determine whether to use tugs or cranes") -- the
patterns match anywhere in a step, and v2 steps pack several actions into one.
Skipping those would hide real errors. So a category match counts as support
only if the step ALSO names no substantive action (SUBSTANTIVE_RE) and the
extractor listed no secondary tool for it. With that guard, 97 of the 278
categorised steps become SUPPORT and 29 of 30 sampled were pure scaffolding;
the 181 it holds back are about two-thirds real content. The guard errs
towards holding back, which is the direction that flatters nothing.
"""

import re

#: Ordered, because a step can match more than one pattern and the first hit
#: wins; earlier entries are the more specific readings. Patterns were written
#: against the corpus and their coverage measured, not guessed.
NO_MATCH_CATEGORIES = (
    ("site_control",
     r"perimeter|exclusion zone|safety zone|secure the (scene|site|area)"
     r"|restrict access|cordon|keep (unauthorized|unauthorised)"),
    ("liaison",
     r"coordinat\w+ with|notify|liaise|contact (local|the) (authorit|port|coast)"
     r"|in (consultation|conjunction) with"),
    ("temporary_stabilisation",
     r"install temporary|temporary (mooring|ballast|bulkhead|barrier|pontoon)"
     r"|shore up|cribbing"),
    ("ongoing_monitoring",
     r"monitor (the )?(weather|sea state|vessel|conditions|stability)"
     r"|continuous(ly)? monitor|keep under observation"),
    ("logistics",
     r"mobili[sz]e|arrange for|procure|stage (the )?equipment"),
    ("documentation",
     r"document|report for (regulatory|insurance)|final inspection"
     r"|post-salvage inspection|record keeping"),
)

_NO_MATCH_RE = tuple((name, re.compile(pat, re.I)) for name, pat in NO_MATCH_CATEGORIES)

#: What a step gets when none of the six patterns fit. These are the honest
#: residual -- what these plans contain that neither the registry nor this
#: categorisation can express -- and their size is the number worth reporting.
UNCATEGORISED = "other"

#: A salvage action, or a deferred decision about one, anywhere in the step.
#: Built from the audit's misses rather than from the registry's tool names:
#: equipment NOUNS are deliberately absent ("mobilise tugs" is logistics, not
#: attach_tug), and inspection only counts when it is of the vessel, not of
#: the equipment. "determine / strategy / plan" catch the deferrals -- a step
#: that postpones choosing the technique is a planning failure, not scaffolding.
SUBSTANTIVE_RE = re.compile(
    r"\btow(s|ed|ing)?\b|inspect\w* (of )?(the )?(hull|vessel|wreck|damage|submerged)"
    r"|(final|post-\w+) inspection|survey|\bdivers?\b|\bright(ing|ed)?\b|refloat"
    r"|pump|dewater|foam|extinguish|firefight|fire suppression|\bboom|skim|spill"
    r"|\blift(ing|ed|s)?\b|\bcut(ting)?\b|patch|muster|accounted for|evacuat|rescue"
    r"|dredg|lighter|offload|beach gear|\bpull|cofferdam|co2|\bseal|\bcool"
    r"|calculat|determine|strategy|\bplan\b|remov",
    re.I)


def no_match_category(step_text: str, verdict: str = "NO_MATCH") -> str:
    """Which missing capability an unmapped step is reaching for.

    Labels both SUPPORT steps and the NO_MATCH steps the guard held back, so
    a genuine NO_MATCH still says what it was reaching for. Returns "" for
    any graded step, so the column stays empty rather than inviting the
    reading that a graded step was also "really" something else.
    """
    if verdict not in ("NO_MATCH", "SUPPORT"):
        return ""
    for name, rx in _NO_MATCH_RE:
        if rx.search(step_text or ""):
            return name
    return UNCATEGORISED


def is_support(step_text: str, secondary_tools=()) -> bool:
    """Is this unmapped step pure scaffolding, safe to leave ungraded?

    Only meaningful for a step the extractor mapped to no_match -- the
    executor calls it on nothing else.
    """
    if secondary_tools:
        return False
    if SUBSTANTIVE_RE.search(step_text or ""):
        return False
    return no_match_category(step_text) != UNCATEGORISED
