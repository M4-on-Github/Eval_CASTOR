"""
WorldState: the knowledge/effects tracker the executor walks a plan through.

Two clocks run inside one object, and the distinction is the whole point
(see salvage_simulation.md's knowledge-vs-truth split, reused here for plan
text instead of a simulated environment):

  knowledge  -- facts an assessment-family tool call has ESTABLISHED
                (e.g. "substrate" after survey_seabed). Sequence checks and
                gate resolution (gates.py) read only from here.
  effects    -- facts an action-family tool call has made TRUE in the world
                (e.g. "vessel_pulled" after pull()). Route/goal completion
                (methods.py) reads from here.

A precondition check (missing_requires) never distinguishes the two --
requires may name either kind of fact -- but keeping them as separate sets
internally is what makes it possible to ask "did the plan ever actually
establish X" (the CONDITIONAL_UNRESOLVED question) as a question distinct
from "is X true" (the completeness question).
"""

from dataclasses import dataclass, field


#: test_atmosphere's establishes depends on its `test_type` param, which a
#: static per-tool `establishes` list (registry/tools.json) can't express --
#: see that entry's `source` note. Special-cased here rather than in the
#: registry: gas_oxygen_tested requires gas_explosive_tested already known,
#: and gas_toxic_tested requires gas_oxygen_tested, enforcing [SH 1-2.6.2]'s
#: explosive -> oxygen -> toxic order across repeated calls to the SAME
#: tool. atmosphere_safe only becomes known once all three sub-facts are.
_ATMOSPHERE_TEST_ORDER = {
    "explosive": ("gas_explosive_tested", None),
    "oxygen": ("gas_oxygen_tested", "gas_explosive_tested"),
    "toxic": ("gas_toxic_tested", "gas_oxygen_tested"),
}
_ATMOSPHERE_SUB_FACTS = frozenset(f for f, _ in _ATMOSPHERE_TEST_ORDER.values())


class WorldState:
    """Mutable, walked forward one ToolCall at a time in plan order.

    Usage (see executor.py):
        ws = WorldState()
        for call in calls:
            missing = ws.missing_requires(call, registry)
            ...  # missing -> SEQUENCE_VIOLATION for this call
            ws.apply(call, registry)
    """

    def __init__(self):
        self._known = set()    # facts established (assessment results)
        self._true = set()     # facts made true (action effects)

    def knows(self, fact: str) -> bool:
        """Has the plan (so far) established this fact via an assessment
        step? This is what gates.py:resolve_conditional and the sequencing
        pass (executor.py pass 3) read."""
        return fact in self._known

    def is_true(self, fact: str) -> bool:
        """Has the plan (so far) made this fact true via an action's
        effects? This is what route-completion / terminal-goal checks read."""
        return fact in self._true

    def known_facts(self) -> frozenset:
        return frozenset(self._known)

    def true_facts(self) -> frozenset:
        return frozenset(self._true)

    def missing_requires(self, call, registry) -> set:
        """Facts `call.tool` requires that are not yet known/true. Non-empty
        -> this call is a SEQUENCE_VIOLATION. Checks both knowledge and
        effects, since a `requires` entry may name either (e.g. `pull`
        requires "freeing_force" (knowledge) and "tank_contents"
        (knowledge), while `right_vessel` requires "crew_rescued", an
        effect)."""
        if not registry.has(call.tool):
            return set()
        spec = registry.spec(call.tool)
        satisfied = self._known | self._true
        missing = set(spec.requires) - satisfied

        if call.tool == "test_atmosphere":
            test_type = call.params.get("test_type")
            _, predecessor = _ATMOSPHERE_TEST_ORDER.get(test_type, (None, None))
            if predecessor is not None and predecessor not in satisfied:
                missing.add(predecessor)

        return missing

    def apply(self, call, registry) -> None:
        """Advance world state by one call: its `establishes` become known,
        its `effects` become true. No-op for a tool the registry doesn't
        recognise (e.g. "no_match") -- the executor still records the call
        for verdict purposes, it just contributes no state change."""
        if not registry.has(call.tool):
            return
        spec = registry.spec(call.tool)

        if call.tool == "test_atmosphere":
            # spec.establishes lists all four facts this tool CAN eventually
            # establish (for the registry-integrity test's reachability
            # check -- see tools.json's test_atmosphere entry) but a SINGLE
            # call must only establish the one fact matching its test_type,
            # so the generic bulk-apply below is skipped entirely here.
            fact, _ = _ATMOSPHERE_TEST_ORDER.get(call.params.get("test_type"), (None, None))
            if fact is not None:
                self._known.add(fact)
            if _ATMOSPHERE_SUB_FACTS <= self._known:
                self._known.add("atmosphere_safe")
            self._true |= set(spec.effects)
            return

        self._known |= set(spec.establishes)
        self._true |= set(spec.effects)
