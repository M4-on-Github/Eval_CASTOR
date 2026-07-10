# ADR-001: Statistical test selection for salvage-plan templating analysis

## Status
Accepted

## Date
2026-07-07

## Context

Pipeline 6 (`SPEC_salvage_analysis.md`) tests whether a VLM's free-text salvage
plan (`recovery_considerations`) is grounded in the specific image or templated
on the state label it assigned — e.g., does the model emit "call a fireboat"
every time it says `on_fire`, regardless of what's actually different about
that vessel/cargo/situation.

The data this analysis runs on has a specific, fixed shape that constrains
which tests are valid:

- **~110 images total**, split across 4-5 states (`aground`, `capsized`,
  `on_fire`, `sunken`, occasionally `good`) — roughly **20-30 records per
  state group**.
- Per record, a **binary presence/absence flag per salvage element**
  (fireboat, tug, crane, divers, ...) extracted by an LLM and normalized via
  embedding-cluster into canonical categories (open-vocabulary, not a fixed
  taxonomy — see spec Stage 1/2).
- Two categorical grouping variables: `predicted_state` (the model's own
  classification) and `gt_state` (human ground truth) — tested **separately**,
  since "does the plan template on the model's own (possibly wrong) belief"
  and "does the plan track reality" are different questions.
- A derived continuous variable, `typicality_score` (Jaccard overlap between a
  record's element set and the modal element set for its state) — one value
  per record, used to test overall templating strength rather than one
  element at a time.

This ADR records why specific tests were chosen over the more commonly
reached-for alternatives (chi-squared, one-way ANOVA, plain t-test), and why
a more rigorous multivariate option (PERMANOVA) was deliberately deferred
rather than built now.

## Decision

**Primary test — Fisher's exact test, per (element, state) pair, framed as
one-vs-rest, with Benjamini-Hochberg FDR correction applied separately
within each (state_source, state) pair -- e.g. `predicted/on_fire` and
`predicted/aground` each get their own independent correction, not pooled
together or with the GT-state tracks -- and odds ratio as the effect
size.**

**Secondary/omnibus test — Kruskal-Wallis on `typicality_score`, grouped by
state (run once for `predicted_state`, once for `gt_state`), with Dunn's test
as the pairwise post-hoc when Kruskal-Wallis is significant.**

**Deferred, not built in v1 — PERMANOVA** (permutation-based multivariate test
on the full element-presence vector via Jaccard distance), documented as a
future extension if the univariate approach proves too fragmented.

## Alternatives Considered

### For element-vs-state association: chi-squared test of independence

- **What it is**: The standard test for association between two categorical
  variables — here, element-present/absent × state (aground/capsized/
  on_fire/sunken/good), as one *k*-level contingency table per element.
- **Pros**: Familiar, directly generalizes to more than 2 groups in one test,
  produces one p-value per element instead of one per (element, state) pair.
- **Cons**: Chi-squared's p-value is an **asymptotic approximation** that is
  only valid when expected cell counts are reasonably large (the common rule
  of thumb: expected count ≥ 5 in at least 80% of cells, none below 1). With
  ~20-30 records per state and salvage elements that may appear in only a
  handful of records, many cells in a 5-state × 2-presence table will have
  expected counts under 5. Using chi-squared here would silently report
  p-values that don't mean what they claim to mean at this sample size.
  Chi-squared is also a global test — a significant result says "presence is
  *not independent of* state somewhere" but doesn't directly answer "is this
  specific state overrepresented for this element," which is exactly the
  fireboat/on_fire question being asked.
- **Rejected**: Wrong assumption regime for the actual per-state sample
  sizes, and doesn't map cleanly onto the one-vs-rest hypothesis of interest.
  **Fisher's exact test** computes the exact hypergeometric probability of
  the observed 2×2 table (or more extreme) with no large-sample assumption —
  it is the correct substitute precisely when chi-squared's asymptotics break
  down, which is the norm here, not the exception.

### Why one-vs-rest 2×2 instead of one *k*-way table per element

Even having chosen Fisher's exact, the table shape still had to be decided:
one *r*×2 table (state has 5 levels) per element, or five 2×2 tables (this
state vs. all others) per element.

- **This-state-vs-rest 2×2** was chosen because the underlying question is
  inherently one-vs-rest: "is fireboat overrepresented specifically when the
  model says `on_fire`," not "is fireboat's distribution across all five
  states different from uniform in some unspecified way." The 2×2 framing
  also yields a directly interpretable **odds ratio** per (element, state)
  pair, which an *r*×2 omnibus test doesn't give you without a follow-up
  step anyway. The cost is more tests (element × state × source, not just
  element × source), which is exactly why FDR correction (below) is
  non-negotiable, not optional polish.

### For multiple-comparisons control: Bonferroni or no correction

- **No correction**: Rejected outright — with an open-vocabulary element set
  (spec Stage 1/2 chose open vocabulary over a fixed taxonomy specifically
  so real, unanticipated patterns could surface), the number of
  (element × state × source) tests will commonly be in the dozens to
  low hundreds for a single run. At even a conservative 15 elements × 5
  states × 2 sources = 150 tests, an uncorrected α = .05 threshold gives an
  *expected* ~7-8 false positives even if there is truly no association
  anywhere. Reporting "significant" findings under those conditions is not
  defensible.
- **Bonferroni**: Controls family-wise error rate (probability of *any* false
  positive) by dividing α by the number of tests. This is the conservative
  choice, appropriate when a single false positive would be costly (e.g., a
  regulatory claim). Here the tests are exploratory — the goal is to
  *identify candidate* templating patterns worth a closer manual look, not to
  make a single binding claim. Bonferroni's stringency at 100+ tests would
  very likely suppress every finding at this sample size, including real
  ones, defeating the purpose of running an open-vocabulary scan in the
  first place.
- **Rejected in favor of Benjamini-Hochberg FDR correction**, which controls
  the *expected proportion* of false positives among findings called
  significant, not the probability of any false positive at all. This is the
  standard choice for exploratory multi-hypothesis scans (its original and
  most common application is exactly this kind of setting: many simultaneous
  categorical association tests, e.g. differential expression scans in
  genomics) — it stays well-calibrated as the number of tests grows without
  Bonferroni's near-total loss of power at this sample size.

### For overall/omnibus templating strength: one-way ANOVA

- **What it is**: Compares the mean of a continuous variable
  (`typicality_score`) across ≥2 groups (states), assuming the variable is
  normally distributed within each group with roughly equal variances.
- **Pros**: Well-known, has a standard post-hoc (Tukey HSD) for pairwise
  follow-up, slightly more statistical power than its non-parametric
  counterpart *when its assumptions actually hold*.
- **Cons**: `typicality_score` is a **bounded Jaccard similarity** (0 to 1),
  computed from small integer-sized element sets — it is not going to be
  well-approximated by a normal distribution, especially with only ~20-30
  samples per group to check that assumption against in the first place.
  Small-sample departures from normality are exactly where ANOVA's Type I
  error rate becomes unreliable.
- **Rejected**: The normality assumption is not just unverified but actively
  implausible for this variable's construction, and the sample per group is
  too small to trust a normality check (e.g. Shapiro-Wilk) as a gate either
  way. **Kruskal-Wallis** — the rank-based, distribution-free analogue of
  one-way ANOVA — was chosen because it makes no distributional assumption
  beyond the groups sharing a similar shape, at the cost of only a modest
  reduction in power when data happen to be normal (which, again, isn't
  expected to be the case here).

### For omnibus post-hoc: Tukey HSD

- Tukey HSD is the standard pairwise post-hoc for one-way ANOVA and assumes
  the same normality/equal-variance conditions ANOVA does.
- **Rejected** for the same reason ANOVA itself was: since Kruskal-Wallis
  (rank-based) was chosen as the omnibus test, its **matching** post-hoc is
  **Dunn's test** (pairwise rank-sum comparisons with a shared-rank variance
  correction) — using Tukey HSD after a Kruskal-Wallis omnibus would mix a
  distribution-free omnibus test with a post-hoc that reintroduces the
  normality assumption Kruskal-Wallis was chosen specifically to avoid.

### For a strict two-group comparison: Student's t-test

- Relevant if the question were narrowed to exactly two groups, e.g. "is
  `on_fire`'s typicality score different from everything else." A t-test
  compares two group means and, like ANOVA, assumes normality (and, in its
  standard form, equal variance).
- **Rejected as the general tool** for the same normality/small-*n* reasons
  as ANOVA above. Where a genuine two-group comparison is useful (e.g.
  drilling into one specific state pair after a significant Kruskal-Wallis
  result), **Dunn's test already produces that pairwise comparison directly**
  as part of the Kruskal-Wallis follow-up, so a separate t-test/Mann-Whitney
  step isn't needed in the current design. If a future need arises for a
  two-group comparison outside the Kruskal-Wallis/Dunn's flow, Mann-Whitney U
  (t-test's rank-based analogue) would be the consistent choice, not a
  parametric t-test, for the same reasons already stated.

### For the full element-presence profile: PERMANOVA (deferred, not rejected)

- **What it is**: A permutation-based multivariate test (common in
  ecology for community-composition analysis) that treats each record's
  full element-presence vector as a point in a dissimilarity space (e.g.
  Jaccard distance) and asks whether groups (states) differ in their overall
  distribution in that space — one omnibus answer for "is the *entire
  profile* of salvage elements different by state," rather than many
  separate per-element tests.
- **Why it's the more statistically complete answer**: unlike running many
  independent Fisher's exact tests, PERMANOVA doesn't fragment the question
  into per-element pieces and naturally accounts for elements co-occurring
  (e.g. "fireboat" and "containment boom" tend to appear together) rather
  than treating each as independent.
  This was actively discussed in spec design and was not dismissed as wrong,
  only postponed.
- **Deferred rather than built in v1** because: (1) it requires a
  permutation-testing implementation and a distance-matrix + pairwise
  post-hoc scheme not otherwise needed by this pipeline, meaningfully
  increasing Task 6/7's scope; (2) the per-element Fisher's + Kruskal-Wallis
  design already directly answers the concrete motivating question (the
  fireboat/on_fire example) with less machinery; (3) per the spec's Success
  Criteria, if the per-element results turn out too fragmented to give a
  clean "is this state templated overall" verdict, PERMANOVA is the
  documented next step — not abandoned, just sequenced after the simpler
  design is shown to be insufficient in practice.

## Consequences

- **Every reported association must carry its FDR-corrected p-value, not the
  raw one.** Reporting a raw Fisher's exact p-value as "significant" without
  the BH correction applied across the full test set for that run would
  reintroduce exactly the false-positive inflation this design exists to
  avoid — this is called out as a hard "never do" in
  `SPEC_salvage_analysis.md`'s Boundaries section, not merely a style
  preference.
- **Small-*n* honesty**: with ~20-30 records per state, it's expected and
  acceptable for many elements to show a suggestive odds ratio without
  reaching FDR-corrected significance. `report_<run_name>.txt` (Stage 4)
  reports odds ratios and raw p-values alongside corrected ones specifically
  so real trends remain visible even when the sample is too small to clear
  the corrected significance bar — "nothing significant" is a valid, honest
  outcome, not a pipeline failure.
- **Two test tracks (predicted-state vs. GT-state) are corrected separately,
  not combined into one BH-FDR pass.** (Updated 2026-07-10; originally this
  ADR combined them, reasoning that predicted_state and gt_state are
  correlated enough that splitting the correction would double-count the
  same underlying signal. That assumption turned out to be wrong in
  practice: checking real runs' actual predicted/gt agreement rate showed
  21-46% accuracy (excluding UNPARSEABLE), not the near-total overlap the
  combined-correction argument required. With disagreement that high, the
  two tracks are testing meaningfully different groupings of the same
  images, not the same partition twice — so correcting them together was
  needlessly cutting power on both (e.g. an element present in only 3-4
  images total has essentially no chance of surviving correction against a
  combined 300-600+ test panel, even in the best-case scenario) for a
  double-counting risk that isn't real at this model's actual accuracy.
  `apply_fdr_correction()` now partitions by `state_source` and runs BH
  independently within each partition. Each track's significance calls are
  independently FDR-controlled findings, not mutual confirmation — an
  element significant under `predicted` and not under `gt` (or vice versa)
  is a real, reportable distinction, not noise.)
- **Further split: each state within a track is also corrected
  independently, not just each track.** (Updated 2026-07-10, same session
  as the predicted/GT split above.) `apply_fdr_correction()` partitions by
  `(state_source, state)`, not just `state_source` — e.g. `predicted/on_fire`
  and `predicted/aground` each get their own BH pass. This is a stronger
  case than the predicted/GT split: different states within one source are
  *mutually exclusive* one-vs-rest groups (an image is `aground` XOR
  `on_fire`, never both), so there's essentially no risk of the same
  underlying signal being double-counted across states the way there was a
  real (if small, per the predicted/GT check above) risk between predicted
  and GT. "Is there a signature element for on_fire" and "...for aground"
  are separable questions, matching how the motivating example in this
  ADR's Context section was framed from the start (a claim about `on_fire`
  specifically, not about all four states as one shared claim).
  Known cost: this shrinks each correction batch further (down to
  roughly 15-40 tests per (source, state) pair instead of 150+ per source),
  which recovers more power but means FDR control is a weaker guarantee in
  practice at that scale — worth remembering when reading a "significant"
  result from a run with very few images in that particular state.
- **If Kruskal-Wallis comes back significant**, Dunn's test's pairwise
  outputs are the only place state-pair-specific conclusions should be drawn
  from for the typicality-score track — the omnibus Kruskal-Wallis p-value
  alone only says "some group differs from some other group," not which.
- **PERMANOVA remains an open, tracked extension**, not a closed decision —
  if the first real pipeline run's per-element results are too scattered to
  answer "is this state templated overall" cleanly (see
  `SPEC_salvage_analysis.md` Open Question 4), building it is the next
  concrete step, not a redesign.
