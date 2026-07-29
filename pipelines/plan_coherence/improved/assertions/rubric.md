# Assertion Rubric — CASTOR Improved Experiment
**Version 2**

An assertion is a sentence of domain guidance injected into the VLM prompt before plan generation.
The experiment uses three assertion conditions:

| Condition | Content |
|---|---|
| **STANDARD** | Specific, discriminative, operationally correct assertions |
| **CONTROL** | Vague, tautological, non-discriminative assertions (same count) |
| **ABLATION** | No assertions — base prompt only |

---

## STANDARD assertion criteria

A STANDARD assertion must satisfy **at least one** of the following:

1. **Technique + selection condition** — names a specific technique AND the condition that selects it.
   - ✓ "Hard grounding on rock or coral requires dredging to clear a channel before pulling."
   - ✗ "The seabed the vessel rests on may affect how easy it is to free."

2. **Safety-critical sequence** — names a specific guard action that must precede a specific trigger action.
   - ✓ "All fuel tanks must be checked before any cutting, welding, or movement begins."
   - ✗ "The vessel's condition may need to be assessed before any action is taken."

3. **Named resource + selection context** — names a specific resource TYPE and the condition under which it is chosen over alternatives.
   - ✓ "Harbor tugs for sheltered water; ocean-going salvage tugs for open water."
   - ✗ "Tugboats may be used for towing."

4. **Named crew role + specific task** — names a crew role and what that role specifically does.
   - ✓ "A salvage engineer or naval architect performs the stability check against GM and righting-arm criteria."
   - ✗ "Engineers may help plan the operation."

---

## CONTROL assertion criteria

A CONTROL assertion must fail **all four** STANDARD criteria AND satisfy **at least one** of:

1. **Tautological** — conclusion implied by the premise; no domain knowledge added.
   - ✓ "A larger vessel may be harder to move and may require more time and equipment."

2. **Non-discriminative** — equally true for any vessel, condition, scenario, or depth.
   - ✓ "The depth of the wreck may determine what recovery method is practical."

3. **Generic resource** — names a resource category without a selection condition.
   - ✓ "Cranes may be used for lifting." (vs. "floating crane for capsized/sunken")

4. **Vague scalar** — identifies a factor but names no mechanism, threshold, or technique.
   - ✓ "Weather and sea conditions may affect how long work can continue each day."

---

## Count rule

Each casualty block must have **exactly the same number of assertions** in STANDARD and CONTROL.
Current target: **8 assertions per casualty type** in both conditions.

Resources and crew lines follow the same rule: STANDARD names specific types with context;
CONTROL names generic categories only. Both have the same number of resource/crew lines.

---

## Resource → casualty type mapping (for MTH evaluation)

| Resource | Valid conditions |
|---|---|
| Harbor tug | aground (sheltered), capsized (post-righting tow) |
| Ocean-going salvage tug | aground (open water), on_fire (vessel tow) |
| Beach gear (anchor-wire-chain) | aground |
| Floating crane | capsized, sunken |
| Barge | aground (lightering), sunken (sectional removal) |
| Submersible pump | capsized (dewatering), sunken |
| Cofferdam / patching material | sunken |
| Containment boom / skimmer | sunken (surface oil) |
| Electro-acoustic / side-scan sonar | sunken (submerged oil, hull survey) |
| Fireboat with external monitors | on_fire (boundary cooling only) |
| Alcohol-resistant (AR) foam | on_fire (polar-solvent/water-soluble cargo) |
| CO₂ fixed system | on_fire (machinery space) |
| Dive and survey team | sunken, aground (hull survey), capsized (hull inspection) |
| Salvage engineer / naval architect | capsized (stability check), sunken (lift plan) |
| Firefighting team | on_fire |
| Rescue swimmers / boat crew | capsized (crew recovery) |
| Hazmat / ordnance specialist | all (when cargo unknown or military vessel) |
