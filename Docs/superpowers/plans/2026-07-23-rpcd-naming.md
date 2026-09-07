# TASK-435 RP&CD naming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Roleplay/Personas destination one public name — **"RP&CD"** (rail) / **"Roleplay & Chat Dictionaries"** (full) — reserving **"Personas"** for the in-screen user-identity mode, with all legacy routes still resolving.

**Architecture:** Pure copy/label change across the destination's user-visible name surfaces (shell destination label + full_label, tab display labels, screen header, two tab tooltips) plus one new `"roleplay"` route alias. No internal id/route/class renames.

**Tech Stack:** Python 3.11+, Textual, pytest.

## Global Constraints

- Destination display name: rail `"RP&CD"`, full `"Roleplay & Chat Dictionaries"`. The in-screen **mode** stays `"Personas"` (do NOT change `MODE_LABELS`).
- Internal `destination_id`/`primary_route`/`TAB_*` id constants and all file/class/module names stay `"personas"`/unchanged.
- `"personas"` must still resolve as a route and remain palette-searchable (it already does via `primary_route`/`destination_id` — do not add it to `legacy_routes`). Add only `"roleplay"` to `legacy_routes`.
- `Tests/` and `tests/` are byte-identical dupes — edit whichever the build treats as source and keep both consistent (verify with a diff after).

---

### Task 1: Rename the destination across all name surfaces + add the `roleplay` alias

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py` (the `personas` `ShellDestination`)
- Modify: `tldw_chatbook/Constants.py` (`TAB_DISPLAY_LABELS[TAB_CCP]`, `[TAB_PERSONAS]`)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (header `title`, ~line 584)
- Modify: `tldw_chatbook/app.py` (tab tooltips for `TAB_PERSONAS` ~:704 and `TAB_CCP` ~:712)
- Test: `Tests/UI/test_shell_destinations.py`, `Tests/UI/test_command_palette_shell_routes.py`, `Tests/UI/test_command_palette_providers.py` (update expectations)

**Interfaces:**
- Produces: `get_shell_destination("personas").label == "RP&CD"`, `.full_label == .accessible_label == "Roleplay & Chat Dictionaries"`, and `resolve_shell_route("roleplay").destination_id == "personas"`.

- [ ] **Step 1: Write/adjust the failing tests**

Update the destination + palette assertions to the new names, and add the `roleplay` alias + `"&"` render assertions. In `Tests/UI/test_shell_destinations.py` (the block asserting the `personas` destination, ~line 14):
```python
    dest = get_shell_destination("personas")
    assert dest.label == "RP&CD"
    assert dest.full_label == "Roleplay & Chat Dictionaries"
    assert dest.accessible_label == "Roleplay & Chat Dictionaries"
    assert "roleplay" in dest.legacy_routes
    for route in ("personas", "ccp", "conversations_characters_prompts", "characters", "roleplay"):
        assert resolve_shell_route(route).destination_id == "personas"
```
(Adapt to the file's existing style/imports — it already imports these helpers. If the existing assertion literally checks `"Personas"`, replace that literal.)

In `Tests/UI/test_command_palette_shell_routes.py` (~line 95), replace the `"Personas"` label assertion:
```python
    assert {"ccp", "conversations_characters_prompts", "characters", "roleplay"} <= alias_terms["personas"]
    assert "RP&CD" in alias_terms["personas"]
    assert "Roleplay & Chat Dictionaries" in alias_terms["personas"]
    assert "personas" in alias_terms["personas"]   # still searchable via id/primary_route
```

In `Tests/UI/test_command_palette_providers.py` (~line 356), change the expected label literal `"Personas"` → `"RP&CD"` (match whatever that assertion targets — the tab display label / command title).

Add a `TAB_DISPLAY_LABELS` assertion (in `test_shell_destinations.py` or the providers test):
```python
    from tldw_chatbook.Constants import get_tab_display_label, TAB_CCP, TAB_PERSONAS
    assert get_tab_display_label(TAB_CCP) == "RP&CD"
    assert get_tab_display_label(TAB_PERSONAS) == "RP&CD"
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `python -m pytest Tests/UI/test_shell_destinations.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py -q`
Expected: FAIL (assertions still see `"Personas"`, no `roleplay` alias).

- [ ] **Step 3: Rename the shell destination** (`shell_destinations.py`)

In the `personas` `ShellDestination`, change `label` `"Personas"` → `"RP&CD"`, add `full_label="Roleplay & Chat Dictionaries"`, and append `"roleplay"` to `legacy_routes`:
```python
    ShellDestination(
        "personas",
        "RP&CD",
        "personas",
        "Characters, personas, dictionaries, and behavior profiles.",
        "Manage behavior profiles and persona context.",
        ("ccp", "conversations_characters_prompts", "characters", "roleplay"),
        full_label="Roleplay & Chat Dictionaries",
    ),
```

- [ ] **Step 4: Rename the tab display labels** (`Constants.py`, `TAB_DISPLAY_LABELS`)

`TAB_CCP: "Personas"` → `TAB_CCP: "RP&CD"` and `TAB_PERSONAS: "Personas"` → `TAB_PERSONAS: "RP&CD"`.

- [ ] **Step 5: Rename the screen header** (`personas_screen.py`, ~line 584)

`title="Roleplay"` → `title="Roleplay & Chat Dictionaries"`.

- [ ] **Step 6: Rename the two tab tooltips** (`app.py`, ~:704 and ~:712)

- `TAB_PERSONAS`: `"Open Personas for characters, prompts, dictionaries, and behavior profiles"` → `"Open Roleplay & Chat Dictionaries for characters, personas, dictionaries, and behavior profiles"`.
- `TAB_CCP`: `"Switch to Personas for characters, personas, prompts, dictionaries, and world books"` → `"Switch to Roleplay & Chat Dictionaries for characters, personas, dictionaries, and world books"`.
(The retired "prompts" is dropped while editing these strings.)

- [ ] **Step 7: Run the tests to confirm they pass**

Run: `python -m pytest Tests/UI/test_shell_destinations.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py -q`
Expected: PASS.

- [ ] **Step 8: Grep for any remaining user-visible "Personas" destination string + confirm the mode/internal ones are intentionally kept**

Run:
```bash
grep -rn '"Personas"' tldw_chatbook/ | grep -vE 'MODE_LABELS|personas_state'
grep -rn "Switch to Personas\|Open Personas\|to Personas\b" tldw_chatbook/
```
Expected: the only remaining `"Personas"` literals are `MODE_LABELS["personas"] = "Personas"` (the mode — intentional) and internal identifiers/docstrings (not user-visible). No stray destination tooltip/label/help string left saying "Personas". If a NEW user-visible destination string surfaces, rename it too and note it.

- [ ] **Step 9: Regression — the mode chip stays "Personas" + broad label/nav suite**

Run:
```bash
python -m pytest Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_shell_destinations.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py Tests/UI/test_screen_navigation.py -q
```
Expected: PASS. In particular `test_destination_visual_parity_correction.py` must still expect the `"Personas"` **mode** button (it is unchanged) — if that test breaks, you changed the mode label by mistake; revert `MODE_LABELS`.

- [ ] **Step 10: Verify `Tests/` vs `tests/` consistency**

Run: `for f in test_shell_destinations test_command_palette_shell_routes test_command_palette_providers; do diff Tests/UI/$f.py tests/UI/$f.py >/dev/null 2>&1 && echo "$f: in sync" || echo "$f: DIFFERS — sync it"; done`
If they differ, apply the same edits to the other copy (or confirm only one is collected). Keep them consistent.

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/UI/Navigation/shell_destinations.py tldw_chatbook/Constants.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/app.py Tests/UI/ tests/UI/
git commit -m "feat(nav): rename destination to RP&CD / Roleplay & Chat Dictionaries (task-435)"
```

---

## Self-review notes

- **Spec coverage:** AC#1 (nav label, header, mode names mutually consistent, one meaning each) → Steps 3/4/5 (rail `RP&CD` + header full name; mode `Personas` untouched). AC#2 (legacy aliases route) → Step 3 keeps `primary_route="personas"` + ccp/…/characters and adds `roleplay`; Step 1 asserts all resolve. Both covered.
- **Placeholder scan:** the `~line NNN` references are approximate anchors; each edit names the exact literal string to change, so no ambiguity. The only judgement call is matching the existing test assertions' shape — the source files are named.
- **Type/name consistency:** `"RP&CD"` and `"Roleplay & Chat Dictionaries"` used identically across shell label/full_label, tab labels, header, and tests; `"roleplay"` added once to `legacy_routes`; `MODE_LABELS` and internal ids deliberately unchanged.
