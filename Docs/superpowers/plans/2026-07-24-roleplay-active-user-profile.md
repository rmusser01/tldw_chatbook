# Roleplay task-442 — Active User Profile + Name Substitution + the "persona"→"user profile" Rename — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user mark one user profile as "who I am", have `{{user}}` render that name everywhere with their messages labeled by it — and rename the user-side "persona" concept to "user profile" end-to-end (labels + internals) with full persistence compatibility.

**Architecture:** A config-pointer (`[character_defaults] active_user_profile`) + a never-raising resolver feed the three call sites that hardcode `"User"`. The rename is a mechanical five-axis sweep (mode id, kind, DTO aliases, messages/widget, `persona_label`) with accept-old-write-new at the one serialization round-trip and explicit boundaries (workbench family, DOM ids, state-dict keys, the tldw_api wire mirror all STAY).

**Tech Stack:** Python ≥3.11, Textual, file-backed profile store (JSON), TOML config.

## Global Constraints

- **PLACEHOLDER TABLE (BINDING — the brainstorm correction):** `{{user}}` / `{{random_user}}` / `<USER>` → **the USER's name** (active user profile's name; fallback `"User"`). `{{char}}` / `{{character}}` / `{{persona}}` / `<CHAR>` → **the AI character's name**. **User-side tokens NEVER receive the character's name.**
- Active pointer = config single value (`[character_defaults] active_user_profile`); the P3b `enabled`/`is_active` per-record flag is a DIFFERENT concept — untouched.
- **Accept old, write new** on the `persona_label` serialization round-trip (`chat_screen.py:9456` `asdict` / `:9472` `ConsoleSessionSettings(**values)`) and on any persisted key found to say "persona" (P1e/P2 rule: guard EVERY read of persisted content). The profile store's on-disk file path and its top-level JSON keys (`profiles`/`exemplars`/… — verified persona-free) stay byte-identical.
- No active profile ⇒ all USER-SIDE output byte-identical (pinned). The new `{{character}}`/`{{persona}}` aliases substitute the character name REGARDLESS of the pointer (deliberate change for previously-literal texts).
- **Rename boundary (spec B2):** STAYS as-is — `PersonasScreen`/`personas_screen.py`/`Persona_Widgets/` directory/`PersonasPreviewPane`/`personas_pane_messages.py` module names (workbench axis), DOM ids/CSS `#personas-*`, saved-state dict keys `personas_workbench`/`personas_preview`, and the `tldw_api/character_persona_schemas.py` wire-mirror class names (app-side aliases only).
- NO DB migration. FOREGROUND tests only; NO background/broad sweeps; NEVER pkill. Implementers PREPEND `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && ` to EVERY Bash call; stage ONLY task files (never `git add -A`, never `.superpowers/`).
- CONCURRENT-SESSION HAZARD: the personas files are hot — re-verify the three substitution sites + rename surfaces AT EACH TASK DISPATCH (the B2-sweep lesson: survey drift mid-cycle).
- Tests/UI asyncio rules (don't mix Tests/UI with other dirs in one invocation OR add explicit `@pytest.mark.asyncio`). **Test env prefix:** `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ... -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`

---

## File Structure

- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` — **modify**: alias rows in `replace_placeholders` (Task 1).
- `tldw_chatbook/Character_Chat/active_user_profile.py` — **create**: resolver + set/clear (Task 1).
- Rename sweep (Task 2): `personas_screen.py` (mode/kind literals, labels), `personas_pane_messages.py` (messages), `Persona_Widgets/persona_profile_editor_widget.py` → `user_profile_editor_widget.py` (file+class), `local_character_persona_service.py` + `server_character_persona_service.py` + `character_persona_scope_service.py` (persona-half naming), `console_session_settings.py` + `chat_screen.py` + `chat_shell_bar.py` + `console_display_state.py` (`persona_label` axis), `tldw_api/character_persona_schemas.py` (aliases only), Settings copy, tests.
- `personas_screen.py` + inspector/library panes — **modify**: marking UX (Task 3).
- `personas_preview_controller.py`, `chat_screen.py`, `chat_events.py` — **modify**: substitution + labeling (Task 4).
- Task 5: pins + task-file bookkeeping.

---

## Task 1: Placeholder aliases + active-profile resolver

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` (the `replacements` dict in `replace_placeholders`, ~:404)
- Create: `tldw_chatbook/Character_Chat/active_user_profile.py`
- Test: `Tests/Character_Chat/test_active_user_profile.py` (create), `Tests/Character_Chat/test_placeholder_aliases.py` (create)

**Interfaces:**
- Produces:
  - `replace_placeholders` additionally maps `{{character}}` and `{{persona}}` → char name.
  - `resolve_active_user_profile_name(service) -> str | None`
  - `set_active_user_profile(name: str) -> bool` / `clear_active_user_profile() -> bool`
  - `get_active_user_profile_pointer() -> str | None`
  - Config location: section `"character_defaults"`, key `"active_user_profile"`.

- [ ] **Step 1: Write the failing tests**

`Tests/Character_Chat/test_placeholder_aliases.py`:
```python
from tldw_chatbook.Character_Chat.Character_Chat_Lib import replace_placeholders


def test_new_character_side_aliases():
    out = replace_placeholders("Hi {{user}}, I am {{char}}/{{character}}/{{persona}}.", "Ada", "Sam")
    assert out == "Hi Sam, I am Ada/Ada/Ada."


def test_user_side_tokens_never_get_character_name():
    # THE brainstorm correction: user-side tokens carry the USER's name only.
    out = replace_placeholders("{{user}} {{random_user}} <USER>", "Ada", "Sam")
    assert out == "Sam Sam Sam"
    assert "Ada" not in out


def test_token_free_text_byte_identical():
    assert replace_placeholders("plain text", "Ada", "Sam") == "plain text"


def test_defaults_unchanged():
    out = replace_placeholders("{{user}} meets {{persona}}", None, None)
    assert out == "User meets Character"
```

`Tests/Character_Chat/test_active_user_profile.py`:
```python
import pytest
from tldw_chatbook.Character_Chat.active_user_profile import (
    resolve_active_user_profile_name,
    set_active_user_profile,
    clear_active_user_profile,
    get_active_user_profile_pointer,
)


class _FakeService:
    def __init__(self, profiles):
        self._profiles = profiles

    def list_user_profiles(self, active_only: bool = False):
        return list(self._profiles)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch):
    """Route the config read/write seam at an in-memory dict."""
    store = {}
    import tldw_chatbook.Character_Chat.active_user_profile as mod
    monkeypatch.setattr(mod, "get_cli_setting", lambda section, key, default=None: store.get((section, key), default))
    def _save(section, key, value):
        store[(section, key)] = value
        return True
    monkeypatch.setattr(mod, "save_setting_to_cli_config", _save)
    return store


def test_unset_pointer_resolves_none():
    assert resolve_active_user_profile_name(_FakeService([{"name": "Sam"}])) is None


def test_set_then_resolve(_isolated_config):
    assert set_active_user_profile("Sam") is True
    svc = _FakeService([{"name": "Sam"}, {"name": "Kai"}])
    assert resolve_active_user_profile_name(svc) == "Sam"


def test_dangling_pointer_resolves_none(_isolated_config):
    set_active_user_profile("Ghost")
    assert resolve_active_user_profile_name(_FakeService([{"name": "Sam"}])) is None


def test_clear(_isolated_config):
    set_active_user_profile("Sam")
    assert clear_active_user_profile() is True
    assert get_active_user_profile_pointer() is None
    assert resolve_active_user_profile_name(_FakeService([{"name": "Sam"}])) is None


def test_resolver_never_raises_on_broken_service(_isolated_config):
    set_active_user_profile("Sam")
    class _Boom:
        def list_user_profiles(self, active_only: bool = False):
            raise RuntimeError("store unreadable")
    assert resolve_active_user_profile_name(_Boom()) is None
```
NOTE: the fake's method is `list_user_profiles` — Task 2 renames the real service method to that name. Until Task 2 lands, the resolver must call the CURRENT name with a fallback (`getattr(service, "list_user_profiles", None) or getattr(service, "list_persona_profiles")`) so Task 1 works standalone; Task 2 removes the fallback.

- [ ] **Step 2: Run to verify RED**

```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_placeholder_aliases.py Tests/Character_Chat/test_active_user_profile.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: alias tests FAIL (`{{character}}`/`{{persona}}` render literally); resolver tests FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement**

In `replace_placeholders`'s `replacements` dict add (keeping every existing row):
```python
        "{{character}}": char_name_actual,   # task-442 alias: the AI character's name
        "{{persona}}": char_name_actual,     # task-442 alias: the AI character's name (NEVER the user)
```
Create `tldw_chatbook/Character_Chat/active_user_profile.py`:
```python
"""The single "who am I" pointer for chats (task-442).

The user marks ONE user profile as their identity; its name feeds the
``{{user}}`` placeholder and message labels. Stored as a single config value
(single-active by construction, persists across sessions). "Persona" never
refers to the user in this app; the user-side concept is "user profile".
"""
from __future__ import annotations

from loguru import logger

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config

_SECTION = "character_defaults"
_KEY = "active_user_profile"


def get_active_user_profile_pointer() -> str | None:
    """Return the configured active-profile name, or None when unset."""
    try:
        value = get_cli_setting(_SECTION, _KEY, None)
    except Exception:
        return None
    text = str(value).strip() if value is not None else ""
    return text or None


def set_active_user_profile(name: str) -> bool:
    """Point the active user profile at ``name``. Returns write success."""
    try:
        return bool(save_setting_to_cli_config(_SECTION, _KEY, str(name)))
    except Exception:
        logger.opt(exception=True).warning("Could not persist the active user profile.")
        return False


def clear_active_user_profile() -> bool:
    """Clear the pointer (no active user profile)."""
    try:
        return bool(save_setting_to_cli_config(_SECTION, _KEY, ""))
    except Exception:
        logger.opt(exception=True).warning("Could not clear the active user profile.")
        return False


def resolve_active_user_profile_name(service) -> str | None:
    """Resolve the pointer to a live profile name, or None.

    Unset pointer, dangling pointer (profile deleted/renamed), or ANY
    service failure -> None (treated as no-active; never raises). Cheap:
    one config read + one profile-list read.

    Args:
        service: the user-profile service (exposes ``list_user_profiles``;
            the pre-rename ``list_persona_profiles`` is accepted until the
            Task-2 rename lands).

    Returns:
        The active profile's name, or None.
    """
    pointer = get_active_user_profile_pointer()
    if pointer is None or service is None:
        return None
    try:
        lister = getattr(service, "list_user_profiles", None) or getattr(
            service, "list_persona_profiles", None
        )
        if lister is None:
            return None
        for record in lister() or []:
            if isinstance(record, dict) and str(record.get("name") or "") == pointer:
                return pointer
    except Exception:
        logger.opt(exception=True).debug("Active user profile resolution failed.")
        return None
    return None
```

- [ ] **Step 4: Run to verify GREEN** (same command; expect 9 passed). Also `... python -c "import tldw_chatbook.app"`.

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/Character_Chat/active_user_profile.py Tests/Character_Chat/test_placeholder_aliases.py Tests/Character_Chat/test_active_user_profile.py
git commit -m "feat(chat): task-442 T1 — character-side placeholder aliases + active user profile resolver"
```

---

## Task 2: The rename — "persona" never means the user (Part B)

**Files (modify unless noted):** `personas_screen.py`; `Widgets/Persona_Widgets/personas_pane_messages.py`; `Widgets/Persona_Widgets/persona_profile_editor_widget.py` → **rename file** `user_profile_editor_widget.py`; `Character_Chat/local_character_persona_service.py`; `Character_Chat/server_character_persona_service.py`; `Character_Chat/character_persona_scope_service.py`; `Chat/console_session_settings.py`; `UI/Screens/chat_screen.py`; `Widgets/Chat_Widgets/chat_shell_bar.py`; `Chat/console_display_state.py`; `tldw_api/character_persona_schemas.py` (aliases only) + `tldw_api/__init__.py`; `UI/CCP_Modules/ccp_persona_handler.py`; Settings copy site(s); every test file pinning the old names.

**Interfaces:**
- Consumes: Task 1's resolver fallback (removes it once `list_user_profiles` is real).
- Produces (later tasks rely on): kind literal `"user_profile"`; mode id `"user_profiles"`; `ConsoleSessionSettings.user_profile_label`; service method `list_user_profiles()`; widget `UserProfileEditorWidget`; messages `EditUserProfileRequested`/`UserProfileSaveRequested`; app-side DTO names `UserProfileCreate/Update/Response`.

**The rename map (apply EXACTLY; grep each old literal before and after):**

| Axis | Old → New | Notes |
|---|---|---|
| Mode id | `"personas"` → `"user_profiles"` (~40 literals in personas_screen + tests) | Saved-state compat FREE by construction: `restore_state` (~:767) only restores `active_mode == "characters"` — add the pin test below. `_apply_mode`/chip ids: DOM ids `#personas-*` STAY. |
| Labels | "Personas" mode chip label, `_MODE_DESCRIPTORS` entry, Settings personas category title/description, inspector `Type:` copy, any "persona" user-facing strings for the user side | "User Profiles" / "user profile" (P0/444 plain style) |
| Kind | `"persona_profile"` → `"user_profile"` (~20 literals) | incl. 443's action matrix constants + 440's `_provider_send_block_reason` kind check |
| DTOs | app-side imports of `PersonaProfileCreate/Update/Response` → `UserProfileCreate/Update/Response` | In `tldw_api/character_persona_schemas.py` ADD alias lines only: `UserProfileCreate = PersonaProfileCreate` (etc.) with a comment: the wire-mirror class names match the server contract — the ONE sanctioned internal "persona" remnant. Switch importers: `ccp_persona_handler`, `personas_screen`, `local_character_persona_service`, `server_character_persona_service`, the editor widget, `tldw_api/__init__` re-exports, tests. |
| Messages/widget | `EditPersonaRequested`→`EditUserProfileRequested`; `PersonaProfileSaveRequested`→`UserProfileSaveRequested`; `PersonaProfileEditorWidget`→`UserProfileEditorWidget` (+ `git mv persona_profile_editor_widget.py user_profile_editor_widget.py`) | Update all posters/handlers/imports/tests. |
| `persona_label` | `ConsoleSessionSettings.persona_label` → `user_profile_label` (~23 refs: `console_session_settings.py:165/:703-707`, `chat_screen.py:7961/:9456/:9472/:13828`, `chat_shell_bar.py:19/:74`, `console_display_state.py:292/:333/:344`) + display copy `f"Persona: {label}"` → `f"As: {label}"` (all twins) | Default `"General"` KEPT. **Compat shim at the deserialize edge** (below). |
| Service naming | `persona_store_path`→`user_profile_store_path`; `list/create/update/delete_persona_profile*`→`..._user_profile*`; `_persona_profile_view`→`_user_profile_view`; `_persist_persona_profiles`→`_persist_user_profiles`; internal attrs `_persona_profiles`→`_user_profiles` etc.; same halves in the server service + scope service | The on-disk JSON's top-level keys (`profiles`, `exemplars`, …) and the store FILE PATH value are byte-identical — rename Python identifiers only. Class names: rename only if they name the user-side concept alone; `LocalCharacterPersonaService`/`CharacterPersonaScopeService` serve BOTH characters and profiles → KEEP class names, rename their profile-half members (record the judgment in the task notes). |

- [ ] **Step 1: Write the failing compat + pin tests FIRST**

Add to `Tests/UI/test_chat_screen_state.py` (or the file that tests the :9456/:9472 round-trip — READ it first and extend in place):
```python
def test_console_session_settings_accepts_pre_rename_persona_label_key():
    """task-442 accept-old-write-new: a pre-rename serialized settings dict
    (persona_label) must deserialize into user_profile_label."""
    from dataclasses import asdict
    old_blob = asdict(ConsoleSessionSettings())          # start from a real shape
    old_blob["persona_label"] = old_blob.pop("user_profile_label")
    restored = <the deserialize seam at chat_screen.py:9472's helper>(old_blob)
    assert restored.user_profile_label == "General"
    assert asdict(restored).get("persona_label") is None  # writers emit only the new key
```
(Bind `<the deserialize seam>` to the REAL helper name once read — the plan cannot know it; it is the function wrapping `ConsoleSessionSettings(**values)`.) And the mode-id pin in `Tests/UI/test_personas_workbench_state.py` (or the save/restore test file):
```python
def test_pre_rename_personas_mode_blob_restores_to_default():
    """Old saved state with active_mode='personas' was ALREADY discarded by
    restore_state (only 'characters' restores) — the rename needs no shim."""
```
…asserting a restore with `{"personas_workbench": {"active_mode": "personas"}}` yields the fresh default state.

- [ ] **Step 2: Apply the map axis-by-axis, committing per axis is NOT required — one commit, but run the gate after each axis locally.** The `persona_label` deserialize shim (at the `:9472` seam):
```python
        values = dict(values)
        if "persona_label" in values and "user_profile_label" not in values:
            # Pre-task-442 blobs serialized the old field name.
            values["user_profile_label"] = values.pop("persona_label")
        values.pop("persona_label", None)
        return ConsoleSessionSettings(**{k: v for k, v in values.items() if k in _FIELD_NAMES})
```
(Match the seam's existing unknown-key handling — READ it; if it already filters by field names, only add the key-mapping lines.) Remove Task 1's `list_persona_profiles` fallback in the resolver.

- [ ] **Step 3: Grep-gates (must ALL pass; run exactly these)**
```
# zero app-side old DTO imports outside the mirror:
grep -rn "PersonaProfileCreate\|PersonaProfileUpdate\|PersonaProfileResponse" tldw_chatbook/ --include="*.py" | grep -v "tldw_api/"
# zero old kind/mode/message/widget/label identifiers anywhere in tldw_chatbook/:
grep -rn '"persona_profile"\|EditPersonaRequested\|PersonaProfileSaveRequested\|PersonaProfileEditorWidget\|persona_label\|list_persona_profiles\|persona_store_path' tldw_chatbook/ --include="*.py"
# mode literal: remaining '"personas"' hits must be ONLY the workbench-namespace allowlist (state-dict keys, DOM ids in strings) — list and justify each:
grep -rn '"personas"' tldw_chatbook/ --include="*.py"
# user-facing copy: no user-side "Persona" strings (allowlist: {{persona}} token handling, tldw_api mirror, workbench identifiers):
grep -rni "persona" tldw_chatbook/ --include="*.py" | grep -viE "tldw_api/|{{persona}}|PersonasScreen|personas_screen|Persona_Widgets|PersonasPreview|personas_pane_messages|personas_workbench|personas_preview|#personas-|personas-" 
```
Every surviving hit must be individually justified in the task notes (expected: near-zero).

- [ ] **Step 4: Full verification gate (FOREGROUND, in this order)**
```
Tests/UI/test_personas_workbench.py  Tests/UI/test_personas_preview.py Tests/UI/test_personas_preview_restore.py Tests/UI/test_personas_inspector_pane.py Tests/UI/test_persona_profile_widgets.py Tests/UI/test_personas_editor_save_in_place.py Tests/UI/test_chat_screen_state.py   (one Tests/UI invocation)
Tests/Character_Chat/test_local_character_persona_service.py Tests/Character_Chat/test_persona_personality_traits_roundtrip.py Tests/Character_Chat/test_active_user_profile.py   (one invocation)
Tests/tldw_api/test_character_persona_client.py   (one invocation)
python -c "import tldw_chatbook.app"
```
Expected: all pass (test files' own old-name pins updated as part of the sweep — update assertions, do not weaken behavior checks).

- [ ] **Step 5: Commit**
```bash
git add -u tldw_chatbook/ Tests/    # ONLY after git status shows exclusively intended files; never .superpowers/
git commit -m "refactor(personas): task-442 T2 — 'persona' never means the user (user-profile rename + persistence compat)"
```

---

## Task 3: Marking UX — "Set as my name"

**Files:** `personas_screen.py` (+ inspector pane, library pane), `Widgets/Persona_Widgets/personas_pane_messages.py` if a message is needed. Test: `Tests/UI/test_personas_workbench.py` (extend).

**Interfaces:**
- Consumes: `set_active_user_profile`/`clear_active_user_profile`/`get_active_user_profile_pointer` (T1); kind `"user_profile"` (T2); 443's `_apply_action_state` matrix; 440's `set_console_actions_enabled` flow.
- Produces: inspector action button (id `#personas-set-my-name`) for `user_profile` selections toggling Set/Clear; active-row indicator; inspector summary line "Chatting as: X".

- [ ] **Step 1: Failing tests** (extend `Tests/UI/test_personas_workbench.py`, mirroring its selection harness):
```python
async def test_set_as_my_name_sets_pointer_and_indicates(...):
    # select a user profile -> click #personas-set-my-name -> pointer == its name,
    # inspector shows "Chatting as: <name>", the library row carries the active marker,
    # button label flips to Clear.

async def test_clear_active_profile(...):
    # with an active profile: click again -> pointer None, indicator gone.

async def test_delete_active_profile_clears_pointer(...):
    # delete the active profile -> get_active_user_profile_pointer() is None.

async def test_set_my_name_absent_for_characters(...):
    # kind-aware: the button does not render for character selections.
```
(Complete assertions written against the real harness — the implementer binds fixture names; RED by definition, new UI.)

- [ ] **Step 2: Implement** — add the button to the inspector's action block gated to kind `"user_profile"` in 443's matrix; handler calls T1's set/clear + refreshes: inspector summary line, library row marker (e.g. `● ` prefix or a `-active` row class via the pane's row-build path), button label. Delete flow: after a successful profile delete, `if get_active_user_profile_pointer() == deleted_name: clear_active_user_profile()`. All config writes via T1's helpers (never raw config calls).

- [ ] **Step 3: Verify** — the workbench file (one invocation) + `-k "readiness or Console or gate"` still green (440 machinery) + import.

- [ ] **Step 4: Commit** `feat(personas): task-442 T3 — Set-as-my-name marking UX (pointer + indicators)`

---

## Task 4: Substitution + labeling at the three sites

**Files:** `UI/Persona_Modules/personas_preview_controller.py` (~:113), `UI/Screens/chat_screen.py` (~:10271), `Event_Handlers/Chat_Events/chat_events.py` (~:4374). Tests: `Tests/UI/test_personas_preview.py` + `Tests/UI/test_console_character_avatar.py`-adjacent Start-Chat tests + `Tests/Event_Handlers/Chat_Events/test_chat_events.py` (extend each).

**Interfaces:**
- Consumes: `resolve_active_user_profile_name` (T1) + the profile service handle each site already has (or can reach — preview controller/screen own one; chat_events resolves via the app's service accessor — READ how each site reaches the service; do NOT construct new service instances).
- Produces: a tiny shared helper on each surface, e.g. `def _active_user_name(...) -> str: return resolve_active_user_profile_name(svc) or "User"` — used at the site.

- [ ] **Step 1: Failing tests** — at each site, with an active profile "Sam":
```python
# preview: seeding a greeting containing {{user}} renders "Sam"; set_speakers called with user="Sam"
# Start-Chat: the handoff greeting containing {{user}} renders "Sam"; session user_profile_label == "Sam"
# chat_events: the character-chat display path substitutes "Sam" (extend the existing ccl.replace_placeholders test)
```
Each with a no-active twin asserting `"User"` (RED for the active case; the twin is the AC3 pin used again in T5).

- [ ] **Step 2: Implement** — replace the literal `"User"` argument at each site with the resolved name (fallback `"User"`); preview additionally `pane.set_speakers(user=name)` only when a profile is active (untouched otherwise); Start-Chat sets `user_profile_label=name` on the session settings it builds when a profile is active. Verify at this step whether Console transcript user-rows have a nameable speaker slot — if none exists, add NOTHING (record in notes).

- [ ] **Step 3: Verify** — the three test files (correct rootdir groupings) + import.
- [ ] **Step 4: Commit** `feat(personas): task-442 T4 — {{user}} renders the active profile name at all three send surfaces`

---

## Task 5: Byte-compat pins + bookkeeping

**Files:** tests only + `backlog/archive/tasks/task-442 - Active-persona-concept-with-user-name-substitution-in-chats.md` (archived as Done; the live-tree duplicate stub was removed by task-544's backlog dedup).

- [ ] **Step 1:** Pin tests (some may exist from T4 twins — consolidate, don't duplicate): no active profile ⇒ preview seed, Start-Chat greeting, chat_events output ALL byte-identical to pre-feature strings; `set_speakers` not called with a user override; `user_profile_label` stays `"General"`. Plus the alias pin: `{{persona}}`/`{{character}}` substitute the character name with NO active profile set.
- [ ] **Step 2:** Task file → Done, ACs checked, Implementation Plan + Notes (the rename map applied, the boundary judgments, the compat shims, the placeholder table restated).
- [ ] **Step 3:** Full gate: the Task-2 verification suite + T4's files + import. Commit: `test(personas): task-442 T5 — byte-compat pins + task bookkeeping`

---

## Self-Review (author)

- **Spec coverage:** A1 resolver (T1) ✓; A2 marking (T3) ✓; A3 substitution (T4) ✓; A4 labeling (T4) ✓; A5 pins (T4 twins + T5) ✓; B1 map (T2 table, all seven axes incl. display copy + file rename) ✓; B2 boundary (Global Constraints + T2 allowlists) ✓; B3 compat (T2 Step 1 tests + shim; mode-id compat-free pin) ✓; spec review-pass items (wire-mirror aliases, `"As: {label}"` copy, alias behavior-change pin) all present ✓.
- **Type consistency:** `resolve_active_user_profile_name(service) -> str | None` consistent T1→T4; kind `"user_profile"`, mode `"user_profiles"`, `user_profile_label`, `list_user_profiles` consistent T2→T3/T4; config section/key consistent T1→T3.
- **Known unbindables (explicitly delegated, not placeholders):** the deserialize-seam helper name at `chat_screen.py:9472` and the T3/T4 test fixture names must be bound by the implementer after fresh reads — each is called out at its step with the reason (concurrent-session drift makes hard-coding them brittle).
- **Placeholder scan:** clean otherwise; every code step shows code or an exact mapping table + gate commands.
