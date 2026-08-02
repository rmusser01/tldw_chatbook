# TASK-1981 Speech and TTS Settings Ownership Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish one pure, testable contract that classifies every current built-in Speech settings control and safely bounds the configuration state, runtime state, navigation, and status values later Settings and Lab slices will share.

**Architecture:** Add a dependency-light contract module beside the existing Speech settings inventory. The module will use explicit built-in records rather than provider plugins or a form schema, validate those records against the current mounted-control inventory, and expose frozen enum/dataclass values for cross-screen state. This slice adds no consumer wiring, persistence, network work, or visible UI changes.

**Tech Stack:** Python 3.11+, frozen dataclasses, `StrEnum`, `datetime`, pytest.

**ADR required:** Yes.

**ADR path:** `backlog/decisions/039-global-and-studio-tts-settings-ownership.md`.

**Reason:** TASK-1981 directly implements ADR-039's accepted ownership, state, runtime-status, and bounded-navigation contracts. It makes no new architectural decision and therefore does not create another ADR.

**Deliberate boundary:** The current mixed-scope Lab editor remains mounted and functional. This task describes future ownership without moving a field, changing a save path, introducing Studio persistence, contacting a provider, or adding managed audio.cpp behavior.

---

### Task 1: Lock the built-in ownership inventory with failing tests

**Files:**

- Create: `Tests/UI/test_speech_settings_contracts.py`
- Create: `tldw_chatbook/UI/Speech/speech_settings_contracts.py`
- Read-only reference: `tldw_chatbook/UI/Speech/speech_settings_model.py`

**Step 1: Write the failing inventory contract tests**

Add focused tests that require:

- the exact seven built-in provider IDs plus the shared-default owner;
- every ID in `ALL_SETTINGS_CONTROLS` exactly once;
- rejection of missing, duplicate, unknown-control, and unknown-owner records;
- exact ADR-039 scope partitions for shared defaults, audio.cpp, OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk;
- documented reasons for the legacy mixed-scope Save and structural audio.cpp container entries that are intentionally retired by later replacement slices;
- distinct constants for `default-provider-select` and the future `configure-provider-select` control.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/task-1981-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_speech_settings_contracts.py -q
```

Expected: collection fails because the contract module does not exist.

**Step 2: Implement the minimal explicit manifest**

Create frozen ownership records with these bounded scopes: global configuration, Studio preference, Voice Profile operation, runtime operation/readout, and retired. List the current control IDs explicitly, retain their canonical provider/shared-default owner, and provide a validator that returns a read-only ID lookup only after completeness and uniqueness checks pass. Require a non-empty reason only for retired records.

Do not derive a generic provider schema, alter `PROVIDER_SETTINGS`, or mount the new contract in the live UI.

**Step 3: Run the inventory tests**

Run the command from Step 1.

Expected: all ownership tests pass.

### Task 2: Add bounded state, navigation, and safe status values test-first

**Files:**

- Modify: `Tests/UI/test_speech_settings_contracts.py`
- Modify: `tldw_chatbook/UI/Speech/speech_settings_contracts.py`

**Step 1: Write failing value-contract tests**

Cover:

- configuration enum values are exactly `Inherited`, `Default`, `Saved`, `Unsaved`, `Incomplete`, and `Invalid`;
- runtime enum values are exactly `Not checked`, `Checking`, `Ready`, `Stale`, `Unavailable`, and `Reconfiguring`;
- navigation accepts only an exact built-in provider ID and optional `configure`, `test`, `refresh-models`, or `refresh-voices` intent;
- navigation rejects shared defaults, aliases, malformed IDs, free-form intents, string subclasses, and unexpected payload fields;
- a frozen safe status snapshot carries only provider ID, saved configuration revision, optional runtime/catalog revisions, runtime state, aware observation time, freshness, bounded diagnostic category, and bounded recovery intent;
- revisions, timestamps, enums, and provider IDs reject malformed values; no raw URL, exception, secret, submitted-text, arbitrary diagnostic, or arbitrary recovery field exists.

Run the focused test command from Task 1.

Expected: the new assertions fail because the value types are absent.

**Step 2: Implement the smallest immutable DTOs**

Add `StrEnum` vocabularies and frozen, slotted dataclasses with strict boundary validation. Reuse the same canonical built-in provider set as the ownership inventory. Represent diagnostics and recovery as enums rather than free-form strings, and require timezone-aware observation timestamps.

Do not integrate the DTOs into `NavigateToScreen`, Settings, Lab status polling, the adapter registry, or persistence in this task.

**Step 3: Run focused and neighboring regression tests**

```bash
PYTHONPYCACHEPREFIX=/tmp/task-1981-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_speech_settings_contracts.py Tests/UI/test_speech_settings_model.py Tests/TTS/test_adapter_types.py -q
```

Expected: all focused and neighboring tests pass.

### Task 3: Verify the non-behavioral boundary and finish TASK-1981

**Files:**

- Modify: `backlog/tasks/task-1981 - Establish-Speech-and-TTS-settings-ownership-contracts.md`

**Step 1: Run static and diff checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/UI/Speech/speech_settings_contracts.py Tests/UI/test_speech_settings_contracts.py
git diff --check
```

Expected: both commands pass.

**Step 2: Audit scope**

Confirm the diff contains only the pure contract, its tests, this plan, and TASK-1981 metadata. Verify no configuration writer, UI compose path, network collaborator, adapter route, character store, or managed audio.cpp path changed.

**Step 3: Complete task documentation**

Check all acceptance criteria, add concise implementation notes including the ADR-039 conformance statement and exact verification commands, then set TASK-1981 to Done only after every Definition-of-Done gate applicable to this contract slice passes.

**Step 4: Commit**

```bash
git add Docs/superpowers/plans/2026-07-31-task-1981-speech-tts-settings-ownership-contracts.md "backlog/tasks/task-1981 - Establish-Speech-and-TTS-settings-ownership-contracts.md" tldw_chatbook/UI/Speech/speech_settings_contracts.py Tests/UI/test_speech_settings_contracts.py
git commit -m "feat(tts): establish settings ownership contracts"
```
