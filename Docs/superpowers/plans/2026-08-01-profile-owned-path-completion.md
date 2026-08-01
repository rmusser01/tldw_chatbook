# TASK-865 Profile-Owned Path Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish TASK-865 by retiring rejected plaintext persistence, routing every swept hardcoded profile-owned path through the canonical accessor, and enforcing the resulting ownership census in source and installed artifacts.

**Architecture:** ADR-040 divides remaining occurrences into effective-config state, active-user data, shared artifacts, inert defaults, compatibility seeds/constants, and a read-only legacy probe. Production call boundaries delegate to `get_cli_config_path()` or `get_user_data_dir()`; shared TTS assets and tokenizer artifacts stay shared. A repository scanner enforces exact, count-sensitive exceptions, while ADR-029 private atomic writes protect newly profile-aware Kokoro state.

**Tech Stack:** Python 3.11+, Textual, pytest, stdlib `ast`/`tokenize`, JSON, `pathlib`, existing `Utils.private_paths` primitives, setuptools wheel/sdist verification.

## Global Constraints

- Do not modify Notes Sync or migrate existing `_get_effective_config_path()` consumers.
- Do not create a reduced, surrogate, or synthetic Textual application. Test an actual production function/method or the full `TldwCli` application.
- Do not copy, move, import, rename, or delete existing user data files. Source-file deletion is limited to the rejected unreachable implementations named by ADR-040.
- Keep TTS model weights, packaged voices, reusable Chatterbox/Higgs profiles, and tokenizer artifacts shared.
- Keep the UI Kokoro JSON and backend `voice_blends/voice_blends.json` formats separate.
- Keep user-selected import/export/workspace/model paths unchanged.
- Resolve profile-owned defaults at the call or construction boundary; do not add import-time profile paths.
- Serialize private state completely before invoking `atomic_private_write_text()`.
- Preserve explicit Kokoro backend directory configuration; only the default follows the effective config parent.
- A `TTSBackendManager` belongs to one application/config session. A second profile uses a second manager.
- Tests must use scratch paths and must never touch the developer's real `~/.config/tldw_cli` or `~/.local/share/tldw_cli` trees.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/040-profile-owned-state-and-shared-asset-paths.md`

Reason: ADR-040 is accepted and defines the storage ownership, migration, compatibility, privacy, and cross-module contracts implemented by this plan. No additional ADR is required unless implementation changes one of those ownership classes.

## File Map

**Create**

- `scripts/check_profile_owned_path_inventory.py` — executable Python-source scanner and exact exception reconciliation.
- `Tests/Architecture/test_profile_owned_path_inventory.py` — scanner fixtures and final production-census gate.
- `Tests/Architecture/test_retired_profile_path_owners.py` — deletion/import/reference contract for rejected modules and symbols.
- `tldw_chatbook/TTS/voice_blend_paths.py` — canonical Kokoro preset paths and private JSON write boundary.
- `Tests/TTS/test_voice_blend_paths.py` — two-profile, private-mode, and atomic-failure coverage.
- `Tests/TTS/test_kokoro_backend_profile_paths.py` — backend default/explicit path and manager-session coverage.
- `Tests/UI/test_dictation_export_path.py` — actual Improved Dictation export-function coverage.
- `Tests/Prompt_Management/test_profile_owned_prompt_paths.py` — prompt default-root function coverage.
- `Tests/UI/test_profile_owned_settings_paths.py` — Settings and Code Repository path-copy function coverage.
- `Tests/Utils/test_embedding_cache_path.py` — active-user runtime cache resolution coverage.

**Modify**

- `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py`
- `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- `tldw_chatbook/UI/CodeRepoCopyPasteWindow.py`
- `tldw_chatbook/UI/Screens/settings_screen.py`
- `tldw_chatbook/UI/STTS_Window.py`
- `tldw_chatbook/UI/Speech/speech_catalog_mixin.py`
- `tldw_chatbook/UI/Speech/speech_settings_mixin.py`
- `tldw_chatbook/TTS/backends/kokoro.py`
- `tldw_chatbook/UI/Dictation_Window_Improved.py`
- `tldw_chatbook/Utils/local_stt_providers.py`
- `tldw_chatbook/Utils/paths.py`
- `tldw_chatbook/Utils/Utils.py`
- `Tests/Chat/test_chat_functions.py`
- `Tests/Local_Ingestion/test_dictation_window_provider_ids.py`
- `Tests/UI/test_disabled_action_recovery_tooltips.py`
- `Tests/UI/test_file_picker_filters_callable.py`
- `Tests/DB/test_private_sqlite_inventory.py`
- `Tests/Packaging/test_installed_distribution.py`
- `backlog/docs/sqlite-private-owner-inventory.md`
- `Docs/security/production-diagnostic-inventory.json`
- `Docs/Development/TTS/TTS-Dictation-Implementation-Complete.md`
- `Docs/Development/TTS/TTS-Improve-1.md`
- `Docs/Development/TTS/Speech-Recording-1.md`
- `CHANGELOG.md`
- `backlog/tasks/task-865 - Sweep-hardcoded-~-.config-tldw_cli-and-~-.local-share-tldw_cli-call-sites-onto-the-real-accessors.md`
- `Docs/superpowers/specs/2026-08-01-profile-owned-path-completion-design.md`
- This plan, as its checkboxes and verification evidence are completed.

**Delete**

- `tldw_chatbook/Audio/transcription_history.py`
- `tldw_chatbook/Widgets/transcription_history_viewer.py`
- `tldw_chatbook/UI/Dictation_Window.py`

---

### Task 1: Retire Rejected Plaintext Persistence and Obsolete DB Path Creation

**Files:**

- Create: `Tests/Architecture/test_retired_profile_path_owners.py`
- Delete: `tldw_chatbook/Audio/transcription_history.py`
- Delete: `tldw_chatbook/Widgets/transcription_history_viewer.py`
- Delete: `tldw_chatbook/UI/Dictation_Window.py`
- Modify: `tldw_chatbook/Utils/paths.py`
- Modify: `tldw_chatbook/Utils/Utils.py`
- Modify: `tldw_chatbook/Utils/local_stt_providers.py`
- Modify: `Tests/Local_Ingestion/test_dictation_window_provider_ids.py`
- Modify: `Tests/UI/test_disabled_action_recovery_tooltips.py`
- Modify: `Tests/UI/test_file_picker_filters_callable.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Modify: `Docs/Development/TTS/TTS-Dictation-Implementation-Complete.md`
- Modify: `Docs/Development/TTS/TTS-Improve-1.md`
- Modify: `Docs/Development/TTS/Speech-Recording-1.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: TASK-1331's decision that transcription history is not a product feature.
- Produces: no importable `Dictation_Window`, `transcription_history`, `transcription_history_viewer`, `get_user_database_path`, `USER_DB_DIR`, or `USER_DB_PATH`; retained `ImprovedDictationWindow` provider normalization.

- [ ] **Step 1: Add a failing retirement contract**

Create the architecture test with exact source and symbol assertions:

```python
from __future__ import annotations

import ast
from pathlib import Path

import tldw_chatbook
from tldw_chatbook.app import TldwCli


REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_FILES = (
    "tldw_chatbook/Audio/transcription_history.py",
    "tldw_chatbook/Widgets/transcription_history_viewer.py",
    "tldw_chatbook/UI/Dictation_Window.py",
)
RETIRED_NAMES = {
    "get_user_database_path",
    "USER_DB_DIR",
    "USER_DB_PATH",
}


def test_rejected_history_modules_are_absent_and_production_app_imports() -> None:
    assert all(not (REPO_ROOT / relative).exists() for relative in RETIRED_FILES)
    assert issubclass(TldwCli, object)
    assert Path(tldw_chatbook.__file__).resolve().is_relative_to(REPO_ROOT)


def test_legacy_user_database_symbols_have_no_production_binding() -> None:
    offenders: list[tuple[str, str]] = []
    for path in sorted((REPO_ROOT / "tldw_chatbook").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in RETIRED_NAMES:
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), node.id))
    assert offenders == []
```

- [ ] **Step 2: Run the retirement contract and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/Architecture/test_retired_profile_path_owners.py -q
```

Expected: FAIL because the three rejected modules and legacy DB symbols still exist.

- [ ] **Step 3: Delete the rejected production implementations and legacy DB helper**

Delete the three files exactly. In `Utils/paths.py`, remove the imported/fallback `USER_DB_DIR` and `USER_DB_PATH` names, the `__main__` diagnostic references, and `get_user_database_path()`. In `Utils/Utils.py`, remove `USER_DB_DIR`, `USER_DB_FILENAME`, and `USER_DB_PATH`. Do not add compatibility shims that reconstruct the retired path.

- [ ] **Step 4: Preserve only Improved Dictation provider coverage**

In `Tests/Local_Ingestion/test_dictation_window_provider_ids.py`:

- remove `test_dictation_window_provider_select_ids_are_real`;
- remove the legacy-window source parsing from `test_no_dropdown_offers_the_bare_misspelled_lightning_whisper_id`;
- change `test_load_settings_normalizes_legacy_provider_id` to import and drive only `Dictation_Window_Improved`;
- retain `_get_provider_options`, `_load_settings`, `_initialize_service`, and `Select` validation coverage for the production window.

The normalized settings assertion becomes:

```python
import tldw_chatbook.UI.Dictation_Window_Improved as dwi

monkeypatch.setattr(dwi, "get_cli_setting", _fake_get_cli_setting_legacy_provider)
save_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
monkeypatch.setattr(
    dwi,
    "save_setting_to_cli_config",
    lambda *args, **kwargs: save_calls.append((args, kwargs)),
)
settings = dwi.ImprovedDictationWindow.__new__(
    dwi.ImprovedDictationWindow
)._load_settings()
assert settings["provider"] == "lightning-whisper-mlx"
assert save_calls == []
```

- [ ] **Step 5: Remove exact test and source-census collateral**

Remove the history imports and `test_transcription_history_disabled_actions_explain_selection_requirement` from `Tests/UI/test_disabled_action_recovery_tooltips.py`. Remove only `tldw_chatbook/Widgets/transcription_history_viewer.py` from the parametrized list in `Tests/UI/test_file_picker_filters_callable.py`. Update the `LEGACY_PROVIDER_IDS` comment in `Utils/local_stt_providers.py` to describe persisted values from the retired and retained dictation implementations without presenting the deleted module as live.

- [ ] **Step 6: Reconcile the curated SQLite owner inventory**

Remove this tuple from `EXPECTED_PARENT_CREATORS` in `Tests/DB/test_private_sqlite_inventory.py`:

```python
(
    "tldw_chatbook/Utils/paths",
    "get_user_database_path",
    "USER_DB_DIR.mkdir(parents=True, exist_ok=True)",
),
```

Keep row P05 in `backlog/docs/sqlite-private-owner-inventory.md`, but change its state to `migrated`, disposition to `remove_obsolete_creation`, and rationale to state that TASK-865 removed the unreachable creator without touching any database.

- [ ] **Step 7: Regenerate and inspect the diagnostic inventory**

Run:

```bash
.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --write
git diff -- Docs/security/production-diagnostic-inventory.json
```

Accept only owner/call records attributable to the three deleted modules and the touched production sources. If another file changes, stop and reconcile that source before staging the generated inventory.

- [ ] **Step 8: Correct current documentation and release notes**

Mark the history/store/viewer and legacy `Dictation_Window.py` descriptions as retired or superseded in the three TTS documents. Preserve historical Backlog tasks unchanged. Add an Unreleased/Removed changelog entry naming the three unreachable modules and an Unreleased/Changed entry noting that unsupported direct imports of `get_user_database_path`, `USER_DB_DIR`, and `USER_DB_PATH` were removed.

- [ ] **Step 9: Run focused verification**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/Architecture/test_retired_profile_path_owners.py \
  Tests/Local_Ingestion/test_dictation_window_provider_ids.py \
  Tests/UI/test_disabled_action_recovery_tooltips.py \
  Tests/UI/test_file_picker_filters_callable.py \
  Tests/DB/test_private_sqlite_inventory.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py -q
.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
git diff --check
```

Expected: all selected tests pass; the diagnostic checker reports the new reviewed counts.

- [ ] **Step 10: Commit the privacy retirement**

```bash
git add tldw_chatbook Tests backlog/docs Docs/security Docs/Development/TTS CHANGELOG.md
git commit -m "refactor(privacy): retire rejected dictation history"
```

---

### Task 2: Build the Exact Executable-Path Scanner

**Files:**

- Create: `scripts/check_profile_owned_path_inventory.py`
- Create: `Tests/Architecture/test_profile_owned_path_inventory.py`

**Interfaces:**

- Consumes: Python source text and ADR-040's five retained exception kinds.
- Produces: `Occurrence`, `ExceptionRule`, `scan_source()`, `scan_tree()`, `reconcile_inventory()`, and a CLI returning nonzero for new/count-changed/stale occurrences.

- [ ] **Step 1: Write scanner fixtures before the scanner**

Define tests against these exact interfaces:

```python
from scripts.check_profile_owned_path_inventory import (
    Disposition,
    ExceptionRule,
    Occurrence,
    reconcile_inventory,
    scan_source,
)


def test_scanner_reports_embedded_and_multiline_physical_lines() -> None:
    source = '''COPY = "edit ~/.config/tldw_cli/config.toml now"
DEFAULTS = """[database]
media = "~/.local/share/tldw_cli/media.db"
"""
'''
    found = scan_source(source, "tldw_chatbook/example.py")
    assert [(item.line, item.expression) for item in found] == [
        (1, "literal:~/.config/tldw_cli/config.toml"),
        (3, "literal:~/.local/share/tldw_cli/media.db"),
    ]


def test_scanner_detects_direct_indirect_and_join_function_roots() -> None:
    source = '''
direct = Path.home() / ".config" / "tldw_cli" / "models"
indirect = base / ".config" / "tldw_cli" / "themes"
data = os.path.join(home, ".local", "share", "tldw_cli", "cache")
'''
    found = scan_source(source, "tldw_chatbook/example.py")
    assert [item.expression for item in found] == [
        "join:.config/tldw_cli",
        "join:.config/tldw_cli",
        "join:.local/share/tldw_cli",
    ]


def test_scanner_ignores_comments_and_actual_docstrings() -> None:
    source = '''# ~/.config/tldw_cli/comment
def sample():
    """~/.local/share/tldw_cli/docstring"""
    return "safe"
'''
    assert scan_source(source, "tldw_chatbook/example.py") == ()


def test_reconcile_rejects_duplicates_new_counts_and_stale_rules() -> None:
    occurrence = Occurrence(
        "tldw_chatbook/example.py",
        4,
        "module:MODEL_DIR",
        "literal:~/.config/tldw_cli/models",
    )
    rule = ExceptionRule(
        "tldw_chatbook/example.py",
        "module:MODEL_DIR",
        "literal:~/.config/tldw_cli/models",
        1,
        Disposition.SHARED_ARTIFACT,
        "reusable model weights",
    )
    assert reconcile_inventory((occurrence,), (rule,)) == ()
    assert reconcile_inventory((occurrence, occurrence), (rule,))
    assert reconcile_inventory((), (rule,))
    assert reconcile_inventory((occurrence,), ())
```

- [ ] **Step 2: Run scanner fixtures and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/Architecture/test_profile_owned_path_inventory.py -q
```

Expected: collection fails because `scripts.check_profile_owned_path_inventory` does not exist.

- [ ] **Step 3: Implement typed scanner records and dispositions**

Create these public records in the script:

```python
class Disposition(StrEnum):
    PERSISTED_DEFAULT = "persisted_default"
    RESOLVER_SEED = "resolver_seed"
    COMPATIBILITY_CONSTANT = "compatibility_constant"
    SHARED_ARTIFACT = "shared_artifact"
    READ_ONLY_LEGACY_PROBE = "read_only_legacy_probe"


@dataclass(frozen=True, order=True)
class Occurrence:
    relative_path: str
    line: int
    context: str
    expression: str


@dataclass(frozen=True)
class ExceptionRule:
    relative_path: str
    context: str
    expression: str
    expected_count: int
    disposition: Disposition
    reason: str
```

Use `tokenize.generate_tokens()` for executable string tokens and `ast.parse()` for docstring spans, qualified ownership contexts, and path-join expressions. A context is the smallest enclosing function/class or the module assignment target, for example `module:BASE_DATA_DIR_CLI` versus `function:_default_base_data_dir`; this is required because those two `config.py` joins have different ADR-040 dispositions. Literal shapes are the root plus its path-like suffix, for example `literal:~/.local/share/tldw_cli/media.db`. Join shapes are exactly `join:.config/tldw_cli` or `join:.local/share/tldw_cli`.

- [ ] **Step 4: Implement physical-line and path-join detection**

The implementation must:

```python
ROOTS = ("~/.config/tldw_cli", "~/.local/share/tldw_cli")
JOIN_SUFFIXES = (
    (".config", "tldw_cli"),
    (".local", "share", "tldw_cli"),
)


def _physical_line(token_text: str, token_line: int, offset: int) -> int:
    return token_line + token_text[:offset].count("\n")


def _literal_expression(value: str, root: str, offset: int) -> str:
    match = re.match(re.escape(root) + r"[A-Za-z0-9_./<>-]*", value[offset:])
    assert match is not None
    return f"literal:{match.group(0)}"
```

Flatten `/`, `os.path.join(...)`, and `.joinpath(...)` expressions into string components, detect either suffix regardless of the base expression, and deduplicate nested `BinOp` nodes by `(relative_path, line, column, context, expression)`.

- [ ] **Step 5: Implement exact reconciliation and CLI output**

`reconcile_inventory()` groups occurrences by `(relative_path, context, expression)`, rejects a missing rule, rejects a count mismatch, rejects duplicate rules, rejects an empty reason, and rejects a stale rule. `main()` scans `tldw_chatbook/**/*.py`, prints every error as `path:line: context: expression: reason`, and exits 1 on any error. Add `--print-occurrences` to print the complete sorted census without changing files.

- [ ] **Step 6: Run scanner unit verification**

Run:

```bash
.venv/bin/python -m pytest Tests/Architecture/test_profile_owned_path_inventory.py -q
.venv/bin/python scripts/check_profile_owned_path_inventory.py --print-occurrences
```

Expected: scanner fixtures pass; the report lists current production occurrences for classification by later tasks.

- [ ] **Step 7: Commit the scanner foundation**

```bash
git add scripts/check_profile_owned_path_inventory.py Tests/Architecture/test_profile_owned_path_inventory.py
git commit -m "test(architecture): inventory profile-owned path literals"
```

---

### Task 3: Route Swept Config-State Readers and Display Copy

**Files:**

- Modify: `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py`
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- Modify: `tldw_chatbook/UI/CodeRepoCopyPasteWindow.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/Chat/test_chat_functions.py`
- Create: `Tests/Prompt_Management/test_profile_owned_prompt_paths.py`
- Create: `Tests/UI/test_profile_owned_settings_paths.py`

**Interfaces:**

- Consumes: `get_cli_config_path() -> Path`.
- Produces: `_default_dictionary_import_directory() -> Path`, `_default_prompt_import_directory() -> Path`, `_github_config_guidance_path() -> Path`, `_theme_save_target() -> Path`, and `_internal_prompts_save_target() -> Path`.

- [ ] **Step 1: Write two-profile function tests**

Use parameterized scratch profile paths and call the actual helpers directly:

```python
@pytest.mark.parametrize("profile_name", ["alpha", "beta"])
def test_config_children_follow_effective_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_name: str,
) -> None:
    config_path = tmp_path / profile_name / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert _default_dictionary_import_directory() == config_path.parent
    assert _default_prompt_import_directory() == config_path.parent / "prompts"
    assert _github_config_guidance_path() == config_path
    assert _theme_save_target() == config_path.parent / "themes"
    assert _internal_prompts_save_target() == config_path
```

Add a no-override case that monkeypatches `tldw_chatbook.config.DEFAULT_CONFIG_PATH` to a scratch `default/config.toml`, deletes `TLDW_CONFIG_PATH`, and asserts the same helpers retain their historical default children. This proves the override repair does not relocate default-profile state.

Extend the existing dictionary test to inspect `mock_validate_path.call_args.args[1]` and assert it equals the selected config parent. For prompt import, monkeypatch `is_initialized()` to true and `validate_path()` to capture the actual default base while driving `import_prompts_from_files()`.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/Chat/test_chat_functions.py::TestChatDictionary::test_parse_user_dict_markdown_file \
  Tests/Prompt_Management/test_profile_owned_prompt_paths.py \
  Tests/UI/test_profile_owned_settings_paths.py -q
```

Expected: FAIL because the helpers do not exist and the current functions still use global literals.

- [ ] **Step 3: Add lazy config-child helpers at their owning modules**

Implement the helpers without module-level path constants:

```python
def _default_dictionary_import_directory() -> Path:
    return get_cli_config_path().parent


def _default_prompt_import_directory() -> Path:
    return get_cli_config_path().parent / "prompts"


def _github_config_guidance_path() -> Path:
    return get_cli_config_path()


def _theme_save_target() -> Path:
    return get_cli_config_path().parent / "themes"


def _internal_prompts_save_target() -> Path:
    return get_cli_config_path()
```

Use the dictionary/prompt helpers only when `base_directory is None`; explicit caller paths remain unchanged.

- [ ] **Step 4: Replace all dynamic Settings path copy**

Replace the two module-level raw Theme strings in `_INSPECTOR_GUIDANCE` with path-neutral wording. In the call-time `_inspector_guidance_rows()` Theme branch, `_category_ownership_records()`, and `_render_detail_pane()`, format `_theme_save_target()` or `_internal_prompts_save_target()` so the active profile is displayed. Do not create either directory merely to display it.

The rendered values are:

```python
f"{_theme_save_target()}{os.sep}"
f"{_internal_prompts_save_target()}  [internal_prompts]"
```

- [ ] **Step 5: Verify Code Repository guidance through the actual method**

Add an async function test without mounting an `App`: use a `MagicMock(spec=CodeRepoCopyPasteWindow)` receiver with a `Mock` `notify`, monkeypatch `tldw_chatbook.config.get_cli_setting` to return no token, await the actual unbound `CodeRepoCopyPasteWindow.configure_token(window, object())`, and assert the notification contains the selected scratch `config.toml` path.

- [ ] **Step 6: Run focused path-isolation tests**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/Chat/test_chat_functions.py \
  Tests/Prompt_Management/test_profile_owned_prompt_paths.py \
  Tests/UI/test_profile_owned_settings_paths.py \
  Tests/UI/test_code_repo_copy_paste_window.py -q
.venv/bin/python scripts/check_profile_owned_path_inventory.py --print-occurrences
git diff --check
```

Expected: tests pass and the dictionary, prompt, Code Repository, and Settings literal occurrences disappear from the scanner report.

- [ ] **Step 7: Commit config-state reader/display ownership**

```bash
git add tldw_chatbook/Character_Chat tldw_chatbook/Prompt_Management tldw_chatbook/UI Tests/Chat Tests/Prompt_Management Tests/UI
git commit -m "fix(config): resolve profile-owned path copy lazily"
```

---

### Task 4: Centralize and Privately Persist UI Kokoro Blends

**Files:**

- Create: `tldw_chatbook/TTS/voice_blend_paths.py`
- Create: `Tests/TTS/test_voice_blend_paths.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify: `tldw_chatbook/UI/Speech/speech_catalog_mixin.py`
- Modify: `tldw_chatbook/UI/Speech/speech_settings_mixin.py`

**Interfaces:**

- Consumes: `get_cli_config_path()`, `application_owned_config_directory()`, and `atomic_private_write_text()`.
- Produces: `kokoro_ui_blend_file() -> Path`, `write_private_json(path, payload, application_owned_directory=None) -> PrivatePathResult`, and `write_kokoro_ui_blends(payload) -> PrivatePathResult`.

- [ ] **Step 1: Write path, mode, and failure-preservation tests**

```python
def test_ui_blend_path_retargets_after_module_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "config.toml"
    second = tmp_path / "second" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(first))
    assert kokoro_ui_blend_file() == first.parent / "kokoro_voice_blends.json"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(second))
    assert kokoro_ui_blend_file() == second.parent / "kokoro_voice_blends.json"


def test_ui_blend_write_is_private_and_atomic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "profile" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    result = write_kokoro_ui_blends({"warm": {"voices": [["af_bella", 1.0]]}})
    assert result.lexical_path == kokoro_ui_blend_file()
    assert json.loads(result.lexical_path.read_text()) == {
        "warm": {"voices": [["af_bella", 1.0]]}
    }
    if os.name == "posix":
        assert stat.S_IMODE(result.lexical_path.stat().st_mode) == 0o600


def test_serialization_failure_preserves_existing_ui_blends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "profile" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    target = kokoro_ui_blend_file()
    target.write_text('{"existing": true}\n', encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setattr(voice_blend_paths.json, "dumps", Mock(side_effect=TypeError))
    with pytest.raises(TypeError):
        write_kokoro_ui_blends({"broken": object()})
    assert target.read_text(encoding="utf-8") == '{"existing": true}\n'
```

Add a no-fallback test that sets `HOME` to `tmp_path / "home"`, seeds only `<scratch-home>/.config/tldw_cli/kokoro_voice_blends.json`, selects a different scratch `TLDW_CONFIG_PATH`, and asserts `kokoro_ui_blend_file()` and the actual blend-choice readers do not read, copy, modify, or delete the decoy.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/TTS/test_voice_blend_paths.py -q
```

Expected: import fails because `voice_blend_paths.py` does not exist.

- [ ] **Step 3: Implement the shared UI path and private JSON writer**

```python
def kokoro_ui_blend_file() -> Path:
    return get_cli_config_path().parent / "kokoro_voice_blends.json"


def write_private_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    application_owned_directory: Path | None = None,
) -> PrivatePathResult:
    serialized = json.dumps(payload, indent=2) + "\n"
    return atomic_private_write_text(
        path,
        serialized,
        application_owned_directory=application_owned_directory,
    )


def write_kokoro_ui_blends(payload: Mapping[str, Any]) -> PrivatePathResult:
    config_path = get_cli_config_path()
    return write_private_json(
        config_path.parent / "kokoro_voice_blends.json",
        payload,
        application_owned_directory=application_owned_config_directory(config_path),
    )
```

- [ ] **Step 4: Route every UI reader and writer through the helper**

Replace both STTS readers, the Speech catalog reader, and all five Speech settings read/export/import/create sites with `kokoro_ui_blend_file()`. Replace the two plain `open(..., "w")`/`json.dump` writes in `_handle_import_file()` and `_show_add_voice_blend_dialog()` with `write_kokoro_ui_blends()`. Keep user-selected export-file writes unchanged because those are explicit exports, not private application state.

- [ ] **Step 5: Prove actual production readers follow two profiles**

Create blend JSON under two scratch profile parents, retarget `TLDW_CONFIG_PATH`, and call the actual static `_kokoro_blend_choices()` methods from `STTSWindow` and `SpeechCatalogMixin`. Assert each call returns only the blend in the selected profile. Do not mount a Textual application.

- [ ] **Step 6: Run UI speech and private-path verification**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_voice_blend_paths.py \
  Tests/UI/test_stts_settings_widget.py \
  Tests/UI/test_speech_settings_pane.py \
  Tests/UI/test_speech_settings_save_equivalence.py \
  Tests/Utils/test_private_paths.py -q
.venv/bin/python scripts/check_profile_owned_path_inventory.py --print-occurrences
git diff --check
```

Expected: tests pass; only shared Chatterbox/Higgs asset paths remain in these UI modules' scanner output.

- [ ] **Step 7: Commit UI Kokoro ownership**

```bash
git add tldw_chatbook/TTS/voice_blend_paths.py tldw_chatbook/UI/STTS_Window.py tldw_chatbook/UI/Speech Tests/TTS Tests/UI
git commit -m "fix(tts): isolate Kokoro UI blends by profile"
```

---

### Task 5: Make the Kokoro Backend Default Profile-Owned and Atomic

**Files:**

- Modify: `tldw_chatbook/TTS/voice_blend_paths.py`
- Modify: `tldw_chatbook/TTS/backends/kokoro.py`
- Create: `Tests/TTS/test_kokoro_backend_profile_paths.py`

**Interfaces:**

- Consumes: `write_private_json()` from Task 4 and the existing `TTSBackendManager`/`KokoroTTSBackend` production classes.
- Produces: `default_kokoro_backend_blend_directory() -> Path`, `_save_blends() -> bool`, rollback-safe `save_voice_blend()` and `delete_voice_blend()`.

- [ ] **Step 1: Write backend default/explicit/session tests**

Add tests that:

1. construct two actual `TTSBackendManager` instances under two `TLDW_CONFIG_PATH` values;
2. monkeypatch only `KokoroTTSBackend.initialize` to an async no-op so model weights are not loaded;
3. call each manager's actual `get_backend("local_kokoro_onnx")`;
4. assert each returned actual `KokoroTTSBackend.voice_blends_dir` equals its own config parent plus `kokoro_voice_blends`;
5. construct `KokoroTTSBackend(config={"KOKORO_VOICE_BLENDS_DIR": explicit})` and assert the explicit directory is unchanged.

The manager assertion is:

```python
assert isinstance(first_backend, KokoroTTSBackend)
assert isinstance(second_backend, KokoroTTSBackend)
assert first_backend is not second_backend
assert first_backend.voice_blends_dir == first_config.parent / "kokoro_voice_blends"
assert second_backend.voice_blends_dir == second_config.parent / "kokoro_voice_blends"
```

- [ ] **Step 2: Write atomic failure and in-memory rollback tests**

Seed an existing `voice_blends.json`, set `backend.saved_blends`, monkeypatch `write_private_json` to raise `PrivatePathError`, and assert:

```python
assert backend.save_voice_blend("new", [("af_bella", 1.0)]) is False
assert "new" not in backend.saved_blends
assert blend_file.read_text(encoding="utf-8") == original_text
```

Add the symmetric deletion test: failed persistence restores the deleted mapping and returns `False`.

- [ ] **Step 3: Run backend tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest Tests/TTS/test_kokoro_backend_profile_paths.py -q
```

Expected: default path assertions fail against the global literal and failure tests expose `_save_blends()` reporting success indirectly.

- [ ] **Step 4: Implement default-directory selection without changing explicit paths**

```python
def default_kokoro_backend_blend_directory() -> Path:
    return get_cli_config_path().parent / "kokoro_voice_blends"
```

In `KokoroTTSBackend.__init__`, choose an explicit `config`/`get_cli_setting` value when present. Otherwise choose the helper and call:

```python
secure_private_directory(
    self.voice_blends_dir,
    create=True,
    application_owned=True,
)
```

For an explicit configured path, preserve the existing `mkdir(parents=True, exist_ok=True)` behavior and do not reclassify the directory as application-owned.

- [ ] **Step 5: Make disk and memory changes transactional**

Change `_save_blends()` to serialize through `write_private_json()` and return `True` only after replacement succeeds. In `save_voice_blend()`, save the prior mapping value, update memory, call `_save_blends()`, and restore memory on false/exception. In `delete_voice_blend()`, retain the removed value and restore it on false/exception. Keep the public boolean API.

- [ ] **Step 6: Run backend and manager verification**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_kokoro_backend_profile_paths.py \
  Tests/TTS/test_legacy_backend_registry.py \
  Tests/TTS/test_legacy_bridge.py \
  Tests/TTS/test_kokoro_validation.py \
  Tests/TTS/test_voice_blend_paths.py -q
.venv/bin/python scripts/check_profile_owned_path_inventory.py --print-occurrences
git diff --check
```

Expected: tests pass; backend blend state is profile-owned while Kokoro model/packaged voice paths remain shared occurrences.

- [ ] **Step 7: Commit backend ownership and atomicity**

```bash
git add tldw_chatbook/TTS Tests/TTS
git commit -m "fix(tts): persist backend blends privately per profile"
```

---

### Task 6: Route Dictation Exports and Embedding Runtime Cache to Active-User Data

**Files:**

- Modify: `tldw_chatbook/UI/Dictation_Window_Improved.py`
- Create: `Tests/UI/test_dictation_export_path.py`
- Create: `Tests/Utils/test_embedding_cache_path.py`

**Interfaces:**

- Consumes: `get_user_data_dir() -> Path` and existing `get_model_cache_dir()`.
- Produces: `dictation_export_directory() -> Path`; unchanged user-requested text/Markdown export semantics.

- [ ] **Step 1: Write two-user export helper and actual-method tests**

```python
def test_dictation_export_directory_retargets_each_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tldw_chatbook.UI.Dictation_Window_Improved as module

    alice = tmp_path / "data" / "alice"
    bob = tmp_path / "data" / "bob"
    monkeypatch.setattr(module, "get_user_data_dir", lambda: alice)
    assert module.dictation_export_directory() == alice / "exports" / "dictation"
    monkeypatch.setattr(module, "get_user_data_dir", lambda: bob)
    assert module.dictation_export_directory() == bob / "exports" / "dictation"
```

For each export method, construct `ImprovedDictationWindow` with `__new__`, set `transcript_text`, `duration`, `word_count`, and an app object with a mocked `notify`, call the actual `_export_as_text()` or `_export_as_markdown()`, and assert exactly one expected-suffix file appears beneath the selected scratch user directory.

- [ ] **Step 2: Write the embedding runtime-owner regression**

Patch `config.get_user_data_dir` to two scratch user directories and make `get_cli_setting("embedding_config", "model_cache_dir", default)` return the shipped default value. Call actual `get_model_cache_dir()` twice and assert the results are `<user>/models/embeddings`; assert the shipped literal is not used as a runtime filesystem target.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_dictation_export_path.py \
  Tests/Utils/test_embedding_cache_path.py -q
```

Expected: Dictation tests fail because both exports still use the global data literal. The embedding test should already pass and therefore pins its correct classification as an inert shipped default.

- [ ] **Step 4: Implement one Dictation export-directory owner**

```python
def dictation_export_directory() -> Path:
    return get_user_data_dir() / "exports" / "dictation"
```

Call this helper inside both export actions immediately before directory creation. Keep normal user-export file creation; do not route exports through `atomic_private_write_text()` and do not probe or move historical exports.

- [ ] **Step 5: Run Dictation, embedding, and existing path suites**

Run:

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_dictation_export_path.py \
  Tests/Utils/test_embedding_cache_path.py \
  Tests/UI/test_speech_recognition_no_write_on_open.py \
  Tests/UI/test_chat_screen_ui_state_path.py \
  Tests/Chatbooks/test_chatbook_importer.py -q
.venv/bin/python scripts/check_profile_owned_path_inventory.py --print-occurrences
git diff --check
```

Expected: all tests pass and no Improved Dictation hardcoded data root remains.

- [ ] **Step 6: Commit active-user path ownership**

```bash
git add tldw_chatbook/UI/Dictation_Window_Improved.py Tests/UI/test_dictation_export_path.py Tests/Utils/test_embedding_cache_path.py
git commit -m "fix(dictation): export beneath active user data"
```

---

### Task 7: Freeze the Final Census and Installed-Artifact Contract

**Files:**

- Modify: `scripts/check_profile_owned_path_inventory.py`
- Modify: `Tests/Architecture/test_profile_owned_path_inventory.py`
- Modify: `Tests/Packaging/test_installed_distribution.py`

**Interfaces:**

- Consumes: final scanner output and the existing `BuiltDistributions`/`INSTALLED_PROBE` production-app test.
- Produces: exact `APPROVED_EXCEPTIONS`, enforced source census, sdist/wheel absence contract, and installed full-`TldwCli` proof.

- [ ] **Step 1: Print and classify the final occurrence set**

Run:

```bash
.venv/bin/python scripts/check_profile_owned_path_inventory.py --print-occurrences
```

Every row must map to exactly one ADR-040 retained class:

- shipped config/Storage/RAG values and embedding-cache text → `PERSISTED_DEFAULT`;
- `DEFAULT_CONFIG_PATH` and `_default_base_data_dir` expression → `RESOLVER_SEED`;
- `BASE_DATA_DIR_CLI` → `COMPATIBILITY_CONSTANT`;
- TTS models, reusable Chatterbox/Higgs voices, and tokenizer artifacts → `SHARED_ARTIFACT`;
- the Evals stranded-data warning path → `READ_ONLY_LEGACY_PROBE`.

No UI Kokoro blend, dictionary, prompt, Settings copy, Code Repository copy, Improved Dictation export, rejected history, or legacy user-DB row may be allowlisted.

- [ ] **Step 2: Encode exact count-sensitive exception rules**

For each printed `(relative_path, context, expression)` triple, add one `ExceptionRule` with its observed final count, class, and asset/default/probe-specific reason. Sort rules by path, context, then expression. Run the checker once after encoding; it must exit zero. Then temporarily duplicate one retained literal in a scratch `scan_source()` fixture and assert reconciliation fails, proving a file-level allowlist cannot mask extra uses.

- [ ] **Step 3: Activate the production-census pytest gate**

Add:

```python
def test_production_profile_owned_path_inventory_is_exact() -> None:
    occurrences = scan_tree(REPO_ROOT / "tldw_chatbook")
    assert reconcile_inventory(occurrences, APPROVED_EXCEPTIONS) == ()
```

Add `test_shared_asset_exceptions_are_explicit()` that filters `APPROVED_EXCEPTIONS` for the Kokoro model modules, Chatterbox/Higgs voice modules, and `Utils/custom_tokenizers.py`; assert every selected rule has `Disposition.SHARED_ARTIFACT`. Separately assert the `config.py` embedding-cache rule has `Disposition.PERSISTED_DEFAULT`, preventing it from drifting back into the shared class.

Add an AST source-census test proving `BASE_DATA_DIR_CLI` has no production resolver consumer and its only repository consumer outside `config.py` is `Helper_Scripts/Prompts/Prompts_Dump.py`. Assert `runtime_policy.DEFAULT_RUNTIME_POLICY_PATH` remains defined only in `runtime_policy/bootstrap.py` and normal `default_runtime_policy_path()` remains callable.

Also invoke the CLI through `subprocess.run([sys.executable, ...])` and assert return code zero so developer and CI entry points share the same contract.

- [ ] **Step 4: Add retired modules to sdist and wheel absence assertions**

In `test_built_artifacts_match_distribution_contract`, define:

```python
retired_modules = {
    "tldw_chatbook/Audio/transcription_history.py",
    "tldw_chatbook/Widgets/transcription_history_viewer.py",
    "tldw_chatbook/UI/Dictation_Window.py",
}
assert retired_modules.isdisjoint(sdist_members)
assert retired_modules.isdisjoint(wheel_members)
```

- [ ] **Step 5: Extend the existing installed full-application probe**

Add `import importlib.util` to `INSTALLED_PROBE` and assert:

```python
for retired_module in (
    "tldw_chatbook.Audio.transcription_history",
    "tldw_chatbook.Widgets.transcription_history_viewer",
    "tldw_chatbook.UI.Dictation_Window",
):
    assert importlib.util.find_spec(retired_module) is None

from tldw_chatbook.config import get_cli_config_path, get_user_data_dir
assert get_cli_config_path().is_relative_to(Path(os.environ["HOME"]))
assert get_user_data_dir().is_relative_to(Path(os.environ["HOME"]))
```

Keep the existing `get_app()`, `isinstance(app, TldwCli)`, `app.run_test()`, Home-to-Chat navigation, source-root exclusion, loaded-module path checks, and target immutability hashes. Do not introduce another `App` class.

- [ ] **Step 6: Run architecture and installed-wheel tests**

Run:

```bash
.venv/bin/python -m pytest Tests/Architecture/test_profile_owned_path_inventory.py -q
.venv/bin/python -m pytest Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
  Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable -q
.venv/bin/python scripts/check_profile_owned_path_inventory.py
git diff --check
```

Expected: exact census passes; both artifacts omit retired modules; the installed wheel mounts and navigates the production `TldwCli` without checkout imports.

- [ ] **Step 7: Commit the enforcement boundary**

```bash
git add scripts/check_profile_owned_path_inventory.py Tests/Architecture/test_profile_owned_path_inventory.py Tests/Packaging/test_installed_distribution.py
git commit -m "test(packaging): enforce profile path ownership census"
```

---

### Task 8: Full Verification, Review, and TASK-865 Closeout

**Files:**

- Modify: `backlog/tasks/task-865 - Sweep-hardcoded-~-.config-tldw_cli-and-~-.local-share-tldw_cli-call-sites-onto-the-real-accessors.md`
- Modify: `Docs/superpowers/specs/2026-08-01-profile-owned-path-completion-design.md`
- Modify: `Docs/superpowers/plans/2026-08-01-profile-owned-path-completion.md`

**Interfaces:**

- Consumes: Tasks 1–7 and ADR-040.
- Produces: fresh verification evidence, completed acceptance criteria, final implementation notes, reviewer verdict, and Backlog status `Done` only if every DoD gate passes.

- [ ] **Step 1: Run the complete targeted matrix**

```bash
.venv/bin/python -m pytest \
  Tests/Architecture/test_retired_profile_path_owners.py \
  Tests/Architecture/test_profile_owned_path_inventory.py \
  Tests/Architecture/test_persistent_diagnostic_inventory.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/DB/test_private_sqlite_inventory.py \
  Tests/Chat/test_chat_functions.py \
  Tests/Prompt_Management/test_profile_owned_prompt_paths.py \
  Tests/TTS/test_voice_blend_paths.py \
  Tests/TTS/test_kokoro_backend_profile_paths.py \
  Tests/UI/test_profile_owned_settings_paths.py \
  Tests/UI/test_dictation_export_path.py \
  Tests/Utils/test_embedding_cache_path.py \
  Tests/Local_Ingestion/test_dictation_window_provider_ids.py \
  Tests/UI/test_chat_screen_ui_state_path.py \
  Tests/Chatbooks/test_chatbook_importer.py \
  Tests/Packaging/test_installed_distribution.py -q
```

- [ ] **Step 2: Run repository-wide tests and static checks**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check tldw_chatbook Tests scripts
.venv/bin/python -m ruff format --check tldw_chatbook Tests scripts
.venv/bin/python -m compileall -q tldw_chatbook scripts
.venv/bin/python scripts/check_profile_owned_path_inventory.py
.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
git diff --check
```

Record the exact pass/fail/skip counts. If an upstream baseline failure appears, reproduce it against `origin/dev` before deciding whether TASK-865 is blocked; do not mark the task Done while a TASK-865 regression remains.

- [ ] **Step 3: Perform a no-migration and no-Sync diff audit**

```bash
git diff --name-status origin/dev...HEAD
git diff origin/dev...HEAD -- tldw_chatbook/Notes tldw_chatbook/Event_Handlers/notes_events.py tldw_chatbook/Event_Handlers/note_ingest_events.py
git diff origin/dev...HEAD | rg 'copy2|copyfile|rename\(|replace\(|unlink\(|rmtree|shutil\.move'
```

Expected: the Notes/Sync diff is empty; any filesystem mutation match is either the approved atomic private replacement primitive call or existing user-export behavior, never migration/deletion of user data.

- [ ] **Step 4: Request final code review**

Use `superpowers:requesting-code-review` with base `origin/dev`, current `HEAD`, ADR-040, the approved design, and this plan. Resolve all Critical and Important findings and rerun every affected command before continuing.

- [ ] **Step 5: Reconcile TASK-865 source of truth**

Check all ten acceptance criteria only after the evidence exists. Replace the partial Implementation Notes with a concise final summary covering:

- retired plaintext/history and obsolete DB path code;
- canonicalized config-state and active-user paths;
- shared/inert/probe exceptions;
- private atomic Kokoro writes and manager-session trade-off;
- generated/curated inventory and installed-wheel evidence;
- exact verification results and any documented upstream baseline exception.

Set the design status to `Implemented and verified`. Check every completed step in this plan.

- [ ] **Step 6: Mark the Backlog task Done through the CLI**

Only when every acceptance criterion, test, static check, documentation update, and review gate is satisfied:

```bash
backlog task edit 865 -s Done
backlog task 865 --plain
```

Verify the plain output shows `TASK-865`, `Done`, all checked acceptance criteria, ADR-040, the design, and final Implementation Notes.

- [ ] **Step 7: Commit closeout documentation**

```bash
git add "backlog/tasks/task-865 - Sweep-hardcoded-~-.config-tldw_cli-and-~-.local-share-tldw_cli-call-sites-onto-the-real-accessors.md" Docs/superpowers/specs/2026-08-01-profile-owned-path-completion-design.md Docs/superpowers/plans/2026-08-01-profile-owned-path-completion.md
git commit -m "docs(config): close TASK-865 path ownership sweep"
```

- [ ] **Step 8: Verify the final commit state**

```bash
git status --short --branch
git log --oneline --decorate origin/dev..HEAD
git diff --check origin/dev...HEAD
```

Expected: clean worktree, reviewable task-scoped commit series, and no uncommitted verification or inventory changes.
