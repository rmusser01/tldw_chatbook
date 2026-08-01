# Console Prompt Improvement Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Console users a `Prompts` workbench action that opens a searchable Prompt/Recipe Library, safely improves the current unsent message with the active provider/model, and supports editable System/User block recipes that are first-class in Library > Prompts.

**Architecture:** First establish the fixed-width composer/overflow-menu contract from reference `TASK-1680`; `Prompts` remains a top Workbench action and never expands the composer button row. Add an artifact-type migration plus a version-and-kind-dispatched block codec around the existing Prompt table, compile v2 blocks back to legacy System/User text, and expose source capabilities through `PromptScopeService`. A shared block editor is hosted by one mode-driven `ConsolePromptsModal` and Library > Prompts. The composer owns an immutable segment snapshot/apply/restore transaction. A headless `PromptImprovementService` calls one typed, non-streaming, sensitive auxiliary completion through `ConsoleProviderGateway`, validates the response and preservation invariants, then asks the composer/session owners to apply changes.

**Tech Stack:** Python 3.11+, Textual 8.2.7, SQLite/FTS5, Pydantic, httpx, existing provider adapters and `chat_api_call`, pytest/pytest-asyncio/Hypothesis, Textual `App.run_test()`/Pilot, `tldw_server2` FastAPI/Pydantic v2 prompt APIs.

**Design spec:** `Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md`

**Reference composer work:** `/private/tmp/ephemeral`, `TASK-1680`, commits `d15e35e1c` and `e2ea3650b`

**Related server plans:**

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/Docs/superpowers/plans/2026-08-01-chat-prompt-improvement.md` (`TASK-12984.1`; WebUI/extension flow, not reused as Console transport)
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/Docs/superpowers/plans/2026-08-01-single-text-structured-recipes.md` (`TASK-12984.2`; coexisting schema-v2 `single_text_recipe` kind)

**ADR required:** yes

**ADR path:** `backlog/decisions/029-versioned-prompt-artifacts-and-safe-improvement-transactions.md`

**Reason:** This feature adds a persisted Prompt/Recipe discriminator, a new structured schema family with compiled compatibility fields, cross-client/server capability and concurrency contracts, a segment-safe composer mutation boundary, sensitive provider-call behavior, and a long-lived unified Console modal.

## Backlog Handoff

Do not invent task IDs in this planning checkout. Before implementation, rebase an isolated `codex/` worktree onto the current integration branch, sweep `backlog/tasks/` across local and remote worktrees using the repository's task-collision lesson, then create one atomic Backlog task for each delivery stage below. Link `TASK-1680`, ADR-029, this plan, and the design spec from each task. Put each task In Progress and add its implementation-plan excerpt before changing production code. Do not mark any task Done until its ACs, tests, static checks, documentation, implementation notes, and ADR hygiene are complete.

## Global Constraints

- Start from the current integration branch in a fresh worktree. `/private/tmp/ephemeral` is a read-only semantic reference whose branch also carries unrelated temporary-conversation history; do not edit it and do not blindly cherry-pick its commits or branch history.
- `TASK-1680` semantics are a prerequisite: the composer's at-rest row is `☰`, `Send`, and `Mic`; `Stop` and attachment clear are conditional; Attach and Save Chatbook route through stable overflow-menu action IDs to existing screen handlers; disabled rows show a visible reason.
- `Prompts` is a top Workbench action immediately after `New tab`. `Settings` is immediately before `Help`. Do not add `Prompts` to `ConsoleComposerBar` or its overflow menu.
- Preserve structured schema v1 byte/behavior compatibility. Dispatch by schema version and then definition kind; never let a Console v2 block artifact or the server's `single_text_recipe` v2 kind fall through the v1 parser.
- Canonical structured content lives in `prompt_definition`. `system_prompt` and `user_prompt` are regenerated compatibility fields, never a second editable source of truth.
- Existing Prompt rows migrate to `artifact_type="prompt"`. No new Prompt or Recipe table.
- A Recipe must be rejected by legacy prompt-use, picker, execution, and usage paths. Selecting a Recipe creates an unsaved Prompt working copy; it never inserts directly into the composer.
- Server support is capability-gated by exact `(schema_version, kind)` pairs. Version-only flags are forbidden because multiple incompatible v2 kinds coexist.
- Do not use the server WebUI `/api/v1/prompts/improve` plan as the Chatbook Console transport. The user required the current Console provider/model setup; use `ConsoleProviderGateway` and the active `ConsoleProviderSelection`.
- Model-dependent actions send only trusted optimizer instructions, the model-facing composer projection, optional system text, and an optional Recipe fill contract. No transcript history, tools, RAG, staged sources, pending attachments, inline-file content/metadata, or unrelated session state.
- One click means at most one provider call. Do not repair malformed output with a hidden second model call.
- Never silently truncate source text, definitions, compiled lanes, request bodies, or expected output. Surface the exact limit and recovery.
- The modal owns working-copy UI state, not the live composer/session. The composer owns draft transactions; `ConsoleChatStore` owns session-system mutation/persistence.
- Escape user/server-derived labels before Rich markup rendering; use `markup=False` for raw status/content Statics.
- Do not recompose unrelated `TextArea` widgets on block edits; preserve their cursor, selection, scroll, and native undo state.
- Workers taking more than 100 ms use Textual workers with named groups and stale-result tokens. Closing/cancelling detaches non-interruptible synchronous provider work and discards its result.
- Edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, then regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` with `.venv/bin/python tldw_chatbook/css/build_css.py`; never hand-edit the generated bundle.
- Use `.venv/bin/python -m pytest ...` from `tldw_chatbook`. Server commands run from `tldw_server2` with its `.venv/bin/python` and include Bandit over touched Python paths.
- Follow strict red-green-refactor. Each behavior test must fail for the intended missing behavior before production changes.
- Commit only the files named by the current task. Preserve unrelated dirty-worktree changes and use targeted `git add`.

## Delivery Map

| Gate | Outcome | Primary repository |
|---|---|---|
| Prerequisite | Composer buttons are unified and width-bounded before Prompt UX lands. | `tldw_chatbook` |
| Stage 1 | Artifact migration, block codec/compiler, source capabilities, server coexistence, import/export. | both |
| Stage 2 | Workbench action, unified Browse/Edit/Recipe modal, shared block editor, Library integration. | `tldw_chatbook` |
| Stage 3 | Exact composer snapshot/apply/restore and temporary Undo. | `tldw_chatbook` |
| Stage 4 | Sensitive auxiliary provider call, improvement orchestration, Auto/Review/Recipe fill, final QA. | `tldw_chatbook` |

---

## Prerequisite Gate: Land Composer Button Unification First

### Task 0: Port or verify `TASK-1680` semantics on the target branch

**Files:**

- Reference only: `/private/tmp/ephemeral/tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Reference only: `/private/tmp/ephemeral/tldw_chatbook/Widgets/Console/console_composer_menu_modal.py`
- Reference only: `/private/tmp/ephemeral/tldw_chatbook/UI/Screens/chat_screen.py`
- Reference only: `/private/tmp/ephemeral/Tests/UI/test_console_composer_menu.py`
- Modify only if the target lacks the behavior: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Create/modify only if absent: `tldw_chatbook/Widgets/Console/console_composer_menu_modal.py`
- Modify only if absent: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify only if absent: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify only if absent: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate only if TCSS changes: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Port/modify: `Tests/UI/test_console_composer_menu.py`
- Modify: `Tests/UI/test_console_composer_collapse.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`
- Modify: `Tests/UI/test_console_workbench_contract.py`

**Interfaces:**

- `BASE_ACTIONS_WIDTH = 28`; `ATTACHMENT_ACTIONS_WIDTH = 32`.
- `ComposerMenuEntry(action_id, label, description, enabled=True)`.
- `build_composer_menu_entries(*, attachment_kind, ephemeral, can_save_chatbook) -> tuple[ComposerMenuEntry, ...]`.
- Stable action IDs `attach-context` and `save-chatbook` route to existing ChatScreen handlers.
- A disabled menu entry renders its reason as `.console-composer-menu-reason`, not only as a tooltip.

- [ ] **Step 1: Rebase and inspect semantic presence**

Run from the implementation worktree:

```bash
git fetch origin
git rebase origin/dev
git show --stat d15e35e1c
git show --stat e2ea3650b
rg -n "BASE_ACTIONS_WIDTH|build_composer_menu_entries|console-composer-menu-reason" tldw_chatbook Tests
```

If all contracts and focused tests already exist, make no production edit and continue with the verification steps. If any contract is missing, port the minimal behavior manually against the rebased files. Do not cherry-pick either reference commit because its parent chain contains unrelated temporary-conversation work.

- [ ] **Step 2: Write/port the failing action-surface tests**

Tests must prove:

```python
assert 'id="console-attach-context"' not in inspect.getsource(ConsoleComposerBar.compose)
assert 'id="console-save-chatbook"' not in inspect.getsource(ConsoleComposerBar.compose)
assert [entry.action_id for entry in build_composer_menu_entries()] == [
    "attach-context",
    "save-chatbook",
    "generate-image",
    "generate-caption",
    "narrate-conversation",
    "impersonate",
]
```

Also assert that Save Chatbook and Generate Caption blocked rows remain visible and include their exact reason.

- [ ] **Step 3: Run the focused tests and verify RED only where the target is missing behavior**

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workbench_contract.py -q
```

Expected on a missing target: failures name the duplicated Attach/Save controls or absent visible disabled reason. Expected on an already-landed target: all pass, and Task 0 becomes a verification-only commit/task note.

- [ ] **Step 4: Port the minimal menu and handler routing if required**

Keep the composer row width fixed. Build the menu entries from current staged-attachment, ephemeral-session, and Chatbook-availability state. In `ChatScreen`, have menu selections invoke the same existing methods used by their prior buttons; do not fork attachment/save implementations.

- [ ] **Step 5: Verify geometry and action parity**

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python -m pytest \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_destination_visual_parity_correction.py -q
git diff --check
```

- [ ] **Step 6: Commit only if a port was needed**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/Widgets/Console/console_composer_menu_modal.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_composer_menu.py Tests/UI/test_console_composer_collapse.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workbench_contract.py
git commit -m "refactor(console): unify composer actions before prompt workbench"
```

---

## Stage 1: Versioned Prompt/Recipe Foundation

### Task 1: Add the Console block-v2 codec, compiler, and legacy decomposition

**Files:**

- Create: `tldw_chatbook/Prompt_Management/prompt_artifact_models.py`
- Create: `tldw_chatbook/Prompt_Management/prompt_artifact_codec.py`
- Create: `tldw_chatbook/Prompt_Management/prompt_block_compiler.py`
- Create: `tldw_chatbook/Prompt_Management/prompt_legacy_decomposer.py`
- Create: `Docs/fixtures/console-block-prompts/render-cases.json`
- Create: `Docs/fixtures/console-block-prompts/error-cases.json`
- Create: `Tests/Prompt_Management/test_prompt_artifact_codec.py`
- Create: `Tests/Prompt_Management/test_prompt_block_compiler.py`
- Create: `Tests/Prompt_Management/test_prompt_legacy_decomposer.py`

**Interfaces:**

```python
ArtifactType = Literal["prompt", "recipe"]
ArtifactDefinitionState = Literal[
    "legacy", "supported_v2", "foreign_v1", "unsupported", "malformed", "mismatched"
]
BlockSyntax = Literal["freeform", "markdown", "xml"]

@dataclass(frozen=True)
class PromptBlock:
    id: str
    title: str
    syntax: BlockSyntax
    content: str
    xml_tag: str | None = None
    mapping_hint: str | None = None

@dataclass(frozen=True)
class PromptLane:
    id: Literal["system", "user"]
    blocks: tuple[PromptBlock, ...]

@dataclass(frozen=True)
class BlockArtifactDefinition:
    kind: Literal["block_prompt", "block_recipe"]
    schema_version: Literal[2]
    lanes: tuple[PromptLane, PromptLane]

@dataclass(frozen=True)
class LegacyLaneOrigin:
    text: str
    fingerprint: str

@dataclass(frozen=True)
class LegacyDecomposition:
    definition: BlockArtifactDefinition
    system_origin: LegacyLaneOrigin
    user_origin: LegacyLaneOrigin

@dataclass(frozen=True)
class DecodedPromptArtifact:
    state: ArtifactDefinitionState
    artifact_type: ArtifactType
    definition: BlockArtifactDefinition | None
    raw_definition: Mapping[str, Any] | None
    compiled_system: str
    compiled_user: str
    compatibility_stale: bool

def decode_prompt_artifact(record: Mapping[str, Any]) -> DecodedPromptArtifact: ...
def compile_block_artifact(definition: BlockArtifactDefinition) -> tuple[str, str]: ...
def decompose_legacy_lanes(system_prompt: str, user_prompt: str) -> LegacyDecomposition: ...
def deserialize_definition(value: Any) -> Mapping[str, Any] | None: ...
def decode_console_v2(
    record: Mapping[str, Any], *, artifact_type: ArtifactType, raw: Mapping[str, Any]
) -> DecodedPromptArtifact: ...
def foreign_definition(
    record: Mapping[str, Any],
    artifact_type: ArtifactType,
    raw: Mapping[str, Any] | None,
    *,
    state: Literal["foreign_v1", "unsupported"],
) -> DecodedPromptArtifact: ...
def malformed_definition(
    record: Mapping[str, Any], artifact_type: ArtifactType
) -> DecodedPromptArtifact: ...
def validate_xml_wrapper(xml_tag: str | None, content: str) -> None: ...
```

- [ ] **Step 1: Write failing codec dispatch tests**

Cover legacy, valid `block_prompt`, valid `block_recipe`, v1 structured, future version, malformed JSON, missing lanes, duplicate block IDs, missing/duplicate lane IDs, kind/artifact mismatch, column/definition version mismatch, and foreign `single_text_recipe` v2. Put accepted/rejected structured cases in the shared JSON fixtures and have the tests read them. Assert exact `state` values; never assert only “raised.”

```python
def test_single_text_recipe_v2_is_foreign_not_console_recipe():
    record = structured_record(
        artifact_type="recipe",
        version=2,
        definition={
            "schema_version": 2,
            "definition_kind": "single_text_recipe",
            "blocks": [],
        },
    )

    decoded = decode_prompt_artifact(record)

    assert decoded.state == "unsupported"
    assert decoded.definition is None
    assert decoded.raw_definition == record["prompt_definition"]
```

- [ ] **Step 2: Write failing compiler tests**

Pin exact whitespace for free-form, Markdown `# {title}`, XML wrappers, two-newline separators, empty blocks, Unicode, and two lanes in `render-cases.json`. Reject invalid XML names and an opening/closing/self-closing wrapper collision through `error-cases.json` while preserving the original in the returned validation issue.

- [ ] **Step 3: Write failing conservative-decomposer tests**

Use fenced Markdown/XML mixtures, nested tags, incomplete wrappers, headings inside code fences, and ambiguous text. Recognize only complete top-level headings/wrappers; everything else becomes free-form. Record each lane's original text/fingerprint so an unchanged lane can reapply byte-for-byte.

- [ ] **Step 4: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/Prompt_Management/test_prompt_artifact_codec.py \
  Tests/Prompt_Management/test_prompt_block_compiler.py \
  Tests/Prompt_Management/test_prompt_legacy_decomposer.py -q
```

- [ ] **Step 5: Implement strict models and dispatch**

Parse `prompt_definition` from local JSON strings or server dictionaries once. Dispatch on `prompt_schema_version`; for v2 accept only `kind in {block_prompt, block_recipe}`. Treat `definition_kind="single_text_recipe"` as foreign v2. Verify record artifact type, definition kind, and schema version agree before returning `supported_v2`.

```python
def decode_prompt_artifact(record: Mapping[str, Any]) -> DecodedPromptArtifact:
    raw_artifact_type = str(record.get("artifact_type") or "prompt")
    if raw_artifact_type not in {"prompt", "recipe"}:
        raise ValueError("Unsupported artifact_type")
    artifact_type = cast(ArtifactType, raw_artifact_type)
    prompt_format = str(record.get("prompt_format") or "legacy")
    if prompt_format == "legacy":
        return DecodedPromptArtifact(
            state="legacy",
            artifact_type=artifact_type,
            definition=None,
            raw_definition=None,
            compiled_system=str(record.get("system_prompt") or ""),
            compiled_user=str(record.get("user_prompt") or ""),
            compatibility_stale=False,
        )

    raw = deserialize_definition(record.get("prompt_definition"))
    version = record.get("prompt_schema_version")
    if version == 1:
        return foreign_definition(record, artifact_type, raw, state="foreign_v1")
    if version != 2:
        return foreign_definition(record, artifact_type, raw, state="unsupported")
    if raw is None:
        return malformed_definition(record, artifact_type)
    if raw.get("definition_kind") == "single_text_recipe":
        return foreign_definition(record, artifact_type, raw, state="unsupported")
    return decode_console_v2(record, artifact_type=artifact_type, raw=raw)
```

- [ ] **Step 6: Implement deterministic compilation and conservative parsing**

Do not strip block content. Omit empty compiled blocks. Preserve unchanged legacy lane text via stored origin fingerprint; after any lane edit compile the edited lane normally.

```python
def compile_block(block: PromptBlock) -> str:
    if block.content == "":
        return ""
    if block.syntax == "freeform":
        return block.content
    if block.syntax == "markdown":
        return f"# {block.title}\n\n{block.content}"
    validate_xml_wrapper(block.xml_tag, block.content)
    return f"<{block.xml_tag}>{block.content}</{block.xml_tag}>"


def compile_lane(lane: PromptLane) -> str:
    return "\n\n".join(
        rendered for block in lane.blocks if (rendered := compile_block(block))
    )
```

- [ ] **Step 7: Run focused tests plus v1 parity**

```bash
.venv/bin/python -m pytest \
  Tests/Prompt_Management/test_prompt_artifact_codec.py \
  Tests/Prompt_Management/test_prompt_block_compiler.py \
  Tests/Prompt_Management/test_prompt_legacy_decomposer.py \
  Tests/Prompts_DB/test_prompts_db_server_parity.py -q
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Prompt_Management/prompt_artifact_models.py tldw_chatbook/Prompt_Management/prompt_artifact_codec.py tldw_chatbook/Prompt_Management/prompt_block_compiler.py tldw_chatbook/Prompt_Management/prompt_legacy_decomposer.py Docs/fixtures/console-block-prompts Tests/Prompt_Management/test_prompt_artifact_codec.py Tests/Prompt_Management/test_prompt_block_compiler.py Tests/Prompt_Management/test_prompt_legacy_decomposer.py
git commit -m "feat(prompts): add versioned block artifact codec"
```

### Task 2: Migrate local Prompt storage and add transactional expected-version updates

**Files:**

- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- Modify: `tldw_chatbook/Prompt_Management/server_prompt_adapter.py`
- Modify: `tldw_chatbook/Prompt_Management/local_prompt_service.py`
- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `Tests/Prompts_DB/test_prompts_db_pytest.py`
- Modify: `Tests/Prompts_DB/test_prompts_db_server_parity.py`
- Modify: `Tests/Prompt_Management/test_local_prompt_service.py`
- Modify: `Tests/Prompt_Management/test_prompt_scope_service.py`
- Modify: `Tests/Prompt_Management/test_server_prompt_adapter.py`

**Interfaces:**

- Local Prompts DB schema v3 adds `artifact_type TEXT NOT NULL DEFAULT 'prompt' CHECK(artifact_type IN ('prompt','recipe'))`.
- `PromptsDatabase.update_prompt_by_id(..., expected_version: int | None = None)` checks the caller's version inside the update transaction.
- Create/update/list/detail/search/sync payloads carry `artifact_type`.
- Brief/list rows include `artifact_type`, `has_system_prompt`, `has_user_prompt`; lane flags are derived in SELECTs from compiled fields.
- `PromptScopeService.save_prompt(..., artifact_type, expected_version)` forwards the concurrency contract only when supported.

- [ ] **Step 1: Add failing migration tests**

Create a real v2 database, reopen with new code, and assert schema v3, all rows `artifact_type == "prompt"`, structured metadata unchanged, FTS still works, sync triggers still require one version increment, and a fresh v3 DB has the same column/default.

```python
def test_v2_migration_defaults_existing_rows_to_prompt(tmp_path):
    database_path = tmp_path / "prompts.db"
    seed_v2_prompt_database(database_path, name="Existing", user_prompt="alpha")

    database = PromptsDatabase(database_path, client_id="migration-test")
    detail = database.fetch_prompt_details("Existing")

    assert database._get_db_version(database.get_connection()) == 3
    assert detail["artifact_type"] == "prompt"
    assert detail["user_prompt"] == "alpha"
```

- [ ] **Step 2: Add failing CRUD/brief/search tests**

Create one Prompt and one Recipe. Assert list/search return type and lane flags without detail fetches, detail round-trips canonical definition, and FTS searches compiled content. Assert invalid artifact types fail at the boundary.

- [ ] **Step 3: Add failing optimistic-concurrency tests**

Update with the captured version succeeds once; a second update with that stale version raises `ConflictError` and leaves every field unchanged. Prove the comparison happens in the same transaction by using two DB instances against one file.

- [ ] **Step 4: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompts_DB/test_prompts_db_server_parity.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Prompt_Management/test_server_prompt_adapter.py -q
```

- [ ] **Step 5: Implement schema v3 and payload propagation**

Update `_CURRENT_SCHEMA_VERSION`, `_MIGRATIONS`, create/overwrite/update/select/sync payloads, interop conversions, and service field lists. Do not infer Recipe from definition in normal records; artifact type is first-class and mismatches are reported by the codec.

```sql
ALTER TABLE Prompts
ADD COLUMN artifact_type TEXT NOT NULL DEFAULT 'prompt'
CHECK (artifact_type IN ('prompt', 'recipe'));
UPDATE schema_version SET version = 3 WHERE version = 2;
```

Every brief SELECT derives lane flags without detail fetches:

```sql
CASE WHEN length(trim(coalesce(system_prompt, ''))) > 0 THEN 1 ELSE 0 END
    AS has_system_prompt,
CASE WHEN length(trim(coalesce(user_prompt, ''))) > 0 THEN 1 ELSE 0 END
    AS has_user_prompt
```

- [ ] **Step 6: Implement expected-version enforcement**

If `expected_version` is provided and differs from the row read inside the transaction, raise before building the update. Keep the SQL `WHERE id=? AND version=?` guard. `Update original` consumers must always pass a captured version; Save as new passes none.

```python
current_version = int(existing_prompt_state["version"])
if expected_version is not None and int(expected_version) != current_version:
    raise ConflictError(
        "Prompt changed after it was opened.", "Prompts", prompt_id
    )
new_version = current_version + 1
cursor.execute(update_sql, (*params, prompt_id, current_version))
if cursor.rowcount != 1:
    raise ConflictError("Prompt update lost a version race.", "Prompts", prompt_id)
```

- [ ] **Step 7: Run focused and property tests**

```bash
.venv/bin/python -m pytest \
  Tests/Prompts_DB \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Prompt_Management/test_server_prompt_adapter.py -q
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/DB/Prompts_DB.py tldw_chatbook/Prompt_Management/Prompts_Interop.py tldw_chatbook/Prompt_Management/server_prompt_adapter.py tldw_chatbook/Prompt_Management/local_prompt_service.py tldw_chatbook/Prompt_Management/prompt_scope_service.py Tests/Prompts_DB Tests/Prompt_Management/test_local_prompt_service.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Prompt_Management/test_server_prompt_adapter.py
git commit -m "feat(prompts): store Prompt and Recipe artifact types"
```

### Task 3: Extend import/export without breaking the compatibility grammar

**Files:**

- Modify: `tldw_chatbook/Prompt_Management/prompt_markdown_export.py`
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- Modify: `Tests/Library/test_prompt_export_roundtrip.py`
- Modify: `Tests/Prompt_Management/test_server_prompt_adapter.py`

**Interfaces:**

- Structured Markdown appends `### ARTIFACT_TYPE ###` and `### STRUCTURE ###` with canonical JSON in one fenced block.
- Unknown future version/kind imports as a new legacy Prompt from compiled `SYSTEM`/`USER` text; it never partially persists foreign structure.
- Known v2 exact round-trip restores IDs, lane/order, title, syntax, XML tag, content, and mapping hints.

- [ ] **Step 1: Add failing round-trip/compatibility tests**

Cover legacy export unchanged, Prompt and Recipe structured exact round-trip, JSON containing section-like text, malformed fenced JSON, discriminator mismatch, foreign v1, foreign `single_text_recipe`, and unknown future version fallback to legacy Prompt.

```python
def test_block_recipe_markdown_round_trip_preserves_definition():
    detail = structured_recipe_detail()

    markdown = render_prompt_markdown(detail)
    [imported] = parse_markdown_prompts_from_content(markdown)

    assert imported["artifact_type"] == "recipe"
    assert imported["prompt_schema_version"] == 2
    assert imported["prompt_definition"] == detail["prompt_definition"]
```

- [ ] **Step 2: Run and verify RED**

```bash
.venv/bin/python -m pytest Tests/Library/test_prompt_export_roundtrip.py Tests/Prompt_Management/test_server_prompt_adapter.py -q
```

- [ ] **Step 3: Extend the exact section map and fenced parser**

Use the existing next-section terminator grammar. Validate through `decode_prompt_artifact`; do not duplicate schema validation in import code. Export canonical JSON with stable key/order settings suitable for exact fixture comparison.

```python
def structured_markdown_sections(detail: Mapping[str, Any]) -> str:
    definition = canonical_definition(detail)
    structure = json.dumps(
        definition,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"\n### ARTIFACT_TYPE ###\n{detail['artifact_type']}\n"
        f"\n### STRUCTURE ###\n```json\n{structure}\n```\n"
    )
```

- [ ] **Step 4: Verify old-reader compatibility explicitly**

Feed the structured export through a characterization copy of the old parser behavior and assert `TITLE`, `SYSTEM`, `USER`, and `KEYWORDS` remain readable while the unknown appended sections are ignored.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest Tests/Library/test_prompt_export_roundtrip.py Tests/Prompt_Management/test_server_prompt_adapter.py -q
git diff --check
git add tldw_chatbook/Prompt_Management/prompt_markdown_export.py tldw_chatbook/Prompt_Management/Prompts_Interop.py Tests/Library/test_prompt_export_roundtrip.py Tests/Prompt_Management/test_server_prompt_adapter.py
git commit -m "feat(prompts): round-trip structured artifacts in markdown"
```

### Task 4: Add server block kinds, artifact type, brief/search fields, and capabilities

Work in a clean `tldw_server2` worktree. Do not modify the currently dirty server checkout. Before production changes, update the active `TASK-12984.2` plan/task so the shared parser explicitly admits all coexisting v2 kinds; whichever server task lands first must preserve the other's discriminants and fixtures.

**Files (`/Users/macbook-dev/Documents/GitHub/tldw_server2`):**

- Modify if Track 2 is not already compatible: `Docs/superpowers/plans/2026-08-01-single-text-structured-recipes.md`
- Modify if Track 2 is not already compatible: `backlog/tasks/task-12984.2 - Implement-single-text-structured-prompt-recipes.md`
- Modify: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/models.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/validator.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/assembler.py`
- Create: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/block_renderer.py`
- Modify: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/__init__.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/prompt_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompts.py`
- Create: `Docs/fixtures/console-block-prompts/render-cases.json`
- Create: `Docs/fixtures/console-block-prompts/error-cases.json`
- Modify: `tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py`
- Modify: `tldw_Server_API/tests/Prompt_Management/test_structured_prompt_validator.py`
- Modify: `tldw_Server_API/tests/Prompt_Management/test_structured_prompt_assembler.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py`

**Interfaces:**

- Server DB schema v6 adds the same `artifact_type` default/check.
- Server structured parser explicitly dispatches:
  - schema v1 -> existing multi-message model unchanged;
  - schema v2 + `definition_kind="single_text_recipe"` -> `TASK-12984.2` model unchanged;
  - schema v2 + `kind="block_prompt"|"block_recipe"` -> Console block model.
- Prompt brief/search/detail/create/update include artifact type and derived lane flags.
- `/api/v1/prompts/health` includes a `capabilities` object with exact supported `(schema_version, kind)` entries, artifact types, search, conditional update, and size limits.
- Server conditional update remains advertised false until an authenticated endpoint enforces `expected_version`; Chatbook keeps Update disabled meanwhile.

- [ ] **Step 1: Write failing coexistence tests before merging model changes**

Pin v1 behavior, `single_text_recipe` behavior if already present, Console block Prompt/Recipe behavior, and cross-kind rejection. Copy the reviewed Chatbook block fixtures byte-for-byte into the server repository and make server tests read them. A v2 block payload without `kind` must not be accepted as v1 or single-text. A `single_text_recipe` must never parse as a Console Recipe merely because both are version 2.

```python
def test_v2_dispatch_keeps_single_text_and_block_recipe_distinct():
    single = parse_prompt_definition(valid_single_text_recipe())
    block = parse_prompt_definition(valid_block_recipe())

    assert isinstance(single, SingleTextRecipeDefinitionV2)
    assert isinstance(block, BlockArtifactDefinitionV2)
    assert single.definition_kind == "single_text_recipe"
    assert block.kind == "block_recipe"
```

- [ ] **Step 2: Write failing DB/API/capability tests**

Migrate v5 rows to Prompt, create/search/list each artifact kind, derive lane flags from compiled text, reject kind/type/version mismatch without mutation, and assert exact capability pairs. Keep each compiled lane within the existing 20,000-character API limit.

- [ ] **Step 3: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py \
  tldw_Server_API/tests/Prompt_Management/test_structured_prompt_validator.py \
  tldw_Server_API/tests/Prompt_Management/test_structured_prompt_assembler.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py -q
```

- [ ] **Step 4: Implement schema v6 and explicit nested dispatch**

Do not use `schema_version` alone as a Pydantic union discriminator for multiple v2 models. Use one parse function: branch v1, then inspect the v2 kind field(s), then validate with the exact model. Preserve `TASK-12984.2`'s `definition_kind` field; do not silently rename persisted single-text definitions.

```python
def parse_prompt_definition(payload: Mapping[str, Any]) -> ParsedPromptDefinition:
    version = payload.get("schema_version")
    if version == 1:
        return MultiMessagePromptDefinitionV1.model_validate(payload)
    if version != 2:
        raise ValueError("unsupported schema_version")
    if payload.get("definition_kind") == "single_text_recipe":
        return SingleTextRecipeDefinitionV2.model_validate(payload)
    if payload.get("kind") in {"block_prompt", "block_recipe"}:
        return BlockArtifactDefinitionV2.model_validate(payload)
    raise ValueError("unsupported schema-v2 definition kind")
```

- [ ] **Step 5: Add server block compilation and capabilities**

Match the Chatbook compilation fixtures exactly. Keep the current v1 assembler response shape. The health response must contain no user data and must remain useful to older clients that ignore the new key.

- [ ] **Step 6: Run server gates**

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Prompt_Management \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py -q
.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/DB_Management/Prompts_DB.py \
  tldw_Server_API/app/core/Prompt_Management/structured_prompts \
  tldw_Server_API/app/api/v1/schemas/prompt_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/prompts.py
git diff --check
```

- [ ] **Step 7: Commit in the server worktree and record the commit in the Chatbook Backlog task**

Use targeted staging. If `TASK-12984.2` landed first, this commit extends its union. If this task lands first, amend the Track 2 plan so its implementation extends rather than replaces this dispatcher.

```bash
git add Docs/superpowers/plans/2026-08-01-single-text-structured-recipes.md "backlog/tasks/task-12984.2 - Implement-single-text-structured-prompt-recipes.md" tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/models.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/validator.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/assembler.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/block_renderer.py tldw_Server_API/app/core/Prompt_Management/structured_prompts/__init__.py tldw_Server_API/app/api/v1/schemas/prompt_schemas.py tldw_Server_API/app/api/v1/endpoints/prompts.py Docs/fixtures/console-block-prompts/render-cases.json Docs/fixtures/console-block-prompts/error-cases.json tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py tldw_Server_API/tests/Prompt_Management/test_structured_prompt_validator.py tldw_Server_API/tests/Prompt_Management/test_structured_prompt_assembler.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py
git diff --cached --check
git commit -m "feat(prompts): support Console block artifacts"
```

### Task 5: Normalize source capabilities and enable honest server search

**Files:**

- Create: `tldw_chatbook/Prompt_Management/prompt_source_capabilities.py`
- Modify: `tldw_chatbook/Prompt_Management/prompt_normalizers.py`
- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `tldw_chatbook/Prompt_Management/server_prompt_service.py`
- Modify: `tldw_chatbook/tldw_api/prompt_chatbook_schemas.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `Tests/Prompt_Management/test_prompt_scope_service.py`
- Modify: `Tests/Prompt_Management/test_server_prompt_service.py`
- Modify: `Tests/tldw_api/test_prompt_chatbook_schemas.py`
- Modify: `Tests/tldw_api/test_prompt_chatbook_client.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PromptSourceCapabilities:
    backend: Literal["local", "server"]
    structured_kinds: frozenset[tuple[int, str]]
    artifact_types: frozenset[str]
    search: bool
    conditional_update: bool
    compiled_lane_limit: int
    definition_limit: int
    request_limit: int

class PromptCapabilityError(ValueError):
    def __init__(self, backend: str, capability: str) -> None:
        self.backend = backend
        self.capability = capability
        super().__init__(f"{backend} prompt source does not support {capability}.")

async def PromptScopeService.get_capabilities(*, mode) -> PromptSourceCapabilities: ...
```

Use exact in-process/fallback limits: `compiled_lane_limit=20_000` characters,
`definition_limit=256_000` UTF-8 bytes, and `request_limit=512_000` UTF-8 bytes.
A modern server's smaller advertised limit wins. These limits are validation
errors, never truncation targets.

- [ ] **Step 1: Add failing normalization tests**

Assert local known capabilities, modern server response, older health without capabilities, malformed health, and a server advertising only `single_text_recipe`. Older servers remain browsable, normalize missing artifact type as Prompt, and disable block-v2 Save/search as appropriate.

- [ ] **Step 2: Add failing server-search routing tests**

`PromptScopeService.search_prompts(mode="server", query="alpha")` must call `ServerPromptService.search_prompts` and normalize the response. Empty-query Browse must call paginated `list_prompts`, not the server search endpoint whose query is non-empty. A policy denial or missing capability becomes a typed unavailable outcome, not an empty result list.

```python
@pytest.mark.asyncio
async def test_server_search_routes_to_server_endpoint():
    server = FakeServerPromptService(search_items=[server_prompt_brief("Alpha")])
    service = PromptScopeService(FakeLocalPromptService(), server)

    items = await service.search_prompts(mode="server", query="alpha", limit=25)

    assert server.search_calls == [{"search_query": "alpha", "page": 1, "results_per_page": 25, "include_deleted": False}]
    assert items[0]["backend"] == "server"
    assert items[0]["name"] == "Alpha"
```

- [ ] **Step 3: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Prompt_Management/test_server_prompt_service.py \
  Tests/tldw_api/test_prompt_chatbook_schemas.py \
  Tests/tldw_api/test_prompt_chatbook_client.py -q
```

- [ ] **Step 4: Implement capabilities and source-aware search**

Normalize collection records through `decode_prompt_artifact` only when full definitions are present. Brief records carry type/lane flags without definition parsing. Preserve backend/source IDs and optimistic version. Add explicit errors for unsupported search and save kinds.

```python
async def search_prompts(self, *, mode: PromptBackend | str, query: str, limit: int = 25, **kwargs: Any) -> list[dict[str, Any]]:
    backend = self._normalize_mode(mode)
    capabilities = await self.get_capabilities(mode=backend)
    if not capabilities.search:
        raise PromptCapabilityError(backend.value, "search")
    service = self._service_for_mode(backend)
    response = await self._maybe_await(
        service.search_prompts(
            search_query=query,
            page=1,
            results_per_page=limit,
            include_deleted=False,
        )
    )
    return normalize_prompt_search(response, backend=backend.value)
```

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/python -m pytest \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Prompt_Management/test_server_prompt_service.py \
  Tests/tldw_api/test_prompt_chatbook_schemas.py \
  Tests/tldw_api/test_prompt_chatbook_client.py -q
git diff --check
git add tldw_chatbook/Prompt_Management/prompt_source_capabilities.py tldw_chatbook/Prompt_Management/prompt_normalizers.py tldw_chatbook/Prompt_Management/prompt_scope_service.py tldw_chatbook/Prompt_Management/server_prompt_service.py tldw_chatbook/tldw_api/prompt_chatbook_schemas.py tldw_chatbook/tldw_api/client.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Prompt_Management/test_server_prompt_service.py Tests/tldw_api/test_prompt_chatbook_schemas.py Tests/tldw_api/test_prompt_chatbook_client.py
git commit -m "feat(prompts): expose source capabilities and server search"
```

---

## Stage 2: Unified Browse, Edit, Recipe, and Library UI

### Task 6: Build the shared block-editor widget without destructive recomposition

**Files:**

- Create: `tldw_chatbook/Widgets/Prompts/__init__.py`
- Create: `tldw_chatbook/Widgets/Prompts/prompt_block_editor.py`
- Create: `tldw_chatbook/Widgets/Prompts/prompt_block_editor_state.py`
- Create: `Tests/UI/test_prompt_block_editor.py`
- Create: `Tests/Prompt_Management/test_prompt_block_editor_state.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PromptBlockValidationIssue:
    block_id: str
    field: Literal["id", "title", "syntax", "xml_tag", "content"]
    code: str
    message: str

@dataclass(frozen=True)
class PromptBlockEditorState:
    artifact_type: ArtifactType
    definition: BlockArtifactDefinition
    compiled_system: str
    compiled_user: str
    issues: tuple[PromptBlockValidationIssue, ...]
    dirty_block_ids: frozenset[str]
    system_origin: LegacyLaneOrigin | None = None
    user_origin: LegacyLaneOrigin | None = None

def update_block(
    state: PromptBlockEditorState, block_id: str, **changes: str | None
) -> PromptBlockEditorState: ...
def move_block(
    state: PromptBlockEditorState, block_id: str, direction: Literal[-1, 1]
) -> PromptBlockEditorState: ...
```

- `PromptBlockEditorState` owns immutable lane/block ordering, dirty flags, origin fingerprints, validation issues, and compiled previews.
- `PromptBlockEditor` emits typed messages for field changes, add/move/duplicate/delete, Save as Prompt, Save as Recipe, Update original, and Apply.
- Stable block widget IDs derive from collision-safe block IDs, not titles or list indexes.
- `replace_block_state(block_id, ...)` patches only the affected block controls; lane reorder moves widgets without reconstructing unaffected `TextArea`s.

- [ ] **Step 1: Add failing pure-state tests**

Cover add, reorder boundaries, duplicate with new stable ID, delete, syntax change, XML validation, dirty/origin tracking, unchanged-legacy exactness, compiled preview, Prompt/Recipe kind changes, and reserved Additional-context ID rejection.

- [ ] **Step 2: Add failing mounted widget tests**

At 120x40 and 80x24, assert stacked System/User lanes, expanded non-empty lanes, every required control, visible validation beside its block, first-error focus, and two-row footer when narrow. Capture one TextArea object's identity/cursor/selection/scroll, edit another block, and assert the first remains unchanged.

```python
@pytest.mark.asyncio
async def test_editing_one_block_preserves_other_textarea_state():
    app = BlockEditorHarness(two_block_state())
    async with app.run_test(size=(80, 24)) as pilot:
        untouched = app.query_one("#prompt-block-content-context", TextArea)
        untouched.cursor_location = (0, 3)
        original_identity = id(untouched)

        await pilot.click("#prompt-block-content-goal")
        await pilot.press("x")
        await pilot.pause()

        same = app.query_one("#prompt-block-content-context", TextArea)
        assert id(same) == original_identity
        assert same.cursor_location == (0, 3)
```

- [ ] **Step 3: Run and verify RED**

```bash
.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_block_editor_state.py Tests/UI/test_prompt_block_editor.py -q
```

- [ ] **Step 4: Implement pure state then incremental widget updates**

Use explicit buttons/keyboard commands for reorder. Show XML tag input only for XML. `Apply system prompt to this session` is unchecked by default; User lane application is selected by default only when non-empty. Empty/unselected lanes are no-ops and all-empty selection disables Apply.

```python
def update_block(
    state: PromptBlockEditorState,
    block_id: str,
    **changes: str | None,
) -> PromptBlockEditorState:
    definition = replace_block_by_id(state.definition, block_id, **changes)
    system_text, user_text = compile_block_artifact(definition)
    return replace(
        state,
        definition=definition,
        compiled_system=system_text,
        compiled_user=user_text,
        issues=validate_block_artifact(definition),
        dirty_block_ids=state.dirty_block_ids | {block_id},
    )
```

- [ ] **Step 5: Verify and commit**

```bash
.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_block_editor_state.py Tests/UI/test_prompt_block_editor.py -q
git diff --check
git add tldw_chatbook/Widgets/Prompts/__init__.py tldw_chatbook/Widgets/Prompts/prompt_block_editor.py tldw_chatbook/Widgets/Prompts/prompt_block_editor_state.py Tests/UI/test_prompt_block_editor.py Tests/Prompt_Management/test_prompt_block_editor_state.py
git commit -m "feat(prompts): add shared System and User block editor"
```

### Task 7: Add the Workbench action and unified `ConsolePromptsModal` Browse/Edit shell

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_workbench_state.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Create: `tldw_chatbook/Widgets/Console/console_prompts_state.py`
- Create: `tldw_chatbook/Widgets/Console/console_prompts_browse.py`
- Create: `tldw_chatbook/Widgets/Console/console_prompts_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_console_prompts_modal.py`
- Create: `Tests/UI/test_console_control_bar_actions.py`
- Modify: `Tests/UI/test_console_workbench_contract.py`
- Modify: `Tests/UI/test_console_workbench_parity_matrix.py`
- Modify: `Tests/UI/test_workbench_visual_snapshots.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PromptBrowseResult:
    source: Literal["local", "server"]
    items: tuple[Mapping[str, Any], ...]
    page: int
    total_pages: int
    total_items: int
```

- Workbench action order is exactly `new-tab`, `prompts`, `attach-context`, `run-library-rag`, `save-chatbook`, `settings`, `help`. Remove the duplicate header `send`/`stop` actions; `ConsoleComposerBar` owns Send and conditional Stop after the prerequisite gate.
- `ConsoleControlBar` is the visible action surface; `#console-workbench-command-strip` remains a hidden compatibility seam. Add `prompts` to `TOP_ACTION_IDS` and map it to `console-control-prompts`.
- Pressing `#console-control-prompts` posts the existing `WorkbenchActionRequested("prompts")` path; `ChatScreen` handles that action by pushing exactly one `ConsolePromptsModal` in Browse mode. Do not add a second button-specific dispatch protocol.
- `CONSOLE_ACTIONS_WIDE_MIN_WIDTH = 112` and `CONSOLE_ACTIONS_SINGLE_ROW_MIN_WIDTH = 80`. Width >=112 uses full labels on one row; width 80-111 uses compact labels on one row; width <80 uses compact labels split 4+3 over two rows in logical order.
- Compact labels are exact: `New`, `Prompts`, `Attach`, `RAG`, `Save`, `Settings`, `Help`. Full labels come from Workbench state: `New tab`, `Prompts`, `Attach context`, `Run Library RAG`, `Save Chatbook`, `Settings`, `Help`.
- `ConsoleControlBar` keeps its existing provider/model chip row above the actions. It therefore changes total height between two rows (one chip + one action) and three rows (one chip + two actions) on resize. `ChatScreen.compose_content` stops wrapping this widget in `_compact_console_workbench_widget(..., height=2)`; the other compact Workbench widgets retain their current helper behavior. Remove the fixed height/min/max declarations for `#console-control-bar` from TCSS so the widget's measured 2/3-row height is authoritative.
- `ConsolePromptsModal` internal modes: Browse, Edit, Improve, Recipe.
- Browse places a visible `Improve My Prompt` button above the source/search controls. It enters Improve mode inside the same modal and preserves the user's Browse query, source, page, selection, and focus return point.
- `console_prompts_state.py` owns the mode stack, source/search tokens, and selected-source identity/version; `console_prompts_browse.py` owns only source/search/pagination rendering and events; `console_prompts_modal.py` owns navigation, focus restoration, dirty dismissal, and child-mode coordination.
- Browse dependencies are injected callables: capabilities, list page, search query, detail fetch, save.
- Empty query -> paginated list; non-empty query -> backend search with 200 ms debounce and monotonic token.

- [ ] **Step 1: Add failing action-order/geometry tests**

Assert full labels at wide width, compact labels at medium width, deterministic two rows at narrow width, no clipping, Prompts after New tab, and Settings immediately before Help. Assert clicking `#console-control-prompts` emits the `prompts` action and opens one Browse modal. Assert `ConsoleComposerBar` still contains no Prompts control.

```python
def test_console_header_actions_have_one_owner_and_expected_order():
    state = build_console_workbench_state(control_state=ready_controls())

    assert [action.id for action in state.actions] == [
        "new-tab",
        "prompts",
        "attach-context",
        "run-library-rag",
        "save-chatbook",
        "settings",
        "help",
    ]
    assert "console-prompts" not in inspect.getsource(ConsoleComposerBar.compose)
```

Mounted `ConsoleControlBar` tests use sizes `(140, 30)`, `(100, 30)`, and
`(70, 30)`. At 70 columns assert row 1 IDs are New/Prompts/Attach/RAG, row 2
IDs are Save/Settings/Help, the chip row remains visible, bar height is 3, and
every chip/button region lies inside the bar region. At 100/140 assert bar
height is 2 and action row 2 is hidden.

- [ ] **Step 2: Add failing modal Browse tests**

Cover a top-of-view `Improve My Prompt` action that enters Improve and returns to the unchanged Browse state, empty Local Library, empty-query pagination, non-empty search, source switching, stale completion rejection, no matches, Retry after failure, source unavailable, selected row deleted before detail fetch, and composer focus restoration on dismiss.

- [ ] **Step 3: Add failing artifact-opening tests**

Supported Prompt -> Edit. Recipe -> unsaved Prompt working copy. Legacy -> conservative blocks. Foreign v1/foreign `single_text_recipe`/future/malformed/mismatched -> read-only compiled compatibility view with explicit Convert and save as new only when compiled content is valid. No open action makes a model call or changes usage metadata.

- [ ] **Step 4: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_console_prompts_modal.py \
  Tests/UI/test_console_control_bar_actions.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_workbench_visual_snapshots.py -q
```

- [ ] **Step 5: Implement action dispatch and stable modal shell**

Copy the visual language of `ConsoleContextModal`; do not subclass it. Keep one modal screen and swap internal mode widgets while retaining each mode's focus key. Escape behaves as Back; dirty Edit/Recipe offers only Keep editing/Discard.

```python
class ConsolePromptsModal(ModalScreen[None]):
    MODES = ("browse", "edit", "improve", "recipe")

    def enter_mode(self, mode: str, *, focus_id: str | None = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported prompt modal mode: {mode}")
        self._focus_by_mode[self._mode] = self.focused.id if self.focused else None
        self._mode = mode
        self._mount_mode(mode)
        self._restore_mode_focus(focus_id or self._focus_by_mode.get(mode))
```

Implement the visible action layout in `ConsoleControlBar` with one pure mode
selector and two stable row containers:

```python
def console_action_layout(width: int) -> tuple[str, tuple[int, ...]]:
    if width >= CONSOLE_ACTIONS_WIDE_MIN_WIDTH:
        return "wide", (7,)
    if width >= CONSOLE_ACTIONS_SINGLE_ROW_MIN_WIDTH:
        return "compact", (7,)
    return "narrow", (4, 3)


def on_resize(self, event: Resize) -> None:
    mode, row_lengths = console_action_layout(event.size.width)
    self._sync_visible_action_rows(mode=mode, row_lengths=row_lengths)
    bar_height = 1 + len(row_lengths)  # persistent chip row + action rows
    self.styles.height = bar_height
    self.styles.min_height = bar_height
    self.styles.max_height = bar_height
```

- [ ] **Step 6: Implement source-aware Browse and guarded Edit**

Fetch latest detail after selection. Capture source ID/version/capability snapshot. Do not client-filter one server page. Provider unavailable disables only model actions; browsing/editing/saving supported kinds remain available.

```python
async def _run_browse(self, query: str, token: int) -> None:
    result = (
        await self._search(self._source, query)
        if query.strip()
        else await self._list_page(self._source, self._page)
    )
    if token != self._search_token or self._source != result.source:
        return
    await self._render_browse_result(result)
```

- [ ] **Step 7: Style, regenerate, and verify**

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python -m pytest \
  Tests/UI/test_console_prompts_modal.py \
  Tests/UI/test_console_control_bar_actions.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_workbench_visual_snapshots.py \
  Tests/UI/test_destination_visual_parity_correction.py -q
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workbench_state.py tldw_chatbook/Widgets/Console/console_control_bar.py tldw_chatbook/Widgets/Console/console_prompts_state.py tldw_chatbook/Widgets/Console/console_prompts_browse.py tldw_chatbook/Widgets/Console/console_prompts_modal.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_prompts_modal.py Tests/UI/test_console_control_bar_actions.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_workbench_parity_matrix.py Tests/UI/test_workbench_visual_snapshots.py
git commit -m "feat(console): open Prompt Library from the workbench"
```

### Task 8: Make Prompt/Recipe blocks first-class in Library > Prompts and guard legacy use paths

**Files:**

- Modify: `tldw_chatbook/Library/library_prompts_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_prompt_picker_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Library/test_library_prompts_state.py`
- Modify: `Tests/UI/test_library_prompts_canvas.py`
- Modify: `Tests/UI/test_console_prompt_picker.py`
- Modify: `Tests/UI/test_console_command_composer.py`
- Modify: `Tests/Library/test_prompt_export_roundtrip.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PromptArtifactDraft:
    artifact_type: ArtifactType
    definition: BlockArtifactDefinition
    system_prompt: str
    user_prompt: str
    definition_bytes: bytes
    request_bytes: bytes

def require_artifact_save_supported(
    draft: PromptArtifactDraft,
    capabilities: PromptSourceCapabilities,
) -> None: ...
def recipe_lane(
    lane_id: Literal["system", "user"], block_ids: tuple[str, ...]
) -> PromptLane: ...
```

- Library list rows show Prompt/Recipe plus System/User/combined lane summary.
- Library editor uses the same `PromptBlockEditor`; compiled preview is read-only.
- Save as Prompt/Recipe validates exact selected-source capabilities and limits.
- Update original is enabled only with conditional update and a current captured version.
- `/prompt`, `/system`, existing picker/apply, and usage recording filter/reject Recipes.

- [ ] **Step 1: Add failing Library list/editor tests**

Assert type labels, lane labels, legacy editor compatibility, shared block editor for supported v2, exact compiled preview, source/limit errors naming the field, Save as new, local expected-version conflict with Reload/Save as new, and server Update disabled when conditional update is false.

- [ ] **Step 2: Add failing Recipe execution-guard tests**

Seed a Recipe and prove `/prompt`, `/system`, Console picker, Library “Use in Console,” and usage counting cannot apply it. Selecting it from the new Browse flow must create a Prompt copy with no usage increment.

```python
@pytest.mark.asyncio
async def test_recipe_cannot_enter_legacy_prompt_picker():
    search = AsyncMock(return_value=[normalized_recipe_brief("Outcome first")])
    app = PromptPickerHarness(prompt_search=search)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert len(app.query(".console-prompt-picker-row")) == 0
        assert app.selected_prompt is None
```

- [ ] **Step 3: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/Library/test_library_prompts_state.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_console_prompt_picker.py \
  Tests/UI/test_console_command_composer.py \
  Tests/Library/test_prompt_export_roundtrip.py -q
```

- [ ] **Step 4: Wire shared editor and save validation**

Keep Library's existing list/filter/import/export behavior. Extend editor state rather than replacing the screen's source/scoping logic. Validate artifact type, exact kind capability, compiled lane lengths, definition bytes, and request bytes before calling save; do not truncate.

```python
def require_artifact_save_supported(
    draft: PromptArtifactDraft,
    capabilities: PromptSourceCapabilities,
) -> None:
    pair = (draft.definition.schema_version, draft.definition.kind)
    if pair not in capabilities.structured_kinds:
        raise PromptCapabilityError(capabilities.backend, f"structured kind {pair}")
    enforce_text_limit("system_prompt", draft.system_prompt, capabilities.compiled_lane_limit)
    enforce_text_limit("user_prompt", draft.user_prompt, capabilities.compiled_lane_limit)
    enforce_bytes_limit("prompt_definition", draft.definition_bytes, capabilities.definition_limit)
    enforce_bytes_limit("request", draft.request_bytes, capabilities.request_limit)
```

- [ ] **Step 5: Add the built-in Outcome-first Recipe**

Create it in a pure factory in `prompt_artifact_models.py` with System blocks Role, Personality, Collaboration style and User blocks Goal, Success criteria, Context and evidence, Constraints, Output, Stop rules. Built-in identity is immutable; editing uses a working copy. Blank Recipe remains available. Saved Recipes may omit current fill content unless the explicit `Include current text as starter content` checkbox is selected.

```python
def outcome_first_recipe() -> BlockArtifactDefinition:
    return BlockArtifactDefinition(
        kind="block_recipe",
        schema_version=2,
        lanes=(
            recipe_lane("system", ("role", "personality", "collaboration-style")),
            recipe_lane(
                "user",
                ("goal", "success-criteria", "context-evidence", "constraints", "output", "stop-rules"),
            ),
        ),
    )
```

- [ ] **Step 6: Verify and commit**

```bash
.venv/bin/python -m pytest \
  Tests/Library/test_library_prompts_state.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_console_prompt_picker.py \
  Tests/UI/test_console_command_composer.py \
  Tests/Library/test_prompt_export_roundtrip.py -q
git diff --check
git add tldw_chatbook/Library/library_prompts_state.py tldw_chatbook/Widgets/Library/library_prompts_canvas.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Console/console_prompt_picker_modal.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Prompt_Management/prompt_artifact_models.py Tests/Library/test_library_prompts_state.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_console_prompt_picker.py Tests/UI/test_console_command_composer.py Tests/Library/test_prompt_export_roundtrip.py
git commit -m "feat(prompts): edit Prompt and Recipe blocks in Library"
```

---

## Stage 3: Exact Composer Transactions

### Task 9: Add public immutable snapshot/projection/apply/restore APIs to the composer

This task starts from the composer segment model present after Task 0. Extend it; do not create a second draft representation in the modal or service.

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Create: `Tests/UI/test_console_composer_improvement_transaction.py`
- Modify: `Tests/UI/test_console_composer_collapse.py`
- Modify: `Tests/UI/test_console_composer_cursor.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**

```python
DraftSegmentOrigin = Literal["literal", "paste", "inline_file"]

@dataclass(frozen=True)
class ComposerDraftSegmentSnapshot:
    text: str
    origin: DraftSegmentOrigin
    collapse_state: Literal["literal", "collapsed", "confirm", "expanded"]
    label: str | None

@dataclass(frozen=True)
class ComposerDraftSnapshot:
    segments: tuple[ComposerDraftSegmentSnapshot, ...]
    cursor_index: int
    selection: tuple[int, int] | Literal["all"] | None
    edit_serial: int
    fingerprint: str

@dataclass(frozen=True)
class ComposerModelProjection:
    text: str
    placeholder_nonce: str
    placeholder_ids: tuple[str, ...]
    fingerprint: str

def capture_draft_snapshot(self) -> ComposerDraftSnapshot: ...
def project_snapshot_for_model(self, snapshot, *, request_nonce) -> ComposerModelProjection: ...
def apply_improvement(self, snapshot, rewritten_model_text) -> ComposerDraftSnapshot: ...
def restore_snapshot(self, snapshot) -> None: ...
```

- [ ] **Step 1: Add failing origin/snapshot tests**

Prove typed text, small/large ordinary paste, expanded/collapsed paste, inline file, cursor, full selection, display state, label, and edit serial round-trip exactly. Change `_DraftSegment` to carry explicit origin; do not infer inline file from `label is not None` after migration.

- [ ] **Step 2: Add failing projection tests**

Literal and paste content must appear. Inline-file content, filename/path/label/size metadata must not. Placeholders use a request nonce verified absent from improvable text. Pending attachment store state is outside the snapshot and unchanged.

```python
def test_model_projection_keeps_paste_and_hides_inline_file_metadata():
    composer = composer_with_segments(
        literal("Draft "),
        paste("ordinary pasted text"),
        inline_file("SECRET FILE BODY", label="notes.md · 2 KB"),
    )
    snapshot = composer.capture_draft_snapshot()

    projection = composer.project_snapshot_for_model(snapshot, request_nonce="nonce-1")

    assert "Draft ordinary pasted text" in projection.text
    assert "SECRET FILE BODY" not in projection.text
    assert "notes.md" not in projection.text
    assert len(projection.placeholder_ids) == 1
```

- [ ] **Step 3: Add failing apply/veto tests**

Exact-once, original-order placeholders rehydrate original inline-file segments. Removed, duplicated, edited, reordered, or user-colliding tokens raise a typed validation error and leave the composer byte/state identical. A stale snapshot fingerprint/edit serial/session owner cannot apply.

- [ ] **Step 4: Add failing exact Undo invalidation tests**

Apply returns/records the pre-apply snapshot. Undo restores segments, cursor, selection, paste state, and inline-file metadata exactly. Undo expires on the next manual edit, send, session switch, or later improvement. `no_change` creates no Undo.

- [ ] **Step 5: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_console_composer_improvement_transaction.py \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_composer_cursor.py \
  Tests/UI/test_console_native_chat_flow.py -q
```

- [ ] **Step 6: Implement the minimal public transaction boundary**

Reuse `dataclasses.replace` for immutable segment copies. Keep send-time `ConsoleDraftStash` semantics but share copying helpers where safe. `load_draft()` remains for legacy callers; improvement code must not call it.

```python
def capture_draft_snapshot(self) -> ComposerDraftSnapshot:
    self._ensure_editable_segments()
    segments = tuple(
        ComposerDraftSegmentSnapshot(
            text=segment.text,
            origin=segment.origin,
            collapse_state=segment.collapse_state,
            label=segment.label,
        )
        for segment in self._segments
    )
    return ComposerDraftSnapshot(
        segments=segments,
        cursor_index=self._cursor_index,
        selection="all" if self._draft_selection_all else None,
        edit_serial=self._user_edit_serial,
        fingerprint=fingerprint_segments(segments),
    )


def project_snapshot_for_model(
    self,
    snapshot: ComposerDraftSnapshot,
    *,
    request_nonce: str,
) -> ComposerModelProjection:
    tokens = build_collision_free_tokens(snapshot, request_nonce=request_nonce)
    text = "".join(
        tokens[index] if segment.origin == "inline_file" else segment.text
        for index, segment in enumerate(snapshot.segments)
    )
    return ComposerModelProjection(
        text=text,
        placeholder_nonce=request_nonce,
        placeholder_ids=tuple(tokens.values()),
        fingerprint=sha256_text(text),
    )
```

`apply_improvement` first verifies the live snapshot fingerprint/edit serial,
then verifies each placeholder once and in original order, rebuilds private
segments from the immutable snapshot, and only after all checks pass swaps
`self._segments` and refreshes the widget. `restore_snapshot` performs the same
single final swap after validating the snapshot shape.

- [ ] **Step 7: Verify mutation sensitivity**

Temporarily negate each placeholder cardinality/order check and the edit-serial guard; confirm a focused test fails. Restore the guard before proceeding.

- [ ] **Step 8: Run and commit**

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_console_composer_improvement_transaction.py \
  Tests/UI/test_console_composer_collapse.py \
  Tests/UI/test_console_composer_cursor.py \
  Tests/UI/test_console_native_chat_flow.py -q
git diff --check
git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_composer_improvement_transaction.py Tests/UI/test_console_composer_collapse.py Tests/UI/test_console_composer_cursor.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): add reversible prompt improvement transactions"
```

---

## Stage 4: Sensitive Provider Call and Improvement UX

### Task 10: Add a strict one-shot auxiliary completion and harden sensitive logging

**Files:**

- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/Chat_Functions.py`
- Create: `tldw_chatbook/Utils/sensitive_llm_logging.py`
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (Cohere full-payload log; metadata-only request logs remain)
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` (Kobold prompt/full-payload logs; metadata-only request logs remain)
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Create: `Tests/Chat/test_sensitive_llm_logging.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class AuxiliaryCompletionRequest:
    resolution: ConsoleProviderResolution
    messages: tuple[Mapping[str, Any], ...]
    response_format: Mapping[str, Any] | None
    max_output_tokens: int
    sensitive: Literal[True] = True

@dataclass(frozen=True)
class AuxiliaryCompletionResult:
    provider: str
    model: str
    text: str

async def ConsoleProviderGateway.complete_auxiliary(
    self, request: AuxiliaryCompletionRequest
) -> AuxiliaryCompletionResult: ...
```

- [ ] **Step 1: Audit adapter logging before writing the policy**

```bash
rg -n "(logger|logging)\.(debug|info|warning|error).*?(payload|messages|prompt|response|content)" tldw_chatbook/LLM_Calls tldw_chatbook/Chat
```

Confirm the only chat-adapter body-content sites are native Kobold prompt/full-payload logging and Cohere full-payload logging. If the audit reports a newly introduced body-content site outside the two named adapter files, stop this task and amend its Backlog AC/file list before editing that file. Do not assume the existing egress URL redaction covers body content.

- [ ] **Step 2: Add failing gateway contract tests**

With injected provider fakes, assert active resolution/model/endpoint/samplers/reasoning settings are pinned, `streaming=False`, tools/tool choice/stop absent, output limit set, compatible `response_format` forwarded, one adapter call, empty remains empty, malformed shape raises, and normal transcript/store methods are never touched.

```python
@pytest.mark.asyncio
async def test_auxiliary_completion_is_one_shot_nonstreaming_and_tool_free():
    calls: list[dict[str, Any]] = []
    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **kwargs: calls.append(kwargs) or '{"kind":"prompt_rewrite","rewritten_prompt":"Better"}')

    result = await gateway.complete_auxiliary(auxiliary_request(openai_resolution()))

    assert result.text.endswith('"Better"}')
    assert len(calls) == 1
    assert calls[0]["streaming"] is False
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert "stop" not in calls[0]
```

- [ ] **Step 3: Add failing log-capture tests**

Capture Loguru and stdlib logs for representative generic, Kobold, Cohere, and direct llama.cpp auxiliary paths. Plant unique canaries in optimizer instruction, system text, user text, placeholder, block content, and generated response. Assert none appear; provider/model/mode/duration/byte counts may appear. Also assert ordinary non-sensitive calls retain existing diagnostics except removed unsafe full-payload logs.

```python
def assert_sensitive_canaries_absent(log_text: str) -> None:
    for canary in (
        "SYSTEM-CANARY",
        "USER-CANARY",
        "BLOCK-CANARY",
        "OPAQUE-CANARY",
        "RESPONSE-CANARY",
    ):
        assert canary not in log_text
```

- [ ] **Step 4: Run and verify RED**

```bash
.venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_sensitive_llm_logging.py -q
```

- [ ] **Step 5: Implement typed auxiliary dispatch**

For direct llama.cpp, use non-streaming completion. For generic providers, extend `_chat_api_kwargs` through an auxiliary-only builder that passes `response_format`, forces non-streaming, removes tools/tool choice/stop, and does not call `normalize_provider_response` fallback-copy synthesis. Run synchronous adapters through `asyncio.to_thread`; cancellation may detach but its result is ignored.

```python
async def complete_auxiliary(
    self,
    request: AuxiliaryCompletionRequest,
) -> AuxiliaryCompletionResult:
    resolution = replace(request.resolution, streaming=False)
    if not resolution.ready or not resolution.model:
        raise ChatConfigurationError("Pinned provider is not ready.")
    if resolution.provider in {"llama_cpp", "local_llamacpp"}:
        text = await self.complete_llamacpp_chat(
            base_url=resolution.base_url,
            model=resolution.model,
            messages=list(request.messages),
            max_tokens=request.max_output_tokens,
        )
    else:
        kwargs = self._auxiliary_chat_api_kwargs(request, resolution)
        text = await asyncio.to_thread(self._complete_sensitive_sync, kwargs)
    if not isinstance(text, str):
        raise ChatProviderError("Provider returned an unsupported auxiliary response.")
    return AuxiliaryCompletionResult(resolution.provider, resolution.model, text)
```

- [ ] **Step 6: Propagate request-scoped sensitive policy to final adapters**

Use a `ContextVar`-backed context manager established around the actual adapter invocation. Replace risky adapter log values with metadata-only helpers that consult the policy. Do not globally disable logging or mutate process-wide logger levels; concurrent normal chat calls must remain observable. Remove unconditional full-payload/full-response logs where no safe metadata form is needed.

```python
_SENSITIVE_LLM_REQUEST: ContextVar[bool] = ContextVar(
    "sensitive_llm_request", default=False
)

@contextmanager
def sensitive_llm_request() -> Iterator[None]:
    token = _SENSITIVE_LLM_REQUEST.set(True)
    try:
        yield
    finally:
        _SENSITIVE_LLM_REQUEST.reset(token)

def safe_llm_log_value(value: object) -> object:
    return "<sensitive-content-redacted>" if _SENSITIVE_LLM_REQUEST.get() else value
```

The gateway's worker invokes `chat_api_call` inside this context. Kobold and
Cohere log only provider, model, URL host, streaming flag, and byte counts when
the flag is set; neither constructs a string containing the body before asking
the policy helper.

- [ ] **Step 7: Prove the flag reaches the final adapter**

Tests must fail if the context manager wraps only the feature service or gateway but not the worker-thread handler. Use `contextvars.copy_context()` when dispatching to a thread if required so the adapter sees the sensitive flag.

- [ ] **Step 8: Run gateway, adapter, and import gates**

```bash
.venv/bin/python -m pytest \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_sensitive_llm_logging.py \
  Tests/Chat/test_console_variant_stream.py \
  Tests/UI/test_console_native_chat_flow.py -q
.venv/bin/python -c "import tldw_chatbook.Chat.console_provider_gateway"
git diff --check
```

- [ ] **Step 9: Commit**

Stage the two named adapter files with the gateway, dispatcher, policy helper, and tests.

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/Utils/sensitive_llm_logging.py tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_sensitive_llm_logging.py
git diff --cached --check
git commit -m "feat(console): add sensitive auxiliary provider completions"
```

### Task 11: Implement `PromptImprovementService`, preservation guards, and typed outcomes

**Files:**

- Create: `tldw_chatbook/Prompt_Management/prompt_improvement_models.py`
- Create: `tldw_chatbook/Prompt_Management/prompt_improvement_prompts.py`
- Create: `tldw_chatbook/Prompt_Management/prompt_preservation.py`
- Create: `tldw_chatbook/Prompt_Management/prompt_improvement_service.py`
- Create: `Tests/Prompt_Management/test_prompt_improvement_service.py`
- Create: `Tests/Prompt_Management/test_prompt_preservation.py`
- Create: `Tests/Prompt_Management/fixtures/prompt_improvement_cases.json`

**Interfaces:**

```python
ImprovementMode = Literal["auto", "review", "recipe"]
ImprovementOutcomeKind = Literal[
    "success", "no_change", "empty", "unsupported", "cancelled",
    "provider_error", "malformed", "preservation_veto", "context_limit", "stale"
]

@dataclass(frozen=True)
class PromptImprovementRequestSnapshot:
    request_id: str
    mode: ImprovementMode
    session_id: str
    composer_snapshot: ComposerDraftSnapshot
    projection: ComposerModelProjection
    system_prompt: str | None
    system_fingerprint: str | None
    resolution: ConsoleProviderResolution
    provider_label: str
    model_label: str
    recipe_source_id: str | None
    recipe_version: int | None
    recipe_definition: BlockArtifactDefinition | None
    recipe_fingerprint: str | None

@dataclass(frozen=True)
class PromptImprovementOutcome:
    request_id: str
    kind: ImprovementOutcomeKind
    rewritten_prompt: str | None = None
    filled_definition: BlockArtifactDefinition | None = None
    provider: str = ""
    model: str = ""
    user_message: str = ""

class PromptImprovementService:
    async def improve(self, snapshot: PromptImprovementRequestSnapshot) -> PromptImprovementOutcome: ...
```

Define `UNKNOWN_MODEL_CONTEXT_CAP_TOKENS = 32_768` and
`MAX_AUXILIARY_OUTPUT_TOKENS = 16_384`. For a known model, use its advertised
context/output limits. Requested output is the estimated source/fill size plus
a 1,024-token envelope allowance, capped by the model and application maximum;
preflight reserves that full output allowance inside the context window.

- [ ] **Step 1: Add failing optimizer-envelope tests**

Auto/Review accept only `{kind:"prompt_rewrite", rewritten_prompt:str}`. Recipe accepts only the captured fingerprint, exactly one fill per selected block ID, no duplicate/unknown/missing IDs, and one `additional_context` string. Test one outer JSON fence unwrapping, exact inner whitespace preservation, empty, fallback-copy strings, answer-like output, and no hidden retry.

```python
@pytest.mark.asyncio
async def test_malformed_response_is_not_repaired_with_second_call():
    gateway = FakeAuxiliaryGateway(responses=["not json"])
    service = PromptImprovementService(gateway=gateway)

    outcome = await service.improve(rewrite_snapshot("Rewrite me"))

    assert outcome.kind == "malformed"
    assert gateway.call_count == 1
```

- [ ] **Step 2: Add failing trusted/untrusted serialization tests**

Source fields are JSON values, not interpolated into closable XML delimiters. Embed adversarial text such as fake instructions, closing tags, and JSON-looking strings; assert they remain data. Assert system omission truly removes it from request bytes.

- [ ] **Step 3: Add failing preservation tests**

Extract and compare supported template placeholders, fenced code blocks, URLs, UUID-like identifiers, XML wrapper names, and opaque file placeholders. A missing/renamed protected item produces `preservation_veto`; prompt byte identity produces `no_change`.

- [ ] **Step 4: Add failing Recipe local-merge tests**

Model output may change only `content` by known block ID. Titles, syntax, XML tags, lane/order, IDs, and mapping hints come from the captured canonical Recipe. Non-empty additional context creates one reserved local block; empty creates none. Validate final block schema/compilation.

- [ ] **Step 5: Add failing context-limit/cancellation/stale tests**

Known model limits and the exact 32,768-token unknown-model cap preflight full input plus the calculated output allowance without truncation. Assert system exclusion recovery. Cancellation returns cancelled once. Changed request/session/draft/system/provider/model/recipe fingerprints return stale and never call apply.

- [ ] **Step 6: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/Prompt_Management/test_prompt_improvement_service.py \
  Tests/Prompt_Management/test_prompt_preservation.py -q
```

- [ ] **Step 7: Implement outcome-first trusted prompts and strict parsing**

Optimizer instructions preserve intent, artifact, language, audience, length, genre, facts, safety/business invariants, output fields, and side-effect limits; remove redundant process narration only when safe. Keep personality and collaboration distinct only when present. Never demand headings for every simple prompt.

```python
def parse_rewrite_envelope(text: str) -> str:
    payload = parse_one_json_object(unwrap_one_outer_json_fence(text))
    if set(payload) != {"kind", "rewritten_prompt"}:
        raise MalformedImprovementResponse("Unexpected rewrite response fields.")
    if payload["kind"] != "prompt_rewrite":
        raise MalformedImprovementResponse("Unexpected rewrite response kind.")
    rewritten = payload["rewritten_prompt"]
    if not isinstance(rewritten, str) or rewritten == "":
        raise EmptyImprovementResponse("Provider returned no rewritten prompt.")
    return rewritten
```

- [ ] **Step 8: Implement preflight, one-call orchestration, and metadata-only telemetry**

Record request ID, provider/model, mode, duration, input/output byte counts, token counts if provided, and typed outcome. Never log source/result content or placeholders.

```python
async def improve(
    self,
    snapshot: PromptImprovementRequestSnapshot,
) -> PromptImprovementOutcome:
    auxiliary_request = build_auxiliary_request(snapshot)
    preflight_request(auxiliary_request, snapshot.resolution)
    try:
        response = await self._gateway.complete_auxiliary(auxiliary_request)
    except asyncio.CancelledError:
        return cancelled_outcome(snapshot.request_id)
    except ChatProviderError as exc:
        return provider_error_outcome(snapshot.request_id, exc)
    return self._parse_and_validate(snapshot, response)
```

- [ ] **Step 9: Verify mutation sensitivity and commit**

Temporarily remove the duplicate-ID guard, Recipe fingerprint guard, and one preservation extractor; each corresponding test must fail. Restore all guards.

```bash
.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_improvement_service.py Tests/Prompt_Management/test_prompt_preservation.py -q
git diff --check
git add tldw_chatbook/Prompt_Management/prompt_improvement_models.py tldw_chatbook/Prompt_Management/prompt_improvement_prompts.py tldw_chatbook/Prompt_Management/prompt_preservation.py tldw_chatbook/Prompt_Management/prompt_improvement_service.py Tests/Prompt_Management/test_prompt_improvement_service.py Tests/Prompt_Management/test_prompt_preservation.py Tests/Prompt_Management/fixtures/prompt_improvement_cases.json
git commit -m "feat(prompts): orchestrate one-shot prompt improvement"
```

### Task 12: Complete Auto, Review, and Structured Recipe flows in the modal

**Files:**

- Create: `tldw_chatbook/Widgets/Console/console_prompt_improve_view.py`
- Modify: `tldw_chatbook/Widgets/Console/console_prompts_state.py`
- Modify: `tldw_chatbook/Widgets/Console/console_prompts_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `Tests/UI/test_console_prompts_modal.py`
- Modify: `Tests/UI/test_console_composer_improvement_transaction.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ConsolePromptsResult:
    kind: Literal["apply"]
    composer_snapshot: ComposerDraftSnapshot
    user_text: str | None
    system_text: str | None
    apply_user: bool
    apply_system: bool
    captured_system_fingerprint: str | None
```

- Improve view exposes exactly three choices: Analyze and auto-improve; Analyze and user review; Create or follow a structured recipe.
- `Include system prompt as analysis context` defaults on.
- Review shows one editable rewritten User prompt and protected opaque file tokens; no score, diff, findings, explanation, or hidden reasoning.
- Structured Recipe always opens filled content in the block editor for review.
- System lane application requires a separate checkbox defaulting off.
- ChatScreen pins session/provider/model/system/draft/recipe fingerprints and owns result application coordination.
- At this stage, update `ConsolePromptsModal` from its Browse/Edit-only `ModalScreen[None]` contract to `ModalScreen[ConsolePromptsResult | None]`; earlier stages must not import the composer transaction type before Task 9 defines it.

- [ ] **Step 1: Add failing modal-state tests**

Assert exact three choices; separate read-only captured System and Unsent message sections; protected opaque tokens instead of inline-file metadata/content; provider/model/system-send disclosure; system-context default on; unavailable-provider recovery; Auto/Review disabled for empty improvable text; Recipe manual editing available empty; and Recipe AI fill disabled empty.

- [ ] **Step 2: Add failing Auto tests**

Success applies through `composer.apply_improvement`, closes modal, and exposes temporary Undo. `no_change` shows `Prompt already looks good` and makes no mutation/Undo/usage event. Preservation veto routes to Review with only `Review required before applying`. Provider/malformed/context errors preserve modal state and offer explicit Retry.

```python
@pytest.mark.asyncio
async def test_auto_no_change_does_not_mutate_or_create_undo():
    app = PromptModalHarness(outcome=no_change_outcome())
    before = app.composer.capture_draft_snapshot()

    async with app.run_test() as pilot:
        await pilot.click("#console-prompts-improve-auto")
        await pilot.pause()

        assert app.composer.capture_draft_snapshot() == before
        assert "Prompt already looks good" in visible_text(app)
        assert len(app.screen.query("#console-undo-improvement")) == 0
```

- [ ] **Step 3: Add failing Review tests**

Review Apply replaces only improvable segments after stale/token validation; Cancel changes nothing. Protected token removal/duplication/edit blocks Apply. Editing Review text does not rewrite the system prompt.

- [ ] **Step 4: Add failing Structured tests**

Select Outcome-first/saved/Blank, fill locally from strict response, inspect editable blocks, unmatched content -> Additional context, and apply User by default. System context excluded prevents reconstruction from the captured session system; explicit role instructions in draft/starter may still fill System. Apply System remains unchecked.

- [ ] **Step 5: Add failing concurrency/cancellation/persistence tests**

Only one request per modal/session. Closing or Cancel shows Cancelling and ignores detached completion. Session/draft/system/provider/model changes prevent auto-apply and leave a reviewable copy. User+System validation occurs before either live mutation. If `ConsoleChatStore.set_session_system_prompt` reports persistence failure, keep the live value and show exactly `Applied to this session, but could not save to the conversation.` with Retry.

- [ ] **Step 6: Run and verify RED**

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_console_prompts_modal.py \
  Tests/UI/test_console_control_bar_actions.py \
  Tests/UI/test_console_composer_improvement_transaction.py \
  Tests/UI/test_console_native_chat_flow.py -q
```

- [ ] **Step 7: Implement request snapshot and worker lifecycle**

Capture the immutable composer snapshot before building model projection. Resolve provider once and display its provider/model. Use one named worker group and a monotonically increasing request ID. The callback re-checks all fingerprints before applying.

```python
def _start_improvement(self, mode: ImprovementMode) -> None:
    self._request_serial += 1
    request_id = f"prompt-improvement-{self._request_serial}"
    snapshot = self._snapshot_factory(mode, request_id)
    self._active_request_id = request_id
    self.run_worker(
        self._run_improvement(snapshot),
        group="console-prompt-improvement",
        exclusive=True,
        exit_on_error=False,
    )

async def _accept_improvement_outcome(self, outcome: PromptImprovementOutcome) -> None:
    if outcome.request_id != self._active_request_id:
        return
    if not self._live_fingerprints_match(outcome.request_id):
        self._open_reviewable_stale_copy(outcome)
        return
    await self._render_typed_outcome(outcome)
```

- [ ] **Step 8: Implement apply/undo/session coordination**

Never call normal send or append transcript rows. Apply the User draft through the composer transaction. If selected, apply System through the existing store method, then refresh rail/settings summary. Invalidate Undo on the documented events only. Do not bind Ctrl+Z unless a focused TextArea/native undo regression proves it is not stolen; a visible `Undo improvement` action is sufficient.

```python
def _apply_console_prompts_result(self, result: ConsolePromptsResult) -> None:
    self._validate_prompt_application(result)
    composer = self._console_composer_or_none()
    if composer is None:
        return
    if result.apply_user and result.user_text is not None:
        undo_snapshot = composer.apply_improvement(
            result.composer_snapshot,
            result.user_text,
        )
        self._console_improvement_undo = undo_snapshot
    if result.apply_system and result.system_text is not None:
        _session, persisted = self._ensure_console_chat_store().set_session_system_prompt(
            self._active_console_session_id(),
            result.system_text,
        )
        if not persisted:
            self._show_prompt_persistence_retry(result.system_text)
    self._sync_console_chat_core_state()
    self._sync_console_settings_summary()
```

- [ ] **Step 9: Run focused suites and commit**

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_console_prompts_modal.py \
  Tests/UI/test_console_composer_improvement_transaction.py \
  Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_command_composer.py \
  Tests/UI/test_console_prompt_picker.py -q
git diff --check
git add tldw_chatbook/Widgets/Console/console_prompt_improve_view.py tldw_chatbook/Widgets/Console/console_prompts_state.py tldw_chatbook/Widgets/Console/console_prompts_modal.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_prompts_modal.py Tests/UI/test_console_composer_improvement_transaction.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): add Auto Review and Recipe prompt improvement"
```

---

## Final Integration and Evidence

### Task 13: Run the end-to-end matrix, live TUI inspection, and close documentation

**Files:**

- Modify: `Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md` only for verified implementation deviations.
- Modify: `backlog/decisions/029-versioned-prompt-artifacts-and-safe-improvement-transactions.md` only if an architectural decision changed; otherwise link it unchanged.
- Create: `Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md`
- Add captures under: `Docs/superpowers/qa/console-prompt-improvement-2026-08/`
- Modify the stage Backlog task files with checked ACs and implementation notes.

- [ ] **Step 1: Run the focused Chatbook matrix**

```bash
.venv/bin/python -m pytest \
  Tests/Prompts_DB \
  Tests/Prompt_Management \
  Tests/Library/test_library_prompts_state.py \
  Tests/Library/test_prompt_export_roundtrip.py \
  Tests/UI/test_prompt_block_editor.py \
  Tests/UI/test_console_prompts_modal.py \
  Tests/UI/test_console_composer_improvement_transaction.py \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_sensitive_llm_logging.py -q
```

- [ ] **Step 2: Run static/import/style gates**

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python -c "import tldw_chatbook.app"
.venv/bin/python -m compileall -q tldw_chatbook/Prompt_Management tldw_chatbook/Widgets/Prompts tldw_chatbook/Widgets/Console
git diff --check
```

Run the repository's configured linter/type checker for every touched module if available. Do not claim it passed if the command is unavailable; record the fallback checks.

- [ ] **Step 3: Run the server compatibility matrix in its clean worktree**

```bash
.venv/bin/python -m pytest \
  tldw_Server_API/tests/Prompt_Management \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py -q
.venv/bin/python -m bandit -q \
  tldw_Server_API/app/core/DB_Management/Prompts_DB.py \
  tldw_Server_API/app/core/Prompt_Management/structured_prompts \
  tldw_Server_API/app/api/v1/schemas/prompt_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/prompts.py
git diff --check
```

- [ ] **Step 4: Perform real TUI verification with an isolated profile**

Use a temporary config/data profile and the real bundled stylesheet. Seed local legacy Prompt, v1 foreign structured Prompt, v2 block Prompt, v2 Recipe, malformed/future records, and enough results for pagination. Verify at 140x40, 100x30, and 80x24:

- composer stays width-bounded with one menu;
- top Workbench action order/wrapping;
- Local/Server Browse, pagination/search/error states;
- legacy/foreign/unsupported guards;
- block edit/reorder/validation without cursor loss;
- provider unavailable state;
- Auto no-change/success/Undo;
- Review protected-token veto;
- Recipe fill and optional System apply;
- cancellation and stale result;
- Library first-class Prompt/Recipe labels and save conflict.

Do not use a bare script that bypasses the app's test environment. Record the launch command, profile paths, terminal dimensions, observed behavior, and screenshots in the QA README.

- [ ] **Step 5: Inspect sensitive logs from the live run**

Search the isolated profile's logs for each planted canary. The count must be zero for prompt/system/block/placeholder/response canaries. Record permitted metadata examples without copying secrets or prompt content into the README.

- [ ] **Step 6: Self-review against every design requirement**

Create a traceability table in the QA README mapping spec sections 1-15 to implementation files and tests. Explicitly record:

- composer-unification prerequisite and reference commits;
- schema-v1 and `single_text_recipe` coexistence;
- server capability behavior on old/modern servers;
- exact Undo invalidation events;
- no hidden provider retries/repairs;
- no content in logs;
- no silent truncation;
- honest system persistence failure behavior.

- [ ] **Step 7: Complete Backlog and ADR hygiene**

Check every AC, add concise Implementation Notes naming both repository commits and verification results, link ADR-029, and set a task Done only when its repository-specific DoD is fully met. Commit each resolved stage task by its exact Backlog path when closing that stage; the IDs are created during Backlog Handoff and must not be guessed in this plan. If an implementation deviation changes storage, authority, or provider boundaries, update/supersede the ADR before marking Done.

- [ ] **Step 8: Final commit in Chatbook**

```bash
git add Docs/superpowers/qa/console-prompt-improvement-2026-08 Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md backlog/decisions/029-versioned-prompt-artifacts-and-safe-improvement-transactions.md
git diff --cached --check
git commit -m "docs(console): verify prompt improvement workbench"
```

Only stage spec/ADR files if they actually changed during implementation. Never use broad `backlog/tasks` staging in a dirty worktree.

## Completion Criteria

The feature is complete only when:

- Composer action unification is present and independently green before Prompt UI commits.
- Local and server migrations preserve every existing row and v1 behavior.
- Console block v2 and server `single_text_recipe` v2 coexist without ambiguous dispatch.
- Prompt/Recipe Browse, block editing, source-aware save, and Library integration work at supported terminal widths.
- Auto/Review/Recipe flows use the active Console provider/model and exactly one sensitive auxiliary call.
- Inline-file content/metadata and pending attachments never enter the improvement request.
- Apply/Undo preserve exact composer segments and stale/placeholder guards prevent unsafe mutation.
- Logs contain none of the planted prompt/system/block/placeholder/response canaries.
- Focused client/server tests, static checks, real TUI inspection, traceability, Backlog DoD, and ADR hygiene are all recorded.
