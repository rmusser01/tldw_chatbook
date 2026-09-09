# Local file-to-note Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author, save, recover, and execute a five-step local file-to-note workflow in the redesigned Textual Workflows destination, with human review, truthful results, and safe cancellation/recovery.

**Architecture:** Preserve the server API definition as the document; project it into a three-pane editor without a second graph serialization. An app-owned sequential runtime dispatches adapters through captured local services; a single run-state service owns durable transitions. Raw drafts, immutable revisions, immutable run manifests, and historical results have separate identities and lifecycles.

**Tech Stack:** Python >=3.11; Textual >=8.0.0,<9; SQLite, existing private-file helpers, Pydantic, stdlib JSON/asyncio/concurrent.futures, existing Chatbook domain services, pytest and Textual Pilot. No new framework, server dependency, or graph library.

**Spec:** [Approved Workflows design](../specs/2026-09-08-workflows-local-first-parity-design.md), approved by the user on 2026-09-08; [complete catalog disposition](../specs/2026-09-08-workflows-parity-matrix.md).

## Global Constraints

The following exact requirements are carried from the approved spec; all tasks inherit them.

- Sequential orchestration in v1, branching in v2, and parallelism in v3.
- Local execution without a connected tldw_server as a v1 requirement.
- One continuous editable form with independently collapsible sections.
- Draft writes are debounced to 500 ms; navigation, version selection, and orderly shutdown flush and await the current generation.
- Invalid JSON cannot update the structured projection, run, publish, or sync; the UI explains that forms still show the last valid structure and prevents form edits from overwriting the invalid buffer.
- Their resolved non-secret values are added to the run input object and private manifest, not the portable definition.
- Dynamic effects whose destination is not yet known cannot receive blanket pre-approval at initial preflight.
- Never place a secret in ordinary `inputs`, metadata, or a persisted run manifest to make an adapter work.
- Unresolved required expressions block the consuming operation rather than being sent literally to a model or tool.
- These counters survive pause/resume/retry and restart.
- Full test sweeps require user opt-in.

Approved local defaults, copied from the spec:

| Bound | Default |
| --- | --- |
| Active workflow execution | One active run per Chatbook workflow runtime, retaining its slot while its adapter has any live owned work; no automatic queue in the first milestone |
| Definition size | 100 authored steps and 2 MiB canonical JSON |
| Run inputs | 10 MiB serialized non-secret inputs |
| Attempts | At most 3 additional attempts per step |
| Active run time | 60 minutes cumulative execution time, excluding only parked waits/pauses with no active work |
| Model work | 100,000 input-plus-output token admission units per run |
| Inline outputs | 1 MiB serialized result per step; 100 MiB aggregate per run |
| Workflow-owned artifacts | 1 GiB per run; 5 GiB total retained workflow-owned artifacts |

Repository constraints also apply: use semantic Textual tokens, existing input/path/private-SQLite boundaries, workers for operations over 100 ms, canonical F9 Settings only, and ADR-031 keybindings. Never bind the reserved globals or terminal-convention Ctrl chords. Footer hints describe only working actions. AGPL licensing and optional-dependency gates remain unchanged.

ADR required: yes

ADR path: `backlog/decisions/138-portable-workflow-definitions-and-local-execution.md`

Reason: implements the accepted storage, runtime, long-lived UI, and interoperability architecture. No additional ADR is needed unless implementation changes those boundaries. The server sync protocol still requires counterpart agreement; this plan does not authorize a server implementation.

## Scope and evidence boundary

This is the **first local milestone**, not the whole v1 release. Its five supported operation subsets are `media_ingest` for one selected UTF-8 plaintext file, `prompt` for admitted dotted-path templates, `llm` for a configured local Ollama-compatible model, `wait_for_human` for the local actor's review/edited fields, and `notes` with `action=create` into the captured Notes database. Broader operations within those names remain visibly unsupported until tested. A registered type is not an execution guarantee.

The full 21-type v1 minimum remains unchanged. Additional adapters, reviewed file exchange, existing-server fetch/publish, and negotiated paired-server workflow sync each retain their own later v1 milestone. There is no branching, parallelism, nested workflow execution, scheduling, remote run executor, auto-queue, auto-sync, machine-wide budget manager, or artifact cleanup UI in this plan. Unsupported definitions/configuration remain preserved and inspectable. Do not expose non-working exchange/sync controls as actions.

For the initial adapters, artifact-emitting options are blocked, so the artifact budget stays zero rather than being falsely advertised as tested storage support. Inline-output bounds are still enforced. Domain-owned source files and notes never count as disposable workflow artifacts. Later artifact-producing adapters must implement the spec's bounded private artifact owner before admission.

Server evidence is pinned to dev commit `6cd2745f696af04668a61c20b84ab8a9e69ca5e4`. Inspect it with `git show <commit>:<path>` in the sibling tldw_server checkout, not that checkout's working files. Task 3 records operation-specific fixture provenance. Source inspection establishes shapes, not executed cross-server conformance. If the dev head changes, record the difference and review relevant contracts before replacing fixtures; do not silently retarget this plan.

Read before execution: root AGENTS.md; the current task file; ADR-011, ADR-029, ADR-031, ADR-033, ADR-036, ADR-068 (local research), ADR-126 and ADR-138; the relevant entries in `backlog/docs/lessons-testing-evidence.md`, `lessons-live-verification.md`, and `lessons-backlog-hygiene.md`. In particular, cancellation of `to_thread` does not terminate its physical worker, and a focused widget can still be offscreen.

## Workspace and task discipline

- This plan was prepared in a heavily dirty shared checkout. No application code or runtime tests were changed/run while planning. Establish an isolated execution worktree using the worktree skill; bring only these approved planning artifacts and explicitly selected prerequisites. Do not snapshot all unrelated changes.
- Reinspect every existing seam below in the chosen execution baseline. `Agents/execution_capacity.py` is concurrent untracked work in the planning checkout; this plan does not depend on copying it. Existing provider/tool constraints remain authoritative. If that capacity owner has landed, integrate its public model/tool reservation contract without replacing workflow-local admission.
- Backlog IDs were checked across 474 local branch/remote refs and 19 worktrees; the pre-allocation maxima were task 32087 and ADR 139. The CLI initially allocated task 32078; only that new record was moved to 32088. Recheck allocations before integration.
- Tasks stay **To Do** until execution. Before touching code for a task, assign it, set **In Progress**, and add its Implementation Plan (including ADR fields) through the Backlog CLI. Never refer to later task IDs from earlier task records.
- Each task has its own red/green cycle, scoped lint/format checks, self-review, and completion notes. Do not mark it Done while any acceptance or runtime evidence is missing. Do not turn a skipped live test into a passing milestone.
- Each task's final commit step is scoped to its listed files and task record, in the isolated worktree with an otherwise reviewed index. If that condition is not met, leave the edits uncommitted and report why; never include unrelated staged changes.

## File map, before task breakdown

Paths below are repository-relative **planned locations**, not claims that new files already exist. Create package `__init__.py` files only where needed; do not fill them with re-export frameworks.

| Files | Responsibility |
| --- | --- |
| `tldw_chatbook/Workflows/models.py`, `document_service.py` | Detached document/revision/draft values; lossless editing, identity, draft generations and save conflicts |
| `tldw_chatbook/DB/Workflows_DB.py`, `DB/migrations/workflows_v0_to_v1.sql`, `workflows_v1_to_v2.sql` | Private SQLite connection/migrations; v1 documents, v2 runs/attempts/events/waits/budgets/effect receipts |
| `Workflows/admission.py`, `expressions.py`, `limits.py`, `catalog.py` | Capability subsets, typed reference evaluation, binding ownership, safety inventory and bounded admission |
| `Workflows/adapters.py`, `local_services.py` | Five small adapters plus captured file/model/Notes/gate calls; no UI or run-state writes |
| `Workflows/run_service.py`, `runtime.py` | Single durable run-state writer; app-owned sequential dispatch, physical worker ownership, control and recovery |
| `Workflows/draft_session.py` | App-owned current draft buffer, debounce/flush and stale-completion protection across fresh screens |
| `UI/Workflows_Modules/library.py`, `navigator.py`, `editor.py`, `reference_picker.py`, `results.py`, `controller.py` | Focused pane widgets, continuous form, modal reference selection, run results and view coordination |
| Existing `UI/Screens/workflows_screen.py` | Compose the real destination; remove its placeholder cards without adding an alternative screen route |
| Existing `app.py`, `UI/Screens/settings_screen.py`, `config.py`, `UI/Console_Modules/` live-work handler | One composition/teardown path, canonical limits settings and exact workflow-run follow |
| Existing `DB/private_sqlite.py`, `backlog/docs/sqlite-private-owner-inventory.md`; new `backlog/docs/workflows-storage-owner.md` | Private owner registration and declared recovery/backup coverage; no invented complete-backup subsystem |
| `Tests/Workflows/`, `Tests/DB/test_workflows_db.py`, `Tests/UI/test_workflows_editor.py`, `test_workflows_runtime_integration.py` | Domain, race, persistence, production-adapter and actual Textual integration evidence |
| `Tests/fixtures/workflows/`, `Docs/User_Guide/workflows.md`, `Docs/superpowers/qa/workflows/local-file-to-note.md` | Pinned fixtures, truthful user documentation and measured milestone evidence |

Use screen/widget `DEFAULT_CSS` for new scoped styles under the repository's consolidated-CSS rules; do not invent a second CSS loader. Resolve the existing Console live-work handler by the `ConsoleLiveWorkLaunch` consumer before editing and list the exact owner file in Task 7's execution plan. It is actively being decomposed; broad edits to `chat_screen.py` are not the default.

## Shared interfaces and reference fixture

All new public types below live in `Workflows/models.py`. JSON strings are the immutable wire/storage boundary; decoding returns a fresh detached dictionary. They contain **no resolved credentials**. Use Pydantic validation at the JSON boundary, preserving unknown keys in the authoritative raw document; do not round-trip through a closed schema serializer that drops fields.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Revision:
    workflow_id: str
    revision_id: str
    parent_revision_ids: tuple[str, ...]
    raw_json: str

@dataclass(frozen=True)
class Draft:
    workflow_id: str
    base_revision_id: str
    generation: int
    raw_text: str
    last_valid_json: str
    error: str | None

@dataclass(frozen=True)
class Bindings:
    profile_id: str
    actor_id: str
    selections_json: str
    # Logical requirement -> selected non-secret values and verified resource IDs.

@dataclass(frozen=True)
class Issue:
    pointer: str
    code: str
    message: str

@dataclass(frozen=True)
class PreparedRun:
    revision: Revision
    bindings: Bindings
    inputs_json: str
    limits_json: str
    contract_revision: str

@dataclass(frozen=True)
class AdmissionReport:
    prepared: PreparedRun | None
    issues: tuple[Issue, ...]

@dataclass(frozen=True)
class StepRequest:
    run_id: str
    step_id: str
    attempt: int
    step_type: str
    config_json: str
    bindings: Bindings
    effect_id: str
    limits_json: str = "{}"

@dataclass(frozen=True)
class StepOutcome:
    kind: Literal["succeeded", "failed", "waiting_human", "needs_permission"]
    output_json: str = "{}"
    error_code: str | None = None
    retryable: bool = False
    replay_safe: bool = False
    uncertain_effect: bool = False

@dataclass(frozen=True)
class RunView:
    run_id: str
    workflow_id: str
    revision_id: str
    profile_id: str
    generation: int
    status: str
    step_id: str | None
    attempt: int
    snapshot_json: str
    manifest_json: str
    outputs_json: str
    wait_generation: int | None = None

@dataclass(frozen=True)
class RunEvent:
    kind: str
    payload_json: str

@dataclass(frozen=True)
class WaitDecision:
    run_id: str
    step_id: str
    wait_generation: int
    actor_id: str
    decision: Literal["approve", "reject"]
    edited_fields_json: str = "{}"
    comment: str = ""
```

Task 1 owns `Revision`, `Draft`, and document errors (`DraftConflict`, `InvalidDraft`, `RevisionConflict`, each a `ValueError` subclass). Task 2 adds binding/admission/request/outcome types. Task 4 adds run/event types; Task 5 adds `WaitDecision`. Do not define duplicate models in widgets.

`Bindings.selections_json` is a JSON object keyed by logical requirement name. Each selection has `kind`, `values` (exactly its owned input keys), and `resource` (non-secret verified identity and captured destination settings). A model selection's resource includes `endpoint`, `account_id: "keyless"`, and a complete `settings` object; a file selection includes its verified root/path and file identity; the local actor selection must match `Bindings.actor_id`. The private Notes destination selection has no portable input keys. Validation compares keys/kinds with the definition's requirements and rejects extras, ambiguous ownership and secret-valued fields. The JSON stores resource references, not transferable permission grants.

Task 1 creates `Tests/Workflows/helpers.py` with `prompt_definition() -> dict`, a fully valid two-prompt definition with stable test UUIDs, IDs `prepare` and `finish`, explicit `retry: 0` / `timeout_seconds: 300`, and `inputs: {"source_text": "hello"}`. Its `metadata.tldw_workflow` includes format version 1, UUID workflow/revision identities and empty parent IDs. Task 2 adds `local_bindings(tmp_path: Path) -> Bindings` and `prepared_prompt_run(tmp_path: Path) -> PreparedRun`; these use only temporary verified paths and keyless numeric-loopback model settings, never installed user config. The prepared-prompt helper adds a model requirement owning `summary_provider` and `summary_model`, then calls the real admission function; this makes the binding-collision test exercise an actually owned key. Task 3 adds the following exact five-step example to `Tests/fixtures/workflows/file_to_note.json` (the UUID namespace/input schema/requirements are supplied by the fixture, not synthesized differently at launch):

```json
[
  {"id":"ingest","type":"media_ingest","retry":0,"timeout_seconds":300,
   "config":{"sources":[{"uri":"{{ inputs.source_uri }}"}],"extraction":{"extract_text":true}}},
  {"id":"prepare","type":"prompt","retry":0,"timeout_seconds":300,
   "config":{"template":"Summarize in three bullets: {{ ingest.text }}"}},
  {"id":"summarize","type":"llm","retry":0,"timeout_seconds":300,
   "config":{"provider":"{{ inputs.summary_provider }}","model":"{{ inputs.summary_model }}","prompt":"{{ prepare.text }}","max_tokens":512}},
  {"id":"review","type":"wait_for_human","retry":0,"timeout_seconds":300,
   "config":{"instructions":"Review the summary before saving the note.","assigned_to_user_id":"{{ inputs.review_actor }}","timeout_seconds":3600}},
  {"id":"save","type":"notes","retry":0,"timeout_seconds":300,
   "config":{"action":"create","title":"{{ inputs.note_title }}","content":"{{ review.text }}"}}
]
```

Requirements own `source_uri`, `summary_provider`/`summary_model`, and `review_actor`; the Notes destination is a captured resource binding, not a path inserted into the definition. `note_title` is an ordinary explicit input. Human approval submits `edited_fields={"text": <reviewed summary>}`; it is not an implicit reference to a mutable screen buffer. Server source shows approved edited fields become the step output with `decision="approved"`; do not invent a `review.response` wrapper. The media adapter expects `sources: [{"uri": ...}]`, not a top-level `source_uri` config field.

## Task 1 — Lossless documents and durable drafts (TASK-32088)

**Files:** Create `Workflows/__init__.py`, `Workflows/models.py`, `Workflows/document_service.py`, `DB/Workflows_DB.py`, `DB/migrations/workflows_v0_to_v1.sql`, `Tests/Workflows/__init__.py`, `Tests/Workflows/helpers.py`, `Tests/Workflows/test_document_service.py`, `Tests/DB/test_workflows_db.py`, `backlog/docs/workflows-storage-owner.md`. Modify `DB/private_sqlite.py` and `backlog/docs/sqlite-private-owner-inventory.md`. All code paths in this and later task file lists are under `tldw_chatbook/` unless they begin `Tests/`, `Docs/`, or `backlog/`.

**Interfaces:**

- `WorkflowsDB(path: Path)` owns its connection/transaction context and `close() -> None`. Register owner `workflows.local` using the existing private-or-memory policy; call `connect_private_sqlite("workflows.local", path, isolation_level=None, check_same_thread=False)` and enable foreign keys. Every connection access is serialized by the store's reentrant lock; transactions are not interleaved across threads. A new store starts at schema 1; a newer unsupported schema refuses writable open.
- `DocumentService(db: WorkflowsDB)`, `create(raw_json: str) -> Revision`, `get_revision(workflow_id: str, revision_id: str) -> Revision`, `list_revisions(workflow_id: str) -> tuple[Revision, ...]`, `list_workflows() -> tuple[Revision, ...]`.
- `put_draft(workflow_id: str, base_revision_id: str, raw_text: str, generation: int) -> Draft`, `get_draft(workflow_id: str, base_revision_id: str) -> Draft | None`, `save_revision(workflow_id: str, base_revision_id: str, expected_generation: int) -> Revision`.

- [ ] Add the model declarations above and this failing test, importing JSON, pytest, `DocumentService`, `WorkflowsDB`, `InvalidDraft`, and `prompt_definition` from their declared modules:

```python
def test_invalid_draft_survives_reopen_without_replacing_projection(tmp_path):
    path = tmp_path / "workflows.sqlite3"
    db = WorkflowsDB(path)
    documents = DocumentService(db)
    original = prompt_definition()
    original["metadata"]["vendor_extension"] = {"opaque": [1, {"x": True}]}
    revision = documents.create(json.dumps(original))
    draft = documents.put_draft(
        revision.workflow_id, revision.revision_id, '{"steps": [', 1
    )
    assert draft.error is not None
    assert json.loads(draft.last_valid_json) == json.loads(revision.raw_json)
    with pytest.raises(InvalidDraft):
        documents.save_revision(revision.workflow_id, revision.revision_id, 1)
    db.close()
    reopened_db = WorkflowsDB(path)
    try:
        recovered = DocumentService(reopened_db).get_draft(
            revision.workflow_id, revision.revision_id
        )
        assert recovered == draft
    finally:
        reopened_db.close()
```

- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_document_service.py Tests/DB/test_workflows_db.py -q`; establish a meaningful failure before production implementation, not a missing optional dependency.
- [ ] Implement schema migration 0→1 and document transactions. Use the following key constraints; include `created_at`, JSON validity checks for valid snapshots, and named indexes for workflow history. Raw draft text intentionally has no JSON-validity constraint.

```sql
CREATE TABLE workflow_revisions (
    workflow_id TEXT NOT NULL,
    revision_id TEXT NOT NULL PRIMARY KEY,
    parents_json TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE workflow_heads (
    workflow_id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL REFERENCES workflow_revisions(revision_id)
);
CREATE TABLE workflow_drafts (
    workflow_id TEXT NOT NULL,
    base_revision_id TEXT NOT NULL REFERENCES workflow_revisions(revision_id),
    generation INTEGER NOT NULL CHECK (generation >= 0),
    raw_text TEXT NOT NULL,
    last_valid_json TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (workflow_id, base_revision_id)
);
```

- [ ] Preserve unknown JSON keys. For valid form changes update only the selected field in a detached document. New revisions get a new UUID and current base in `parent_revision_ids`; a compare-and-swap head update must succeed before commit. Reject a stale/equal-but-different draft generation; replay of the same generation/text is idempotent. Invalid buffers retain the prior valid projection and a bounded parse error without logging raw text.
- [ ] Add parameterized cases for unknown step/config/root metadata round-trips, duplicate step IDs, reserved expression-context IDs (`inputs`, `last`), absent identity initialization, conflicting head updates from two connections, failed transactions, new-schema refusal, and file/symlink/private-mode failures. Use real SQLite, including a file-backed reopen test; do not substitute a mocked connection.
- [ ] Declare the store's revisions, drafts, run history (schema 2), and future artifacts in `workflows-storage-owner.md`. Register private storage now. Current complete-recovery owner APIs are not present in this checkout: explicitly record **not yet integrated** and the ADR-126 inactive-restore requirement rather than fabricate an owner registration call. If those APIs land before execution, integrate their actual registry in this task and add its exact files/tests to the task plan before editing. Do not claim a complete backup while this owner is omitted.
- [ ] Rerun the two tests plus `Tests/DB/test_private_sqlite_inventory.py`; run scoped formatter/linter and inspect the entire task diff. Update task notes/AC and commit only this task's files with message `feat(workflows): persist lossless revisions and recoverable drafts`.

## Task 2 — Typed inputs, bindings and admission (TASK-32089)

**Files:** Create `Workflows/admission.py`, `expressions.py`, `limits.py`, `catalog.py`, `Tests/Workflows/test_admission.py`, `test_expressions.py`. Modify `Workflows/models.py`, `Tests/Workflows/helpers.py`.

**Interfaces:** Consumes `Revision` and detached definitions. Produces `Bindings`, `PreparedRun`, `AdmissionReport`, `Issue`, `StepRequest`, `StepOutcome`; `prepare_run(revision: Revision, overrides: dict, bindings: Bindings, limits: WorkflowLimits) -> AdmissionReport`; `resolve_value(value: object, context: dict) -> object` (raises `ExpressionError(ValueError)`); `validate_step(request: StepRequest) -> tuple[Issue, ...]`. `WorkflowLimits` is a frozen dataclass with fields/defaults in the following code and `to_json() -> str`. `catalog.py` holds an immutable tuple of explicit `StepContract(step_type: str, fields: tuple[str, ...], output_fields: tuple[str, ...], effects: tuple[str, ...])` records for the five subsets, not dynamic plugin registration.

- [ ] Add this regression test and the typed-expression tests before implementing admission:

```python
def test_binding_owned_input_cannot_be_overridden(tmp_path):
    revision = prepared_prompt_run(tmp_path).revision
    report = prepare_run(
        revision,
        {"summary_model": "unreviewed-model"},
        local_bindings(tmp_path),
        WorkflowLimits(),
    )
    assert report.prepared is None
    assert "binding_owned_input" in {issue.code for issue in report.issues}

@pytest.mark.parametrize("value", [None, True, 7, ["a"], {"nested": 1}])
def test_pure_reference_preserves_json_type(value):
    assert resolve_value("{{ inputs.value }}", {"inputs": {"value": value}}) == value
```

- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_admission.py Tests/Workflows/test_expressions.py -q` and confirm the intended red cases.
- [ ] Implement bounds with exact byte arithmetic, rejecting bool as an integer for numeric controls. Materialize inputs by rejecting any binding-owned key in defaults/overrides, shallow-merging ordinary inputs, then adding verified bound keys. An explicit null replaces a default. Unknown schema keywords requiring unsupported validation block readiness rather than being ignored.

```python
@dataclass(frozen=True)
class WorkflowLimits:
    active_runs: int = 1
    steps: int = 100
    definition_bytes: int = 2 * 1024 * 1024
    input_bytes: int = 10 * 1024 * 1024
    additional_attempts: int = 3
    active_seconds: int = 60 * 60
    token_units: int = 100_000
    step_output_bytes: int = 1024 * 1024
    run_output_bytes: int = 100 * 1024 * 1024
    run_artifact_bytes: int = 1024 * 1024 * 1024
    retained_artifact_bytes: int = 5 * 1024 * 1024 * 1024

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
```

- [ ] Implement the admitted expression grammar as dotted dictionary paths only; no attribute access, eval, Jinja statements, arbitrary filters, calls, or Python objects. Preserve values for pure expressions, render mixed strings, and recursively visit JSON lists/maps. Reject missing keys, forward references, reserved IDs, unsupported syntax and inferred unsafe type conversions with JSON-pointer issues. The `prompt` adapter receives already-resolved text; do not evaluate twice.
- [ ] Check the **whole document**: nonempty `on_completion_webhook`, branch/map/parallel/nested orchestration, unsupported routes and effectful unknown fields block local execution without modifying the revision. Accept only routes absent or the exactly equivalent next-step success chain; reordering with explicit routes is a reviewed atomic document edit. `retry` is the step-level count of additional attempts; `timeout_seconds` is the positive integer step deadline. Missing/null/zero/fractional imported deadlines require an explicit effective policy choice recorded in the manifest. Reject `config.retry` as an unimplemented execution control.
- [ ] Capture one local profile/actor, Notes identity, selected file identity/root, and numeric-loopback model endpoint. Require explicit binding selection; imported paths, credential references, and user IDs grant nothing. Do no model call, file content read, note write, remote fallback, dependency installation, or readiness network probe in pure admission. `validate_step` repeats config/type/effect checks after previous-output resolution immediately before dispatch. Unknown dynamic targets return a permission requirement.
- [ ] Add cases for input-key ownership collisions, mutable caller dictionaries, two destination bindings with unchanged revision JSON, callback refusal, unsupported-but-preserved definitions, all budget thresholds, missing explicit maximum model output, and unavailable local service capabilities. Run the two test files plus Task 1's document tests, then scoped lint/format/review. Commit with message `feat(workflows): validate typed bindings and local admission`.

## Task 3 — Five production adapter subsets (TASK-32090)

**Files:** Create `Workflows/adapters.py`, `local_services.py`, `Tests/Workflows/test_adapters.py`, `Tests/LLM_Calls/test_workflow_local_request.py`, `Tests/fixtures/workflows/file_to_note.json`, `server_contracts.json`, `README.md`. Modify `Workflows/catalog.py`, `Tests/Workflows/helpers.py`, `LLM_Calls/LLM_API_Calls_Local.py`, `Local_Ingestion/local_file_ingestion.py`, `Tests/Local_Ingestion/test_ingest_parse_worker.py`, and the bounded diagnostics paths in `Notes/Notes_Library.py` exercised by note creation. Reuse underlying parsing/transport/storage; do not build a second ingestion or provider pipeline.

**Interfaces:** Consumes `StepRequest`, `StepOutcome`, `validate_step`. Produces `LocalAdapters(services: LocalWorkflowServices)` with `execute(request: StepRequest) -> StepOutcome`, called only in an owned executor worker. `LocalWorkflowServices(notes: NotesInteropService, gate: BuiltinToolGate, model_call: Callable[[StepRequest], dict], file_read: Callable[[StepRequest], dict])` captures dependencies; `read_file(request: StepRequest) -> dict`, `call_model(request: StepRequest) -> dict`, `create_note(request: StepRequest) -> dict` enforce exact bound resources. Production factories `build_local_services(notes: NotesInteropService, gate: BuiltinToolGate) -> LocalWorkflowServices` supply the real functions, not test lambdas. `AdapterFailure(RuntimeError)` carries `code`, `retryable`, and `uncertain_effect`.

Additional narrow domain entry points owned by this task: `Local_Ingestion.local_file_ingestion.read_plaintext_for_workflow(file_fd: int, max_bytes: int) -> str` reads a caller-verified descriptor with a byte cap and strict UTF-8, reusing the plaintext normalization path without analysis/DB writes; `LLM_Calls.LLM_API_Calls_Local.chat_with_captured_local_model(*, endpoint: str, model: str, messages: list[dict], settings: dict, timeout_seconds: float, response_byte_limit: int) -> dict` invokes the existing OpenAI-compatible local transport with a complete non-secret settings snapshot, keyless decision, no automatic retries, no environment proxy and no redirects. These are new planned APIs, not currently existing functions.

- [ ] Build fixture metadata from these pinned files: server `adapters/media/ingest.py`, `adapters/control/flow.py`, `adapters/llm/llm.py`, `adapters/knowledge/crud.py`, `core/Workflows/engine.py` and `api/v1/endpoints/workflows.py`. Record accepted fields, ignored-vs-blocked fields, exact result keys, error codes and intentional safety restrictions. Verify `source_uri` and extraction behavior, required human assignee, and `notes` create return shape before checking the JSON example in. Server test-mode echoes are explicitly not execution evidence.
- [ ] Add deterministic tests for all five shapes; for example:

```python
def test_prompt_returns_canonical_text_without_a_model_call():
    services = LocalWorkflowServices(
        notes=Mock(), gate=Mock(), model_call=Mock(), file_read=Mock()
    )
    request = StepRequest(
        run_id="r1", step_id="prepare", attempt=1, step_type="prompt",
        config_json=json.dumps({"template": "Summarize: hello"}),
        bindings=Bindings("test-profile", "test-actor", "{}"), effect_id="e1",
    )
    outcome = LocalAdapters(services).execute(request)
    assert outcome.kind == "succeeded"
    assert json.loads(outcome.output_json) == {"text": "Summarize: hello"}
    services.model_call.assert_not_called()
    services.file_read.assert_not_called()
```

- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_adapters.py -q` and inspect the expected failing behaviors.
- [ ] Dispatch explicitly by type, validating before any call. Normalize error dictionaries as failures, never successful payloads. Use a complete `StepOutcome` for every adapter path; retained error messages use bounded codes, with private detail only in run results.

```python
def execute(self, request: StepRequest) -> StepOutcome:
    issues = validate_step(request)
    if issues:
        return StepOutcome("failed", error_code=issues[0].code)
    config = json.loads(request.config_json)
    if request.step_type == "prompt":
        return StepOutcome("succeeded", json.dumps({"text": config["template"]}))
    if request.step_type == "wait_for_human":
        return StepOutcome("waiting_human", json.dumps({
            "__status__": "waiting_human",
            "assigned_to": str(config["assigned_to_user_id"]),
        }))
    calls = {
        "media_ingest": self.services.read_file,
        "llm": self.services.call_model,
        "notes": self.services.create_note,
    }
    try:
        payload = calls[request.step_type](request)
    except AdapterFailure as error:
        return StepOutcome(
            "failed", error_code=error.code, retryable=error.retryable,
            uncertain_effect=error.uncertain_effect,
        )
    return StepOutcome("succeeded", json.dumps(payload, ensure_ascii=False))
```

The class constructor stores `self.services`. A permission ask is returned as `StepOutcome("needs_permission", ...)` by an explicit pre-dispatch gate before the shown service-call branch; it must not fall through to note creation. File capability denial and unsupported type are validation issues, not a `KeyError` dispatch path. Exceptions outside `AdapterFailure` are normalized by the runtime as non-retryable adapter failures; never log their unbounded message/body.

- [ ] Implement plaintext reads using `validate_path(..., redact_paths=True)` plus selected-file identity verification at open. A path string check alone is insufficient: reject symlink/identity replacement between preview and dispatch, or bind a verified descriptor. Reuse the plaintext parser only through a bounded read path with analysis/network/media persistence disabled. Return server-shaped `text`, `media_ids`, `metadata`, `transcripts`, `rag_indexed` keys for the accepted subset. Empty/error extraction does not advance the workflow. Stop before unbounded input allocation.
- [ ] For `llm`, implement the captured local entry point above around the existing `_chat_with_openai_compatible_local_server` transport in the same module. Inspection found that the current `chat_with_ollama` wrapper re-reads settings, and its dispatcher mappings do not establish request-pinned retries/timeouts/keylessness; merely passing those kwargs to `chat_api_call` would not prove enforcement. Do not add a second HTTP client or alter other providers' default behavior. This slice accepts only explicitly keyless local Ollama-compatible bindings; cloud or credential-requiring providers show Needs setup/Unsupported in this slice, not a silent keyless fallback. Normalize the provider envelope to the pinned `{"text": ...}` result contract and retain private usage metadata separately.
- [ ] Add optional keyword-only `response_byte_limit: int | None = None`, `allow_redirects: bool = True`, and `trust_env: bool = True` to the existing local transport helper. The captured wrapper passes a positive cap, False, False, and zero retries; all existing callers retain defaults. For non-streaming generation, use HTTP streamed transport (`stream=True` while payload `stream=False`) and bounded `iter_content` before JSON parsing. Bound error bodies too, close response/session on every path, reject redirects, and fail on excess bytes before concatenating. Streaming display is optional in this slice; non-streaming generation remains supported. Test a single overlong chunk, no Content-Length, lying Content-Length, stalled read, 307 redirect and proxy environment without sending real external traffic. Do not use a post-hoc result-length check as a memory bound.
- [ ] For `notes`, use captured `NotesInteropService.add_note(user_id, title, content, note_id)` and `get_note_by_id(user_id, note_id)`. Do not call `CreateNoteTool.execute()`, which reads mutable global config. Use its tool metadata with the real `BuiltinToolGate` for permission: denial wins, ask parks before writing, and an exact approved effect receives a one-call/run stamp immediately before `gate.check`. `effect_id` determines a stable UUID note ID; on a receipt gap, compare that exact note's title/content/owner before acknowledging a prior write. Mismatch or unverifiable outcome becomes Needs review, never overwrite or a second random ID.
- [ ] Add real temporary Notes DB cases (create/get/duplicate effect), permission deny/ask/changed approval cases, stale file identity, provider result/error envelopes, local binding drift, missing dependencies, and diagnostics capture proving input/result bodies absent on success and error. Run adapter and touched domain tests, lint/format and self-review. Commit with message `feat(workflows): add bounded file-to-note adapter subsets`.

## Task 4 — Sequential runtime, budgets and physical ownership (TASK-32091)

**Files:** Create `Workflows/run_service.py`, `runtime.py`, `DB/migrations/workflows_v1_to_v2.sql`, `Tests/Workflows/test_runtime.py`, `test_runtime_ownership.py`, `test_run_limits.py`. Modify `Workflows/models.py`, `DB/Workflows_DB.py`, `Tests/DB/test_workflows_db.py`, `Tests/Workflows/helpers.py`.

**Interfaces:**

- Consumes `PreparedRun`, `StepRequest`, `StepOutcome`, `resolve_value`, `LocalAdapters.execute`, and `WorkflowsDB`.
- `RunService(db: WorkflowsDB)` is the only public run-state writer. `launch(prepared: PreparedRun, operation_id: str) -> RunView`, `get(run_id: str) -> RunView`, `list_runs(workflow_id: str) -> tuple[RunView, ...]`, `commit(run_id: str, expected_generation: int, event: RunEvent) -> RunView`, `events(run_id: str, after_sequence: int = 0) -> tuple[RunEvent, ...]`. `StateConflict`, `LaunchConflict`, `CapacityUnavailable`, and `BudgetExceeded` are typed `RuntimeError` subclasses in `models.py`.
- `WorkflowRuntime(runs: RunService, adapters: LocalAdapters)` owns its executor, retained physical futures and app-loop observer tasks. `async launch(prepared: PreparedRun, operation_id: str) -> str`, `async request_control(run_id: str, action: Literal["pause", "resume", "cancel", "retry"]) -> RunView`, `async wait_until_settled(run_id: str) -> RunView`, `async shutdown() -> None`. Settled means parked, terminal, or needs-review **and no still-owned work**, not merely that a UI handler returned.
- Event kinds are a closed set: `attempt_reserved`, `attempt_started`, `attempt_finished`, `control_requested`, `wait_opened`, `wait_decided`, `policy_amended`, `recovery_recorded`. Each kind gets a validated payload schema, not arbitrary SQL patch fields. A persisted monotonic event sequence and run generation are distinct counters.

- [ ] Add duplicate-launch and immutable snapshot tests:

```python
async def test_duplicate_launch_returns_one_run_and_detached_snapshot(tmp_path):
    prepared = prepared_prompt_run(tmp_path)
    db = WorkflowsDB(tmp_path / "runs.sqlite3")
    runs = RunService(db)
    services = LocalWorkflowServices(Mock(), Mock(), Mock(), Mock())
    runtime = WorkflowRuntime(runs, LocalAdapters(services))
    try:
        first = await runtime.launch(prepared, "one-user-activation")
        second = await runtime.launch(prepared, "one-user-activation")
        assert first == second
        view = await runtime.wait_until_settled(first)
        assert view.status == "completed"
        assert json.loads(view.snapshot_json) == json.loads(prepared.revision.raw_json)
        assert len(runs.list_runs(prepared.revision.workflow_id)) == 1
    finally:
        await runtime.shutdown()
        db.close()
```

- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_runtime.py -q` and confirm red.
- [ ] Implement migration 1→2 with tables `workflow_runs`, `workflow_attempts`, `workflow_events`, `workflow_waits`, `workflow_budget_ledger`, `workflow_effects`, and a single workflow-runtime capacity row. Run IDs and launch IDs are UUIDs in production; a unique `(profile_id, operation_id)` plus canonical payload digest permits identical redelivery and rejects changed payloads. Persist launch before dispatch. Record raw snapshot, captured manifest, schema/adapter-contract revisions and all selected bindings. No resolved credential field is permitted.
- [ ] Encode transition invariants in `RunService.commit` under one transaction and generation check. An attempt is identified by `(run_id, step_id, attempt_number)`; a budget reservation and its attempt admission commit together. A cancelled/stopping run can record a late attempt outcome but cannot advance its step pointer. A run never points at a newer document revision. Validate event payload size before entering the transaction. UI/adapters must not receive the run database connection.
- [ ] Keep sync SQLite work off the UI loop using one serialized service-operation lane; guard connection ownership with that lane and a transaction lock. Capture the application event loop at runtime start; worker callbacks marshal to that loop rather than mutate UI/state directly. Do not put a long adapter call into the state-operation lane.
- [ ] Implement physical worker ownership with an owned `ThreadPoolExecutor(max_workers=1)` and retained `concurrent.futures.Future`, not only a cancellable asyncio wrapper. Use this await pattern inside the runtime; `_settle_physical` is a runtime method defined here with signature `async _settle_physical(request: StepRequest, future: Future[StepOutcome], awaited: asyncio.Future[StepOutcome]) -> None`, and commits the outcome and releases ownership only after `future.done()`. `async _commit_event(run_id: str, event: RunEvent) -> RunView` performs the current-generation read and commit inside the serialized state lane:

```python
future = self._executor.submit(self.adapters.execute, request)
self._physical[request.run_id] = future
wrapped = asyncio.wrap_future(future)
done, pending = await asyncio.wait({wrapped}, timeout=timeout_seconds)
if not done:
    await self._commit_event(request.run_id, RunEvent(
        "control_requested", json.dumps({"action": "timeout", "step_id": request.step_id})
    ))
# Retain this observer in self._observers; shutdown must drain it.
observer = asyncio.create_task(self._settle_physical(request, future, wrapped))
self._observers.add(observer)
observer.add_done_callback(self._observers.discard)
```

Call service methods through the serialized lane. `_settle_physical` shields the same `awaited` future from observer cancellation (do not create an unobserved second wrapper), records sanitized unexpected exceptions as failures, consumes every exception, then releases capacity. Install the observer from an exception-safe `finally` path after submission as well: cancellation or failure during the timeout transition cannot orphan an already-submitted worker. A user cancel request may mark Stopping immediately; it cannot destroy `future` or signal Stopped before the worker exits. Normal adapter completion and timeout/control checks race through one durable transition owner.

- [ ] Add an event/barrier-controlled adapter test: block inside a real executor thread, request cancel and then retry/second launch, assert refusal and one invocation; release the barrier, await settlement, assert late output remains attached to the cancelled attempt and the next step never starts. Repeat with timeout and both completion/cancel orderings. Avoid sleep-based race tests; use `threading.Event` and bounded waits.
- [ ] Persist cumulative active time and token/byte reservations before work. Estimate model input conservatively with the existing tokenizer when qualified; otherwise use a documented UTF-8-byte upper reservation plus explicit output maximum, never zero. Unknown usage keeps the reservation. Provider-level retries are disabled for this slice so they cannot bypass attempt accounting. Charge active time while work remains alive after timeout; checkpoint it, and conservatively account uncertain elapsed time after interruption. Exclude only parked periods with no worker.
- [ ] Add exact-boundary tests for all limits and two concurrent launch requests. Output limit failure preserves prior results and never truncates into a success. Require user review when the next attempt exceeds a limit; manual retry consumes the same ledger. Reject artifact requests before dispatch in this slice; do not create unaccounted files. Only retry a retryable failure when both budget and adapter replay/idempotency evidence permit it.
- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_runtime.py Tests/Workflows/test_runtime_ownership.py Tests/Workflows/test_run_limits.py Tests/DB/test_workflows_db.py -q`; run scoped lint/format and inspect worker shutdown/SQL transitions manually. Commit with message `feat(workflows): own sequential attempts and durable run budgets`.

## Task 5 — Durable human review, permission waits and recovery (TASK-32092)

**Files:** Modify `Workflows/models.py`, `run_service.py`, `runtime.py`, `local_services.py`; create `Tests/Workflows/test_waits.py`, `test_recovery.py`. Schema-2 wait/effect tables are introduced by Task 4; if an integrated schema change is needed now, increment to 3 with a new migration instead of editing a shipped migration.

**Interfaces:** Consumes `RunView`, `RunEvent`, `WorkflowRuntime` and captured service bindings. Produces `WaitDecision`; `async WorkflowRuntime.decide(decision: WaitDecision) -> RunView`, `async recover() -> tuple[RunView, ...]`, `async amend_limits(run_id: str, limits: WorkflowLimits, actor_id: str) -> RunView`. `StaleDecision` and `ReauthorizationRequired` are `RuntimeError` subclasses in `models.py`. Permission waits are distinct from authored `wait_for_human` steps and identify the exact effect payload digest.

- [ ] Add a direct state-machine regression, with `human_wait_run(tmp_path: Path) -> tuple[WorkflowsDB, RunService, WorkflowRuntime, str]` in `Tests/Workflows/helpers.py`: it starts a workflow whose first step waits for the test actor and returns only after the waiting state is persisted. All its paths are temporary and it uses real SQLite.

```python
async def test_approval_is_bound_to_wait_generation_and_actor(tmp_path):
    db, runs, runtime, run_id = await human_wait_run(tmp_path)
    try:
        waiting = runs.get(run_id)
        assert waiting.status == "waiting_human"
        wrong_actor = WaitDecision(
            run_id, waiting.step_id, waiting.wait_generation,
            "different-actor", "approve", '{"text":"reviewed"}',
        )
        with pytest.raises(StaleDecision):
            await runtime.decide(wrong_actor)
        assert runs.get(run_id).generation == waiting.generation
        decision = dataclasses.replace(wrong_actor, actor_id="test-actor")
        await runtime.decide(decision)
        with pytest.raises(StaleDecision):
            await runtime.decide(decision)
    finally:
        await runtime.shutdown()
        db.close()
```

- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_waits.py Tests/Workflows/test_recovery.py -q` and confirm the intended failures.
- [ ] Persist wait kind, run/step/attempt, generation, actor, review payload, effect digest and absolute UTC deadline. Engine attempt timeout and response timeout are separate fields. For human response waits, omitted/nonpositive config timeout is unlimited; positive timeout sets one absolute deadline. Check expiry in the same transaction as decision acceptance, not only through a UI timer. On approval the exact edited JSON fields plus canonical `decision="approved"` become that step's output; rejection never creates a note. The local actor mapping is destination-owned and cannot be chosen by an untrusted incoming user ID.

```sql
UPDATE workflow_waits
SET decision_json = ?, decided_at = ?, status = 'decided'
WHERE run_id = ? AND step_id = ? AND generation = ?
  AND actor_id = ? AND status = 'pending'
  AND (deadline_at IS NULL OR deadline_at > ?);
```

Check `rowcount == 1`; otherwise roll back with `StaleDecision`. Commit the output, event and run transition in that same transaction. Edited content receives the same type/output-size checks as adapter output before commit.

- [ ] Treat a permission wait as a gate for the exact resolved effect, not a future tool-wide grant. On resume, refresh `BuiltinToolGate` permission state; a new deny or kill switch wins. Bind the one-call stamp to the workflow run and immediately recheck its gate; changing destination/content invalidates the approval. Imported prior decisions never authorize current execution. Missing/changed captured account/resource identities require explicit reauthorization; the keyless local provider subset must not expand into credential handling by accident.
- [ ] Implement `recover()` as read/reconcile, not automatic execution. Expire overdue waits first. Persisted complete attempts remain complete; parked unexpired waits remain visible. A run interrupted while an adapter/effect was live becomes Needs review unless physical completion and its exact receipt are verified. Lease age or a stale PID alone does not establish that a worker/effect is safe to replay. A note receipt gap reconciles only the stable effect note ID and exact captured content/owner. A mismatch remains Needs review.
- [ ] For normal same-process resume, reacquire capacity and revalidate remaining bindings before dispatch without changing the saved definition or resetting ledgers. Record explicit increases through `policy_amended` with actor, prior/new limits and timestamp; never lower a limit below already-accounted usage without an explicit stopped state. Only Run again gets a fresh ledger and launch operation.
- [ ] Implement shutdown as idempotent admission-close → draft flush at the app coordinator → stop request → drain every physical future/observer → release leases → close executor/store. A parked human wait remains recoverable. If a worker does not terminate, show Stopping and retain ownership; do not claim graceful completion or close its database under it. Abrupt process termination recovers as uncertain, not cleanly stopped.
- [ ] Add file-backed close/reopen tests for deadline expiry (extend the constructor compatibly to `RunService(db: WorkflowsDB, clock: Callable[[], datetime] = utc_now)`; `utc_now() -> datetime` is a timezone-aware function in `run_service.py`), changed actor, duplicate approval, cancel-versus-approval, permission payload drift, note write-before-receipt interruption, conservative budget recovery and shutdown-before-DB-close ordering. Advance the injected clock instead of waiting for wall time. Run these files and Task 4's ownership tests, lint/format/review; commit with message `feat(workflows): recover waits and uncertain effects safely`.

## Task 6 — Three-pane authoring and continuous collapsed form (TASK-32093)

**Files:** Create `Workflows/draft_session.py`, `UI/Workflows_Modules/__init__.py`, `library.py`, `navigator.py`, `editor.py`, `reference_picker.py`, `controller.py`, `Tests/Workflows/test_draft_session.py`, `Tests/UI/test_workflows_editor.py`. Modify existing `UI/Screens/workflows_screen.py`; keep destination registration and canonical route `workflows` unchanged.

**Interfaces:**

- Consumes `DocumentService`, `Draft`, `Revision`, `AdmissionReport`, `Issue`, `catalog.py` and existing Textual form components.
- `DraftSession(documents: DocumentService)` owns buffers, generations and pending writes independently of screen lifetime. `async select(workflow_id: str, base_revision_id: str) -> Draft`, `update(raw_text: str) -> Draft`, `async flush() -> Draft`, `async save_revision() -> Revision`, `async close() -> None`. `DraftWriteFailed` is a typed `RuntimeError` in `models.py`; explicit discard is `async discard_pending() -> Draft`, called only after UI confirmation, restoring the last durable buffer rather than silently clearing history.
- `WorkflowEditor(Widget)` publishes field edits and renders one `Draft`; `show_step(step_id: str) -> None`, `show_issue(issue: Issue) -> None`. `ReferencePicker(ModalScreen[str | None])` returns the selected canonical expression or None. `WorkflowsController` coordinates these widgets and document service, never owns run execution. `visible_pane_ids(content_width: int) -> tuple[str, ...]` lives in `controller.py` and provides the existing app pane-navigation owner its eligible IDs.
- Stable selectors: `workflows-library`, `workflows-navigator`, `workflows-editor`, `workflow-raw-json`, `workflow-draft-status`, `workflow-step-selector`, `workflow-library-selector`, `workflow-save-revision`, `workflow-run`. Input bindings are data-driven by JSON pointer, not by brittle generated CSS selectors containing arbitrary step IDs.

- [ ] Add these first red tests. `WorkflowEditorHarness` in `Tests/UI/test_workflows_editor.py` extends `ConsolidatedCSSApp`, constructs real `WorkflowsScreen` with an app-like document/draft owner and existing route/focus dependencies, and exposes `draft_session`; it does not replace the production screen class.

```python
@pytest.mark.parametrize("width, expected", [
    (160, ("workflows-library", "workflows-navigator", "workflows-editor")),
    (110, ("workflows-navigator", "workflows-editor")),
    (60, ("workflows-editor",)),
])
def test_only_visible_panes_participate_in_focus_cycle(width, expected):
    assert visible_pane_ids(width) == expected

async def test_invalid_json_remains_after_navigation_and_reopen(tmp_path):
    harness = WorkflowEditorHarness(tmp_path)
    async with harness.run_test(size=(110, 36)) as pilot:
        editor = harness.screen.query_one("#workflow-raw-json", TextArea)
        editor.load_text('{"steps": [')
        await pilot.pause()
        await harness.draft_session.flush()
        assert harness.screen.query_one("#workflow-run", Button).disabled
        assert harness.draft_session.update('{"steps": [').error is not None
    reopened = WorkflowEditorHarness(tmp_path)
    async with reopened.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert reopened.screen.query_one("#workflow-raw-json", TextArea).text == '{"steps": ['
```

- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_draft_session.py Tests/UI/test_workflows_editor.py -q` and establish red.
- [ ] Implement app-owned draft debounce/flush. Every async write completion compares workflow/base/generation before publishing a Saved indication. A failed flush raises `DraftWriteFailed` and keeps navigation on the current buffer; a user may retry or explicitly acknowledge losing only pending changes. Passive parse validation never steals focus/cursor. Historical revision inspection flushes first and uses a separate read-only projection; editing that revision creates a deliberate new draft/revision path without replacing an unrelated dirty draft.
- [ ] Replace shell cards with Library → Navigator → Canvas. The navigator contains Overview, Inputs, Requirements, Versions, then ordered steps. Overview is a selectable linear list/box chain, not editable wiring. Focused step view shows compact previous/next previews with Enter/Space activation. Use one continuous scroll form with independently collapsed Parameters, Inputs/references, Execution controls and Advanced JSON sections; no horizontal carousel animation or modal per field.

```python
def visible_pane_ids(content_width: int) -> tuple[str, ...]:
    if content_width >= 132:
        return ("workflows-library", "workflows-navigator", "workflows-editor")
    if content_width >= 96:
        return ("workflows-navigator", "workflows-editor")
    return ("workflows-editor",)

with Collapsible(title="Parameters", collapsed=False):
    yield Label("Template")
    yield TextArea(id="workflow-template")
with Collapsible(title="Execution controls", collapsed=True):
    yield Label("Additional attempts")
    yield Input(value="0", type="integer", id="workflow-retry")
    yield Label("Step timeout (seconds)")
    yield Input(value="300", type="integer", id="workflow-timeout")
```

Measure thresholds against usable content width, not terminal width before app chrome. Library collapses first, then navigator. At 60x20 use labeled selectors to open the hidden regions as overlays, preserving opener and cursor. Set hidden regions `display=False` and exclude them from the existing global F6 pane traversal; do not bind F6 locally. A failed width/focus assumption is a test failure, not a reason to lower the minimum size.
- [ ] Build type-specific fields for the five supported subsets and show other catalog names only under explicit Show all, labeled unavailable. Use friendly Render text (`prompt`) and Call model (`llm`) labels. Implement add/rename/duplicate/reorder/delete step with stable IDs, dependency validation, and guarded destructive actions. Reorder preview lists affected earlier-step references and explicit success routes; do not silently rewrite semantic dependencies. Preserve unsupported config in Advanced JSON and block destructive structured editing where preservation is not proven.
- [ ] Implement Fixed value / Workflow input / Earlier step output reference selection. Filter choices by declared type and earlier step ordering; historical result insertion is explicitly Copy value, never live data. Opening/dismissing a picker restores an eligible opener or stable-ID replacement, then selector fallback. Explicit issue activation expands the section and scrolls/focuses its precise field; passive issues do not. Old results do not change the selected editing card unless Follow execution is enabled.
- [ ] Add actual Textual tests at 160x48, 110x36 and 60x20: long form scroll, nested collapsed field error reveal, pointer and Enter/Space neighbor activation, failed flush navigation, raw invalid buffer reopening, revision switching with dirty draft, resize with focused hidden control, selector dismissal, no hidden Tab/F6 stop, printable shortcut safety inside TextArea/Input and footer truthfulness. Assert compositor visibility in addition to focus. Capture screenshots of the real screen for review; do not substitute a browser mockup.
- [ ] Run the two new test files and the Workflows cases in `Tests/UI/test_destination_shells.py`; scoped lint/format and UI self-review. Commit with message `feat(workflows): author workflows in a recoverable three-pane editor`.

## Task 7 — App-owned execution, result provenance and Console follow (TASK-32094)

**Files:** Create `UI/Workflows_Modules/results.py`, `Tests/UI/test_workflows_runtime_integration.py`; modify `app.py`, `config.py`, `UI/Screens/settings_screen.py`, `UI/Screens/workflows_screen.py`, `UI/Workflows_Modules/controller.py`, `UI/Console_Modules/workspace.py`, the existing exact live-work action consumer in `UI/Screens/chat_screen.py` if it still owns workflow dispatch, and `Tests/UI/test_console_live_work_handoffs.py`. Keep any Console addition a delegation to the existing owner, not a new workflow engine in Console.

**Interfaces:** Consumes `DraftSession`, `prepare_run`, `WorkflowRuntime`, `RunService.list_runs/get/events`, and existing `ConsoleLiveWorkLaunch`/`PendingHandoffStore`. Produces one app-composed `workflow_documents`, `workflow_drafts`, `workflow_runs`, and `workflow_runtime` per local profile lifetime; `_wire_workflow_services() -> None` called once after dependencies exist. `WorkflowResults.show_run(view: RunView) -> None` displays the exact identity. `WorkflowsController.async run_selected() -> str`, `async follow_run(run_id: str) -> None`, `async navigate_away() -> bool` are the UI coordination entry points.

- [ ] Add a production-app regression, using `_build_test_app` from `Tests/UI/app_factory.py`, real isolated stores and controlled adapters injected at the service boundary. Add a helper `open_workflows(app, pilot) -> WorkflowsScreen` in this test module that sends the existing `NavigateToScreen("workflows")` message and waits for that actual mounted screen, never manually constructing a different route.

```python
async def test_editing_draft_does_not_relabel_selected_run(tmp_path):
    app = _build_test_app()
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await open_workflows(app, pilot)
        prepared = prepared_prompt_run(tmp_path)
        revision = app.workflow_documents.create(prepared.revision.raw_json)
        prepared = dataclasses.replace(prepared, revision=revision)
        await app.workflow_drafts.select(
            prepared.revision.workflow_id, prepared.revision.revision_id
        )
        run_id = await app.workflow_runtime.launch(prepared, "launch-original")
        original = await app.workflow_runtime.wait_until_settled(run_id)
        await screen.controller.follow_run(run_id)
        app.workflow_drafts.update('{"name":"unfinished"')
        await pilot.pause()
        view = app.workflow_runs.get(run_id)
        assert view.revision_id == original.revision_id
        assert view.snapshot_json == original.snapshot_json
        assert run_id in str(screen.query_one("#workflow-run-identity", Static).render())
        assert "Different draft" in str(screen.query_one("#workflow-result-context", Static).render())
```

- [ ] Run `.venv/bin/python -m pytest Tests/UI/test_workflows_runtime_integration.py -q` and establish the intended failure.
- [ ] Compose document/store/run/runtime dependencies exactly once in `TldwCli`, following ADR-036. Capture Notes, permission service and local provider binding owners at launch. Do not cache a `WorkflowsScreen` or read its widget state from the runtime. Register durable draft flush and runtime drain in `_shutdown_app_owned_lifecycles` before dependent DB/services close; failure must retain ownership and explicit recovery state. Navigation constructs a fresh screen, with selection-only snapshots in `ScreenStateStore`; durable draft content remains private in the document owner.
- [ ] Implement Run as flush → structural revision save/selection → input/binding form → preflight → explicit side-effect review → persisted operation identity → runtime launch. Persist the operation token before dispatch and reuse it on double activation or delivery retry. Disable only genuinely unavailable actions; do not keep Save blocked merely because dependencies prevent Run. Unsaved draft vs run revision is explicit. Run again is a separate deliberate new operation.
- [ ] Mount the results drawer with workflow/revision/run/target/step/attempt identity, status, bounded output, errors and review controls. Show Stopping while physical work remains; show Needs review for uncertain effects; show Older run / Different draft where appropriate. Inspect run definition is a read-only snapshot mode that preserves the editor draft. Remember selected run per workflow without inventing a default result when none exists. Stale async event/query results must match current run identity and UI generation before repainting.

```python
launch = ConsoleLiveWorkLaunch.from_values(
    source="workflows",
    title="Workflow run",
    payload={"run_id": view.run_id, "workflow_id": view.workflow_id,
             "revision_id": view.revision_id, "target": "local"},
    status=view.status,
    recovery="Inspect this exact workflow run; execution remains owned by Workflows.",
)
```

Stage the payload through the existing revisioned handoff channel. Console follows that exact run with the same run service and offers its supported controls; it must not substitute the latest Home active-work item, start another model conversation, or take ownership of execution. A missing run is an explicit stale target. Do not put source text/results/credentials into the handoff's metadata preview.
- [ ] Add `[workflows]` default limits in canonical config and numeric controls on F9 Settings only. Saving defaults affects new runs; changing an existing run uses the explicit amendment method with review. Do not allow settings to increase the number of active runs beyond this milestone's single-run capability. File paths continue following existing restart-bound storage settings; no hot profile migration is added.
- [ ] Add tests for screen leave/revisit during execution, runtime-source change without authority drift, double Run, real full-app route mounting, exact Console handoff/stale target, hidden-screen waits and decisions, results restoration across workflows, active worker shutdown ordering, settings/default-vs-amendment behavior and action/footer truthfulness. Run new integration tests plus `Tests/UI/test_console_live_work_handoffs.py -k workflow`, `Tests/UI/test_destination_shells.py -k workflow`, and applicable settings tests. Lint/format/self-review; commit with message `feat(workflows): connect owned runs and exact result follow`.

## Task 8 — Qualify the complete local milestone (TASK-32095)

**Files:** Create `Tests/Workflows/test_file_to_note.py`, `Tests/Workflows/test_file_to_note_live.py`, `Docs/User_Guide/workflows.md`, `Docs/superpowers/qa/workflows/local-file-to-note.md`; modify fixture README, add a separate qualified-subset evidence section to the parity matrix without changing its catalog dispositions, and add narrowly scoped regressions/source fixes only if a discovered defect is inside these acceptance criteria. Record any such source change in the execution task before editing it.

**Interfaces:** Consumes the exact production screen, app-composed services and file-to-note fixture. No new production service interfaces. `run_file_to_note(app, pilot, source: Path, model: str) -> RunView` is a helper in `Tests/Workflows/test_file_to_note.py` that drives the actual authoring/run/binding/review controls, rather than calling the engine directly. `assert_isolated_paths(app, root: Path) -> None` in that module checks workflow/Notes/config/data paths are descendants of the pytest-owned scratch root before any durable effect.

- [ ] Add a deterministic full-path test with a controlled model-call boundary and real temporary file/SQLite notes. Drive workflow creation, parameter editing, Save revision, mappings, Run, human edit/approval and saved-note inspection through Textual. This test proves wiring, not real model availability. Add a second run rejected at human review and assert no new note.

```python
async def test_file_to_note_uses_approved_text_and_one_note(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("A small local source about workflow ownership.", encoding="utf-8")
    app = _build_test_app()
    # Inject only the controlled model call; keep production file, Notes and gates.
    app.workflow_runtime.adapters.services.model_call = lambda request: {"text": "Draft summary"}
    async with app.run_test(size=(110, 36)) as pilot:
        result = await run_file_to_note(app, pilot, source, "test-local-model")
        assert result.status == "completed"
        outputs = json.loads(result.outputs_json)
        assert outputs["review"]["text"] == "Reviewed summary"
        assert outputs["save"]["success"] is True
        assert outputs["save"]["note"]["content"] == "Reviewed summary"
```

- [ ] Run `.venv/bin/python -m pytest Tests/Workflows/test_file_to_note.py -q` and confirm it exposes a broken/missing path before fixing the relevant code; do not stub the production screen, runtime, file adapter or Notes write to make it pass.
- [ ] Add the opt-in live test using `@pytest.mark.loopback_network`, **not** unrestricted `live`/`allow_network`. Read a task-specific `TLDW_WORKFLOW_TEST_MODEL` environment value; skip with a precise missing-prerequisite reason when unset, and do not classify that skip as milestone success. Require a preinstalled local model and numeric loopback endpoint; do not pull models, install dependencies, call tldw_server or enable external internet as part of verification. Disable simulated/test-mode adapter branches for this path. Existing pytest isolation owns the profile; do not manually repurpose HOME or CODEX_HOME.

```python
@pytest.mark.loopback_network
async def test_real_local_model_file_to_note(tmp_path):
    model = os.environ.get("TLDW_WORKFLOW_TEST_MODEL")
    if not model:
        pytest.skip("Set TLDW_WORKFLOW_TEST_MODEL to a preinstalled local model")
    source = tmp_path / "source.txt"
    source.write_text("Workflow runs keep their original bindings.", encoding="utf-8")
    app = _build_test_app()
    async with app.run_test(size=(110, 36)) as pilot:
        result = await run_file_to_note(app, pilot, source, model)
        assert result.status == "completed"
        outputs = json.loads(result.outputs_json)
        assert outputs["summarize"]["text"].strip()
        assert outputs["save"]["note"]["id"]
```

The helper performs `assert_isolated_paths` before Run and verifies the configured service identities; `_build_test_app` alone is not proof that every new store is isolated. Apply an exact endpoint allowlist around the existing network guard for this test: permit only the selected model host/port, reject/record all other loopback ports (including tldw_server) and all external destinations. An attempted forbidden connection that is swallowed by application code still fails verification. Do not claim unavailable server proof merely because no server credentials were supplied.
- [ ] Verify restart with a persisted human wait in one fresh app process/profile and explicit resume in a second; use an actual local model run before parking, retain its output, and confirm no repeated model call after restart. Run separate controlled barrier tests for cancellation/timeout during a physical worker and note effect receipt interruption; do not force-crash the user's normal profile. Show that a new draft, different workflow selection or Console navigation does not retarget the run.
- [ ] Capture actual Textual screenshots at 160x48, 110x36 and 60x20: empty library, focused continuous form, invalid draft, preflight issue, human review, stopping/needs-review, and completed note. Include keyboard-only walkthrough results, viewport dimensions, code revision, dependency/model versions, real model endpoint identity (no credential), isolated path proof, test commands and test counts in the QA document. Never record source bodies, credentials or user's unrelated data in persistent diagnostics/evidence.
- [ ] Run only the plan's targeted files and touched seam tests. Select the repository's installed formatter/linter (confirm executables/config before choosing commands); do not add a new tooling dependency or reformat entire existing modules to manufacture a green lint result. Explicitly distinguish pre-existing lint failures from introduced issues. Self-review the final scoped diff, permission transitions, worker races and UI screenshots against the approved design.
- [ ] Update user docs with the supported five operation subsets, local prerequisites, limit behavior, recoverable draft interval, exact run provenance, and honest recovery/backup limitations. Preserve all 130 catalog entries and the 21-step target. Record Local subset tested only for operations actually exercised; do not replace this with a full-parity or Synced badge.
- [ ] Mark the milestone task Done only after the real local-model path and required UI/recovery evidence pass. If the model or native environment is unavailable, retain the task In Progress with the precise missing gate; passing deterministic tests may be reported separately. Commit the reviewed changes with message `test(workflows): qualify local file-to-note milestone`.

## Self-review and execution handoff

Coverage map for this **first slice**:

| Approved design requirement | Owning tasks |
| --- | --- |
| Lossless portable definition, immutable revisions, invalid drafts, private storage | 1, 6 |
| Typed input precedence, resource ownership, exact control fields, unsupported import preservation, callback refusal | 2, 3 |
| Actual local five-step chain, captured services, permission and note idempotency | 3, 5, 8 |
| Whole-run target snapshot, single writer, physical ownership, budgets, duplicate launch | 4, 5, 7 |
| Durable waits, stale decisions, uncertain effect recovery, orderly shutdown | 5, 7, 8 |
| Three-pane/linear overview/focused card/continuous collapsed form, narrow focus, typed picker | 6, 8 |
| Exact result provenance, historical inspection, Console follow, truthful controls | 7, 8 |
| Real SQLite, barrier races, isolated production app, offline local model and screenshots | 1–8 |
| Full 21 adapters, file/server exchange, negotiated paired-server sync | Explicit later v1 milestones; not claimed by this plan |
| Branching / parallelism | v2 / v3; preserved but not admitted here |

Planning review checks: verify all existing Modify/Read paths; parse embedded JSON; compile complete Python examples; check exact shared type/method spelling; scan for unfinished placeholders; verify all eight Backlog records/AC/dependencies, design approval and ADR links. Some implementation steps deliberately fail closed when an existing service cannot meet its safety contract; they do not authorize weakening the bound or claiming verification.

Execution is a separate approval boundary. Recommended: subagent-driven execution with one task at a time and review between tasks; inline execution with the executing-plans skill is also supported. No application implementation, merge or full test sweep is authorized by preparation of this plan alone.
