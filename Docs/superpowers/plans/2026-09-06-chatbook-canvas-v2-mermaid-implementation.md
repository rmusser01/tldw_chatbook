# Canvas V2 Mermaid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the approved offline Mermaid subset to versioned Canvas without weakening zero-egress execution or revision ownership.

**Architecture:** One process-lifetime profile snapshot selects exact compatibility; a pinned grammar, closed semantic adapter and bounded layout execute in QuickJS-WASM. The compiler emits inert declarations, the worker renders them once before authored scripts, and existing native/served owners retain revision, transport and bridge authority. Product admission stays off until qualification; development fixtures inject candidate snapshots only inside tests.

**Tech Stack:** Python >=3.11, Textual >=8.0.0,<9, SQLite, existing html5lib/tinycss2 compiler, QuickJS-WASM 0.32.0, pinned Mermaid 11.17.2 grammar, existing esbuild-wasm 0.25.9 vendor tooling, pytest and mandatory real Chromium/Playwright.

**Spec:** [Approved Canvas V2 design](../specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md).

ADR required: yes

ADR path: [ADR-124](../../../backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md), extending [ADR-121](../../../backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md).

Reason: Implement the accepted runtime/dependency, profile compatibility and policy-deployment contracts. No new architectural decision or database migration is authorized by this plan.

## Global Constraints

- Native system-browser and same-origin `--serve` delivery; no new public listener.
- No network, filesystem, cookies, persistent page storage, parent DOM or Chatbook APIs in generated execution. No native JS fallback, dynamic import, module loader, CDN or extra SVG/CSS privileges.
- One complete HTML document per revision. Preserve `canvas_create(title, html)` and `canvas_update(canvas_id, expected_parent_revision_id, html)` parameters, list/read tools, reversible mutations, branch selection, revisioned titles, Temporary ownership and atomic promotion.
- Profile IDs use the existing 64-byte safe-identifier contract. Reuse schema 68 and Canvas archive format 3.0; never import executable catalog bytes. No Canvas synchronization; that remains TASK-31003.
- Exact grammar/adapter/layout, engine build, facade/plan compatibility, Unicode rules/data and integrity are pinned. Changed pinned inputs require a new profile, including engine security fixes. Unknown/revoked profiles are source-only without substitution.
- Verified catalog/policy snapshots last for the process lifetime. Packaged updates require stopping/restarting native Chatbook or the served parent and all children. Browser refresh is insufficient. Existing explicit Canvas disable remains immediate live containment.
- Retain V1's 512 KiB source, 256 KiB evaluated-script bytes, 32 MiB guest heap, 512 KiB stack, 250 ms startup interrupt, 50 ms event interrupt, 100 pending-job drain cap, 1,800 DOM nodes, 900 CSS rules and 500 patches per operation. Preserve all existing timer/rate/bridge and worker-backstop limits.
- Library evaluation, all diagrams and authored scripts share those limits. No per-diagram counter reset or moving library evaluation outside startup accounting. Unexpected engine errors are not successful quota refusals.
- Diagrams render once before authored JS. All declarations prepare in a detached transaction; any startup failure discards that transaction. No observation/rerender API. Artifact staging/commit is independent of browser preview success.
- Default `monospace`, 16 CSS px, normal weight/style and 24 px line spacing. Intrinsic-size scrollable output; no automatic shrink-to-fit or clipped labels. Explicit authored CSS/JS overrides remain outside default-layout fidelity.
- Profile-aware authoring guidance is at most 8 KiB UTF-8 per effective profile, deduplicated in active model context; no automatic browser-error submission.
- Use repository pytest pre-import isolation and owned temporary data. Do not run ad hoc application imports against ambient user configuration. Do not run the full repository suite without user approval.
- Do not reformat or restructure unrelated code. New helpers belong to Canvas, not the unrelated Library widgets named `canvas`.

| Additional limit | Per diagram | Document |
| --- | --- | --- |
| Declarations | — | 4 |
| Decoded input | 8 KiB | 16 KiB |
| Flow nodes / edges | 16 / 24 | 24 / 32 |
| Participants / messages / notes | 6 / 16 / 8 | 8 / 24 / 12 |
| Individual label / all labels | 512 bytes / 4 KiB | 8 KiB all labels |
| SVG elements / serialized output | 250 / 48 KiB | 400 / 64 KiB |
| Layout work units | 10,000 | 20,000 |
| Width / height | 2,048 / 4,096 CSS px | Each diagram |
| Logical area | 4,194,304 square CSS px | 8,388,608 square CSS px |

These are conjunctive refusal ceilings, not a promise that every combination fits.
The accepted syntax/exclusions table in the spec is normative; do not widen it to
make fixtures pass. Freeze the final executable profile manifest only after all
qualification gates. A test candidate is not a shipped immutable version.

## Workspace, prerequisites and task execution

Planning baseline: `69b343df1` in the existing linked worktree
`.worktrees/canvas-v1`, branch `codex/canvas-v2-mermaid-design`. Do not touch the
dirty main checkout or remove existing worktrees/evidence. At implementation
start, inspect the branch/worktree and latest dev, preserve these documentation
commits and follow the selected execution skill's isolation procedure.

Read the spec, ADRs, owning Backlog file, and the Canvas incidents in
`backlog/docs/lessons-testing-evidence.md` (completion telemetry, real quota owner,
resource-limit controls) and `backlog/docs/lessons-live-verification.md`
(opaque-origin worker/CSP). The spike summary is historical evidence only.

Each task is one reviewable PR-sized slice, with no dependency on a later task.
Before coding a slice, use `backlog task edit ID -s "In Progress" -a @codex`,
then add that slice's implementation plan and ADR check to its task file. Verify
the actual file after CLI edits. To Do tasks intentionally have no Implementation
Plan/Notes yet. Execute checklist steps as small red/green cycles; split the
listed parameterized cases into separate cycles rather than one large edit.

Use `../../.venv/bin/python -m pytest` from this worktree after verifying that
interpreter exists; otherwise use the project's configured environment without
installing optional dependencies blindly. The commands below use that interpreter.
For changed Python files run the repository's configured Ruff checks and formatter
checks on those files only; for authored JS run `node --check FILE`. Run
`git diff --check`, inspect the exact diff and commit only the named task files,
tests, docs and Backlog record. Regenerate integrity manifests when owned runtime
assets change; never hand-edit output hashes to conceal a failed build.

## File and ownership map

| Area | New focused files | Existing integration owners |
| --- | --- | --- |
| Profiles | `Canvas/profiles.py`, `Canvas/static/profile-catalog.json` | `runtime_assets.py`, `models.py`, `limits.py` |
| Pinned library build | `scripts/vendor_canvas_mermaid.py`, `Canvas/mermaid/{inputs.json,semantic.js,text.js,entry.js}`, generated `Canvas/static/mermaid-subset.json` | Existing runtime vendor script, package-data configuration and notices |
| Layout | `Canvas/mermaid/{budget.js,flow_layout.js,sequence_layout.js,scene.js}` | No application/DB dependency |
| Compile/runtime | `Tests/Canvas/test_mermaid_compiler.py`, browser Mermaid fixtures | `compiler.py`, `compilation.py`, `models.py`, renderer and worker |
| Revision authority | Profile lifecycle tests in existing suites | `Chat/console_canvas_controller.py`, `Canvas/{service,staging,native_authority,archive}.py` |
| Delivery/policy | Snapshot identity tests in existing suites | `Canvas/{gateway,control_protocol,runtime_assets}.py`, `Web_Server/serve.py` |
| User/model workflow | `Canvas/authoring.py`, `Canvas/static/mermaid-authoring.txt` | Agent provider, message actions, chat screen/wiring, native and served shells |
| Qualification | `Docs/Canvas/V2_VERIFICATION.md` and owned fixtures | Existing zero-egress/native/served/archive/packaging tests and CI gate |

Paths in this table beginning `Canvas/` are under `tldw_chatbook/`.
The `mermaid/` directory contains build inputs, not a runtime module loader.
The installed JSON library artifact carries verified **inert source text**, not a
native JS module that executes the parser when imported by the browser.

### Task 1 / TASK-31934: Runtime profile snapshots and admission

**Files:** Create `tldw_chatbook/Canvas/profiles.py`,
`tldw_chatbook/Canvas/static/profile-catalog.json`,
`Tests/Canvas/test_profiles.py`, `Tests/Canvas/conftest.py`.
Modify `runtime_assets.py` and its existing tests; document the manifest schema
in `Docs/Canvas/V2_RUNTIME_COMPATIBILITY.md` (new).

**Interfaces:** Produce frozen `ProfileRecord(profile_id: str, manifest_sha256:
str, executable: bool, reason: str | None, library_bytes: int)`,
`ProfileSnapshot(build_id: str, policy_id: str, profiles: tuple[ProfileRecord,
...], default_diagram_profile: str | None)` and
`ProfileResolution(profile_id: str, executable: bool, reason: str | None)`.
`load_profile_snapshot() -> ProfileSnapshot` verifies packaged inputs and the
process owner retains its result. `resolve_profile(snapshot, *, operation,
parent_profile, has_diagrams) -> ProfileResolution`, where operation is one of
`create`, `update`, `rename`, `load`, has no I/O or mutation. A manifest validator
checks all pinned contract fields; the record is its bounded admission projection,
not a replacement for the verified full manifest/bytes owned by runtime assets.
Also produce `runtime_snapshot_id(snapshot: ProfileSnapshot) -> str`: SHA-256 of
canonical JSON containing build/policy IDs, the diagram default and sorted profile
record fields (including manifest hashes and execution/refusal policy). This is a
source-free cross-process identity, not a browser credential.

- [ ] Add this fixture to the new Canvas conftest and red tests in `test_profiles.py`:

  ```python
  import pytest
  from tldw_chatbook.Canvas.profiles import ProfileRecord, ProfileSnapshot

  @pytest.fixture
  def profile_snapshot():
      return ProfileSnapshot(
          build_id="a" * 64, policy_id="b" * 64,
          profiles=(
              ProfileRecord("canvas-v1", "c" * 64, True, None, 0),
              ProfileRecord("canvas-v2-mermaid-1", "d" * 64, True, None, 77817),
          ), default_diagram_profile="canvas-v2-mermaid-1",
      )
  ```

  ```python
  from tldw_chatbook.Canvas.profiles import resolve_profile

  def test_removing_diagrams_retains_parent_profile(profile_snapshot):
      result = resolve_profile(profile_snapshot, operation="update",
          parent_profile="canvas-v2-mermaid-1", has_diagrams=False)
      assert result.profile_id == "canvas-v2-mermaid-1"
      assert result.executable
  ```

- [ ] Run `../../.venv/bin/python -m pytest Tests/Canvas/test_profiles.py -q`.
  Initially fail because the profile module/API is absent, then implement and
  parameterize every transition in the spec. Keep unknown/retired/revoked rename
  and load identity intact but non-executable; updates refuse at their caller.

- [ ] Implement selection with this precedence, followed by exact manifest/policy
  lookup, never a version-range or latest-version fallback:

  ```python
  if operation in {"rename", "load"}:
      selected = parent_profile
  elif operation == "update" and parent_profile != "canvas-v1":
      selected = parent_profile
  elif has_diagrams:
      selected = snapshot.default_diagram_profile
  else:
      selected = "canvas-v1"
  ```

  Validate operations, missing required parents and candidate identity before
  returning. If no admitted diagram default exists, return bounded
  `profile-unavailable`, not V1. Catalog input is strict JSON with closed fields,
  unique safe IDs, lower-case SHA-256 identities and immutable verified byte
  inventory; cap input with the existing runtime manifest ceiling. Reject
  duplicate keys/IDs, unknown fields, ID reuse, tampered/missing files and claims
  that omit engine/facade/plan/Unicode/quota identity. No archive input is accepted.

- [ ] Add byte-tamper, duplicate-ID, retired profile and snapshot retention tests;
  rereading the same owned snapshot after changing a fixture file cannot change
  its bytes/identity. Keep V2 `executable=False` and default `None` in production.
  Test fixtures may construct candidate records; never add an environment bypass.
- [ ] Run profile/runtime-asset tests, targeted static checks and commit
  `feat(canvas): add immutable runtime profile admission` with the task record.

### Task 2 / TASK-31935: Reproducible grammar, Unicode and semantic admission

**Files:** Create the vendor/input/semantic/text/entry files in the ownership map,
`Tests/Canvas/test_mermaid_semantics.py`, `Tests/Canvas/mermaid_probe.mjs`,
`Tests/Canvas/mermaid_probe.py` and `Tests/Canvas/fixtures/mermaid/semantics.json`.
Modify `Tests/Canvas/test_runtime_assets.py`, notices and package-data rules.

**Interfaces:** The generated inert library JSON contains a closed inventory and
one source string whose evaluation in QuickJS returns private callable handles.
`parseMermaid(source: string, budget: DiagramBudget) -> DiagramModel` produces
either `{kind:"flow", direction:"TD"|"LR", nodes, edges}` or
`{kind:"sequence", participants, messages, notes}`. Flow nodes are
`{id,label,shape}` (`rect|rounded|diamond`); edges are `{from,to,label}`.
Participants are `{id,label}`; messages are `{from,to,label,dashed,order}`;
notes are `{side:"left"|"right"|"over",participants:string[],label,order}`.
Arrays preserve source order. IDs are plain strings, never object-prototype keys.
Task 3 extends the same private entry with layout; this task supplies bounded
parse accounting through `DiagramBudget` defined in `budget.js` here, including
input, label and semantic-count charge methods, then Task 3 adds layout charges.

The test-only `run_mermaid_case(case: dict) -> dict` in `mermaid_probe.py` invokes
Node with JSON stdin and an owned temporary environment. The Node harness imports
only trusted packaged QuickJS, verifies library bytes, creates a disposable VM
with existing limits, passes source using `newString`, calls the private parse
handle and returns `{ok,model,error}` as bounded JSON. No guest source goes to
native `eval`, `Function`, native modules or application imports. Each failed VM
is discarded, and host/API failures fail the test rather than becoming refusals.

- [ ] Add the fixture rows and a red test using the new helper:

  ```python
  import pytest
  from Tests.Canvas.mermaid_probe import run_mermaid_case

  @pytest.mark.parametrize("source", [
      "flowchart TD\nA[Start] --> B[Finish]",
      "sequenceDiagram\nparticipant A\nparticipant B\nA->>B: Hello",
  ])
  def test_admitted_models_use_quickjs(source):
      result = run_mermaid_case({"operation": "parse", "source": source})
      assert result["ok"] is True
      assert result["model"]["kind"] in {"flow", "sequence"}
  ```

- [ ] Run `../../.venv/bin/python -m pytest Tests/Canvas/test_mermaid_semantics.py -q`;
  expect the missing harness/library contract to fail before implementation.
  Build the harness with this execution boundary (all variables here are owned
  by its setup; `librarySource` is the verified inert JSON's source string):

  ```javascript
  const module = await newQuickJSWASMModule();
  const runtime = module.newRuntime();
  runtime.setMemoryLimit(32 * 1024 * 1024);
  runtime.setMaxStackSize(512 * 1024);
  runtime.removeModuleLoader();
  const deadline = performance.now() + 250;
  runtime.setInterruptHandler(() => performance.now() > deadline);
  const vm = runtime.newContext();
  const evaluated = vm.evalCode(librarySource, "canvas-mermaid-private.js");
  ```

  Check `evaluated.error` before using its value; dispose every argument/result
  handle and context/runtime in `finally`. Parent-process timeout is an independent
  backstop, not a successful expected parser refusal.

- [ ] Implement a separate stdlib-only vendor command following
  `scripts/vendor_canvas_runtime.py`'s `PackageSpec`, redirect refusal, archive
  member/type/path and integrity checks. Pin Mermaid 11.17.2 and its SRI from the
  retained spike report, exact grammar source files and build tools. Extract
  grammar source from exact `sources`/`sourcesContent` entries in the pinned maps:
  `dist/chunks/mermaid.core/chunk-SHT3W25Y.mjs.map` entry
  `../../../src/diagrams/flowchart/parser/flow.jison`, and
  `dist/chunks/mermaid.core/sequenceDiagram-WJ2MYXX4.mjs.map` entry
  `../../../src/diagrams/sequence/parser/sequenceDiagram.jison`. Both entries in
  the investigated package contain generated Jison 0.4.18 JavaScript, not raw
  grammar needing a new generator. Validate exact source hashes and expected
  exported parser shape; do not use comment-marker substring matching.
  If that closed extraction cannot be demonstrated, stop this slice for review;
  do not import the upstream renderer or promote the disposable extractor.

  Record exact Unicode segmentation/width data versions, hashes, notices and
  generation rules in `inputs.json`; use an explicitly pinned source input and
  deterministic table generation, not the machine's Unicode database or `Intl`.
  Download only declared build inputs; no npm lifecycle/install hooks. Regenerate
  twice into owned temporary output directories and compare all output hashes.

  Register the verified candidate descriptor in the development catalog with
  execution still refused and no diagram default. Its final immutable manifest
  is frozen only at qualification; no normal operation may create a revision
  referencing an unqualified candidate. Add a separate fixture for integration
  tests, retaining real manifest/asset hashes rather than Task 1's synthetic IDs:

  ```python
  from dataclasses import replace
  from tldw_chatbook.Canvas.profiles import load_profile_snapshot

  @pytest.fixture
  def candidate_snapshot():
      base = load_profile_snapshot()
      candidate = "canvas-v2-mermaid-1"
      assert any(row.profile_id == candidate for row in base.profiles)
      return replace(base,
          profiles=tuple(replace(row, executable=True, reason=None)
                         if row.profile_id == candidate else row
                         for row in base.profiles),
          default_diagram_profile=candidate)
  ```

  This is only in `Tests/Canvas/conftest.py`; no production bypass. Other test
  directories needing the fixture register that exact fixture explicitly in
  their test module rather than relying on sibling conftest discovery.

- [ ] Implement explicit semantic callbacks and token-level refusal for directives
  which upstream lexing could otherwise discard as comments. Reject every spec
  exclusion, conflicting nodes, duplicate participants, missing endpoints, cycles,
  unsupported shapes and IDs before layout. Do not normalize label code points or
  substitute empty labels for unsupported rich content. Errors use allowlisted
  codes, ordinal/line/column and no raw parser message or source snippet.
- [ ] Add separate positive/negative cases for each syntax row, comments versus
  init/front matter, HTML/Markdown labels, `__proto__`/constructor-like IDs, escaped
  label text, Unicode graphemes and input/count limits. Assert exact model values,
  not only success. Run semantic and vendor-integrity tests; commit
  `feat(canvas): vendor bounded Mermaid grammar and text rules`.

### Task 3 / TASK-31936: Bounded deterministic layout and scenes

**Files:** Create `flow_layout.js`, `sequence_layout.js`, `scene.js` in
`Canvas/mermaid/`; extend `budget.js`, `text.js`, `entry.js` and the generated
inert JSON. Add `Tests/Canvas/test_mermaid_layout.py` and layout fixtures beside
the semantics corpus. Update compatibility documentation and notices/hashes.

**Interfaces:** `layoutDiagram(model: DiagramModel, budget: DiagramBudget) ->
DiagramScene` returns `{width,height,root,metrics}`. A scene node has only
`{tag,attributes,text,children}` with existing allowlisted HTML/SVG tags and
attribute pairs; no IDs, executable source, URLs or markup strings. The worker
assigns private node identities in Task 4. `metrics` contains input/labels,
semantic counts, scene elements/bytes, work units and area. The private entry's
`renderDiagrams(records)` shares a document budget across models/scenes.

- [ ] Extend the probe operation to `layout` and add the branch/rejoin red test:

  ```python
  from Tests.Canvas.mermaid_probe import run_mermaid_case

  def test_branch_rejoin_geometry_is_deterministic():
      case = {"operation": "layout", "source":
          "flowchart TD\nA[Start] --> B{Ready?}\n"
          "B -->|Yes| C(Continue)\nB -->|No| D[Revise]\n"
          "C --> E[Join]\nD --> E\nE --> F[End]"}
      first = run_mermaid_case(case)
      assert first["ok"] is True
      assert first["scene"] == run_mermaid_case(case)["scene"]
      assert 0 < first["scene"]["width"] <= 2048
      assert 0 < first["scene"]["height"] <= 4096
  ```

- [ ] Run `../../.venv/bin/python -m pytest Tests/Canvas/test_mermaid_layout.py -q`;
  expect missing layout operation/output. Implement DAG ranking using Kahn's
  traversal with source-index tie ordering, then two bounded forward/backward
  ordering sweeps. Route orthogonal edges in deterministic rank lanes; charge
  every inspected node/edge/collision candidate before processing it. Reject
  cycles before allocating layout state. Start work charging with:

  ```javascript
  function chargeWork(budget, amount) {
    if (!Number.isSafeInteger(amount) || amount < 0) throw new Error("work-limit");
    if (budget.diagramWork + amount > 10000 || budget.documentWork + amount > 20000)
      throw new Error("work-limit");
    budget.diagramWork += amount;
    budget.documentWork += amount;
  }
  ```

  Capture private intrinsics and use null-prototype/Map records rather than
  authored-modifiable global helpers. Map internal errors to the bounded error
  vocabulary at the private entry; never expose arbitrary exceptions.

- [ ] Lay out sequence columns in declared order, events by combined message/note
  order, and reserve note rectangles/lane spacing before routing arrows. Render
  arrowheads with allowed geometric primitives rather than SVG marker references.
  Wrap labels on pinned grapheme boundaries using deterministic logical widths;
  no native measurements or fetched fonts. Emit ordinary HTML source/description
  and scroll wrapper, explicit typography and safe SVG text nodes.
- [ ] Add exact scene/model fixtures for TD/TB/LR, every shape/note position,
  dashed/solid messages, long/unbreakable labels, CJK, combining marks, emoji and
  RTL. Reject nonfinite/out-of-viewBox geometry and area overflow before patches.
  Parameterize every per-diagram/document cap independently. Four small diagrams
  and mixed flow/sequence must pass unchanged budgets; a failure is a design gate,
  not permission to increase limits. Run layout/semantic tests and reproducible
  generation; commit `feat(canvas): add deterministic bounded diagram layout`.

### Task 4 / TASK-31937: Closed V2 plans and transactional worker startup

**Files:** Modify `Canvas/{models,compiler,compilation}.py`, runtime worker,
renderer and asset manifests. Create `Tests/Canvas/test_mermaid_compiler.py` and
`Tests/Canvas/browser/test_canvas_mermaid.py`. Extend the existing zero-egress
harness asset inventory and private test helper support; do not relax assertions.

**Interfaces:** Add frozen `CanvasDiagram(ordinal:int, target_node_id:str,
kind:str, source:str)` and `CanvasRenderPlanV2` with V1's data fields plus
`diagrams: tuple[CanvasDiagram, ...]`; existing `CanvasRenderPlan` remains V1.
Use `CanvasCompiledPlan = CanvasRenderPlan | CanvasRenderPlanV2` internally.
Extend `compile_canvas_document(source, *, limits=None,
runtime_profile="canvas-v1", snapshot=None) -> CanvasCompiledPlan`; V2 requires
an exact verified snapshot. Add
`prepare_canvas_document(source, *, operation, parent_profile, snapshot) ->
CanvasCompiledPlan` to `compilation.py`, using one bounded HTML parse/structural
declaration inspection, Task 1's resolver and the selected-profile compiler.
Keep this pure work inside the existing compilation admission/executor owner.

- [ ] Add the source-preservation red test:

  ```python
  from tldw_chatbook.Canvas.compiler import compile_canvas_document

  def test_v2_declaration_is_data_not_script(candidate_snapshot):
      source = '<pre data-canvas-diagram="mermaid">flowchart TD\nA[Tea &amp; cake]</pre>'
      plan = compile_canvas_document(source,
          runtime_profile="canvas-v2-mermaid-1", snapshot=candidate_snapshot)
      assert plan.diagrams[0].source == 'flowchart TD\nA[Tea & cake]'
      assert plan.scripts == ()
      assert plan.source_identity.source_bytes == len(source.encode("utf-8"))
  ```

- [ ] Run `../../.venv/bin/python -m pytest Tests/Canvas/test_mermaid_compiler.py -q`;
  initially fail on the absent V2 keyword/model. Implement text-only pre extraction,
  nonempty declarations, target identity/uniqueness and byte/count admission.
  Unknown kinds, nested markup, excess declarations and combined library/authored
  script bytes fail with typed compiler errors. Default V1 compilation must still
  leave the attribute inert and emit its exact existing six-field wire format.

- [ ] Give V2 a closed eight-field wire schema: V1's six fields plus `diagrams`
  and `profile_manifest_sha256`. No `compatibility_issues` diagnostic enters the
  wire. Validate schema, manifest, exact source and diagram target/text records
  independently in Python, renderer and worker. Fetch only verified inert library
  data before execution acknowledgement; never import/evaluate it natively.

- [ ] Extend `runStartup` with private QuickJS handles using this ordering:

  ```text
  begin existing startup operation
  evaluate verified private library source in QuickJS (charged script bytes/time)
  parse and validate all declarations against one document budget
  prepare scenes, then apply them through virtual DOM typed mutations
  run authored scripts and drain bounded jobs
  validate and publish one startup transaction
  ```

  Assign unique virtual IDs through the existing allocator, not Mermaid IDs.
  Keep library handles outside authored globals. Use one private budget and
  transaction; failed authored scripts also discard diagram mutations. Never
  reset the deadline between library evaluation and generated script execution.

- [ ] Reuse `test_canvas_zero_egress.py`'s `_new_page`, `_load` and
  `_assert_zero_generated_egress` in the new browser tests with explicit imports.
  Extend `_wire_plan(source, *, runtime_profile="canvas-v1", snapshot=None)` in
  that test module to serialize the exact profile schema. Add tests asserting
  authored JS observes the diagram before its first statement, later declaration
  mutation does not rerender, an error in diagram 2 emits no diagram-1 startup
  output, and private handles are inaccessible. Drive the real worker/renderer
  with recorded network acknowledgements, not a native layout mock.
- [ ] Run compiler, asset, new browser and existing zero-egress suites; inspect
  skipped/failed outcomes. Regenerate trusted asset hashes and commit
  `feat(canvas): execute declarative diagrams in transactional QuickJS startup`.

### Task 5 / TASK-31938: Exact-profile production revision and archive ownership

**Files:** Modify `Chat/console_canvas_controller.py`,
`Canvas/{service,staging,native_authority}.py`; touch repository/archive code only
where exact-profile admission requires it. Extend
`Tests/Chat/test_console_canvas_controller.py`, `Tests/Canvas/{test_service,
test_staging,test_repository,test_native_authority,test_compiler_scheduling}.py`
and `Tests/Chatbooks/test_chatbook_canvas_round_trip.py`.

**Interfaces:** The existing controller accepts an internally injected
`profile_snapshot: ProfileSnapshot | None` (default loads the production snapshot
once). `prepare_import` and mutation preparation call `prepare_canvas_document`;
all staged metadata carries `plan.runtime_profile`. Existing public Canvas tool
and revision APIs remain unchanged. Exact stored reads call the selected-profile
compiler, never the create resolver. No additional mutation owner is introduced.

- [ ] Add a red production-controller test beside `_scope` in its existing suite:

  ```python
  def test_v2_revision_keeps_profile_after_removing_diagram(candidate_snapshot):
      controller = ConsoleCanvasController(profile_snapshot=candidate_snapshot)
      controller.register_run(_scope(), assistant_message_id=ASSISTANT_ID,
                              temporary=False)
      created = controller.create_canvas(_scope(), tool_call_id="create-v2",
          title="Flow", html='<pre data-canvas-diagram="mermaid">flowchart TD\nA[Start]</pre>')
      changed = controller.update_canvas(_scope(), tool_call_id="remove-diagram",
          canvas_id=created.revision.canvas_id,
          expected_parent_revision_id=created.revision.revision_id,
          html="<p>Summary</p>")
      assert changed.revision.runtime_profile == "canvas-v2-mermaid-1"
  ```

- [ ] Run that exact test with pytest; expect the absent snapshot keyword or V1
  hardcoded profile to fail. Replace production hardcodes at metadata/prepared-plan
  admission, not just standalone staging helpers. Preserve post-compilation owner,
  stale-parent, selection, run-replay and temporary-incarnation revalidation.
  The mutation gate must compare these identities before writing:

  ```python
  if plan.source_identity != CanvasSourceIdentity.from_source(html):
      raise CanvasLimitError("prepared-plan-source")
  if plan.runtime_profile != resolved.profile_id or not resolved.executable:
      raise CanvasLimitError("prepared-plan-profile")
  ```

  For metadata-only rename, copy the exact parent source/profile and perform
  ordinary revision/quota/ownership validation without executing an unavailable
  profile. Do not broaden the public HTML-update operation to an inert profile.

- [ ] Parameterize all spec transition rows through the actual provider/controller
  owner and native exact reads. Test V1-to-V2, removal, rename, old branch update,
  replay, concurrent stale completion, canceled turn and unavailable profile.
  Verify temporary close destroys staged state and promotion rollback/retry keeps
  every exact revision/profile in one transaction. Keep source-free projections.
- [ ] Extend real archive fixtures with V1/V2/sibling/rename/revoked/unknown rows;
  verify actual conversation and Chatbook exports, import-as-new and same-identity
  restore with remapped origins. Inject source/manifest/commit failures and assert
  no partial graph. Archives cannot install a library or change an executable
  snapshot. Existing schema, format and sync exclusion must remain unchanged.
- [ ] Run the named controller, Canvas ownership, archive and scheduling suites;
  commit `feat(canvas): preserve runtime profiles through revision lifecycles`.

### Task 6 / TASK-31939: Native and served snapshot delivery

**Files:** Modify `Canvas/{gateway,control_protocol,runtime_assets,native_authority}.py`,
`Web_Server/serve.py`; extend `Tests/Canvas/{test_gateway,test_control_protocol,
test_native_authority,test_canvas_kill_switch}.py`,
`Tests/Web_Server/{test_canvas_control_spawn,test_canvas_kill_switch}.py` and
native/served browser flows. Update `Web_Server/README.md`.

**Interfaces:** Task 1's `runtime_snapshot_id(snapshot)` produces the source-free
identity from the complete verified snapshot. Add this required identity to the authenticated private
child handshake. Bump `CONTROL_PROTOCOL_VERSION` to 2 because a closed auth
message changes; mixed versions fail closed without a legacy Canvas bypass.
Gateway/child must retain the same immutable verified snapshot/bytes for their
lifetime. Extend `CanvasControlBroker` and `CanvasControlClient` constructors with
the internal `runtime_snapshot_id: str` argument; normal ownership supplies it
from the verified snapshot, not an external browser request. Browser capabilities
stay scoped to existing exact session/load IDs.

- [ ] Add these red tests to `test_control_protocol.py`. Parameterize missing,
  malformed, mismatched, correct and old-version identities around these controls:

  ```python
  def test_v2_auth_codec_retains_snapshot_identity():
      message = ControlMessage(version=2, message_type="auth.request",
          request_id="auth-1", deadline_ms=None,
          payload={"child_id": "child-a", "secret": "a" * 64,
                   "runtime_snapshot_id": "b" * 64})
      decoded = decode_control_frame(encode_control_frame(message)[4:])
      assert decoded.payload["runtime_snapshot_id"] == "b" * 64

  def test_parent_refuses_foreign_snapshot():
      async def scenario():
          broker = CanvasControlBroker(runtime_snapshot_id="a" * 64)
          await broker.start()
          launch = broker.issue_child("child-a")
          client = CanvasControlClient(launch.environment,
                                       runtime_snapshot_id="b" * 64)
          try:
              with pytest.raises(ControlProtocolError,
                                 match="runtime_snapshot_mismatch"):
                  await client.start()
          finally:
              await client.aclose()
              await broker.aclose()
      asyncio.run(scenario())
  ```

  Import these already-existing protocol classes/codecs plus asyncio and pytest
  in the test module. Ensure terminal service remains alive in the served test.
  Run `../../.venv/bin/python -m pytest Tests/Canvas/test_control_protocol.py -q`;
  the new closed field/version must fail before implementation.

- [ ] Validate the handshake before accepting child scope. Implement the exact
  matching gate in the broker's authenticated attach path:

  ```python
  if child_snapshot_id != parent_snapshot_id:
      raise ControlProtocolError("runtime_snapshot_mismatch")
  ```

  Capture parent snapshot at server ownership initialization and child snapshot
  at its Canvas ownership initialization. Never accept child-supplied executable
  bytes or substitute the parent's profile for a child mismatch. Serve cached
  verified assets rather than rereading mutable files for individual routes.

- [ ] Run native and served tests with a candidate-injected snapshot; unknown,
  unavailable and revoked profiles expose source-only recovery with no renderer
  execution. Restart into a changed policy, verify old loads and pending bridge
  receipts cannot authorize effects, and verify browser refresh alone preserves
  the old process snapshot. Explicit disable still blanks/stops active previews.
- [ ] Exercise two real browser sessions against the served harness: copied source,
  renderer and action capabilities deny indistinguishably; branch/selection/load
  freshness survives delayed replies. Qualify the exact bootstrap asset closure
  under the unchanged production CSP and opaque iframe. No credentials/logged
  source, remote port, auth bypass or global conversation listing.
- [ ] Run the listed protocol/gateway/kill-switch/browser suites and commit
  `feat(canvas): fence browser delivery by immutable runtime snapshots`.

### Task 7 / TASK-31940: Mermaid actions, guidance and honest recovery

**Files:** Create `Canvas/authoring.py`, `Canvas/static/mermaid-authoring.txt`;
modify `Agents/canvas_tool_provider.py`, `Chat/console_message_actions.py`,
`UI/Screens/chat_screen.py`, `UI/Console_Modules/wiring.py`,
`Canvas/static/{canvas_shell.js,canvas_shell.html,canvas_shell.css}` and
`Web_Server/static/served_shell.js`. Extend provider/message-action/UI card tests
and native/served browser flows; update `Docs/User_Guide/console/canvas.md`.

**Interfaces:** `wrap_mermaid_document(source: str) -> str` creates a bounded
complete HTML document. `canvas_authoring_guide(snapshot, profile_id) -> str`
returns the exact profile's <=8 KiB guide or bounded unavailability guidance.
Deduplicate guides in the current tool/model context, not a new persistent ledger.
Extend existing block references with `language="html"` default, preserving old
HTML ordinal/identity semantics and using a distinct `canvas-mermaid` namespace.
Existing HTML fence replay must not change when a Mermaid fence precedes it.

- [ ] Add this red unit test in `Tests/Chat/test_console_message_actions.py`:

  ```python
  from tldw_chatbook.Canvas.authoring import wrap_mermaid_document

  def test_mermaid_wrapper_never_creates_script_markup():
      source = 'flowchart TD\nA[</pre><script>bad()</script>&]'
      html = wrap_mermaid_document(source)
      assert '<script>bad()' not in html
      assert '&lt;/pre&gt;&lt;script&gt;bad()&lt;/script&gt;&amp;' in html
      assert 'data-canvas-diagram="mermaid"' in html
  ```

- [ ] Run that exact test; implement wrapping without JS interpolation:

  ```python
  from html import escape

  def wrap_mermaid_document(source: str) -> str:
      return ('<!doctype html><html><head><meta charset="utf-8">'
              '<title>Diagram</title></head><body>'
              '<pre data-canvas-diagram="mermaid">' + escape(source, quote=False)
              + '</pre></body></html>')
  ```

  Add the existing UTF-8/declaration byte validation before constructing output.
  Parse source fences with the existing Markdown parser; do not inspect rendered
  Markdown. Test interleaved/repeated HTML/Mermaid blocks, stream-incomplete messages,
  changed-message rejection, replay and Open as new with existing origin rules.

- [ ] Supply accepted-syntax/exclusion/budget guidance with complete executable
  flow and sequence examples. Use the exact selected-profile resolver, not hardcoded
  V2 descriptions for V1 or unavailable historical revisions. Keep tool parameter
  JSON identical; extend safe result/context metadata only where needed and test
  each model, card, log and persistence projection with source sentinels.
- [ ] Keep source acceptance separate from browser status. Reuse load identity and
  nonce fencing for pending/ready/failed/source-only UI; on new failure do not show
  the old diagram under the new revision identity. Expose exact source and explicit
  View previous, show safe ordinal/code/location and allowlisted repair hints.
  Browser errors never cause automatic tool retries or chat submission. Confirmed
  repair inserts an unsent draft into the same current composer only.
- [ ] Test source copy/download (declarations are not standalone rendered HTML),
  auto-open preference, successful update notice, historical selection, Temporary
  badge, disable, renderer failure and stale status messages in actual shells.
  Run provider/action/UI/browser tests and commit
  `feat(canvas): expose Mermaid authoring and recoverable preview states`.

### Task 8 / TASK-31941: Qualification, immutable admission and release record

**Files:** Extend the existing Canvas browser adversarial fixtures, quota probe,
`Tests/Packaging/test_canvas_gateway_distribution.py`, runtime-asset reproducibility
tests and `.github/workflows/test.yml` with its contract tests in
`Tests/CI/test_github_actions_test_workflow.py`. Create
`Docs/Canvas/V2_VERIFICATION.md`; finalize compatibility/user/operator documentation
and the reviewed profile-catalog entry. The current workflow runs the broad
non-UI collection rather than naming the Canvas file: verify that the browser
selection is actually collected with its required dependencies, and add an
explicit focused gate if needed. Preserve unrelated lanes and required-browser
failure behavior; do not count an omitted browser suite as green.

**Interfaces:** Final immutable profile/manifest and process policy admit only the
exact qualified build. Test candidate construction stays test-only. No new tool,
schema, archive format or runtime permission is introduced by admission.

- [ ] Add production-catalog admission tests which initially assert the V2 entry
  is unavailable, plus real-browser cases for every release gate. After evidence
  is complete, switch the final admission assertion to the reviewed immutable
  identity and observe RED before changing the catalog. Never make an environment
  variable or archive manifest enable an unqualified candidate.

- [ ] Run targeted qualification against actual product paths:

  ```bash
  ../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Canvas/browser/test_canvas_zero_egress.py Tests/Canvas/browser/test_canvas_mermaid.py Tests/Canvas/browser/test_canvas_native_flow.py Tests/Canvas/browser/test_canvas_served_flow.py Tests/Canvas/browser/test_canvas_quota_probe.py
  ../../.venv/bin/python -m pytest -q --tb=short --show-capture=no Tests/Canvas Tests/Chat/test_console_canvas_controller.py Tests/Agents/test_canvas_tool_provider.py Tests/Chat/test_console_message_actions.py Tests/Chatbooks/test_chatbook_canvas_round_trip.py Tests/Packaging/test_canvas_gateway_distribution.py Tests/Web_Server/test_canvas_control_spawn.py Tests/Web_Server/test_canvas_kill_switch.py
  ```

  This is the Canvas-targeted selection, not the full repository suite. Respect
  required sandbox/network approval for owned loopback servers; do not turn denied
  binds or missing Chromium into skips or security passes. No actual provider or
  user database is required. Request separately any broader pre-merge sweep.

- [ ] Add diagram-specific hostile inputs to the existing HTTP/WebSocket/navigation/
  popup/download/worker observers and independent egress listener. Exercise private
  handle access, prototype pollution, label markup, directives, dense DAGs, large
  graphemes, parser exhaustion, multiple diagrams and startup script failure.
  Assert positive benign controls, expected typed refusals, worker termination and
  no post-start activity; do not classify arbitrary exceptions as quota success.
- [ ] Inspect screenshots and actual interactions for six-node branch/rejoin,
  three-party notes/messages, four small diagrams, mixed HTML+flow+sequence,
  Unicode/RTL, narrow scrolling, inherited CSS and explicit restyling. Verify
  source-only/previous/unsent recovery and new-Canvas migration after revocation.
  Record exact browser/platform versions and coverage gaps, cold startup,
  near-limit timing, byte/patch/work/heap counts and worker-backstop outcomes.
- [ ] Reproduce all vendored bytes twice, build wheel/sdist and verify package data,
  complete notices and offline execution. Re-run genuine archive round trips and
  disabled/restarted policy scenarios. Review stale-source/profile/policy admission
  after asynchronous compilation and browser delivery.
- [ ] Only after every required gate is green, freeze `canvas-v2-mermaid-1`'s exact
  manifest and admit it in the production catalog. If any semantic/limit/egress
  requirement fails, keep V2 unavailable and report the failing gate for design
  review. Run final targeted checks against the admitted build, record evidence,
  request independent code review under the review skill, and commit
  `feat(canvas): admit qualified offline Mermaid profile`.

## Coverage and closeout

| Spec area | Owning tasks |
| --- | --- |
| Pinned identity, revocation and defaults | 1, 6, 8 |
| Grammar, explicit subset refusal, Unicode | 2, 3, 8 |
| Layout, typography, geometry and aggregate quotas | 3, 4, 8 |
| Declarative lifecycle, V1 wire parity, zero egress | 4, 6, 8 |
| Durable/temporary revisions, branches, archive and sync exclusion | 5, 8 |
| Native/served authentication, snapshot and load fencing | 6, 8 |
| Fence actions, model guidance and source-private projections | 7, 8 |
| Honest save/preview state and confirmed recovery | 4, 5, 7, 8 |

Task acceptance includes targeted tests, relevant static/format checks, docs,
ADR links, self-review and implementation notes. Mark a task Done via Backlog CLI
only after its own deliverable is verified; the final admission task owns release
qualification. Preserve historical spike limitations and existing evidence.

This plan does not authorize PR creation, pushing, merging or starting a new Codex
task. Select subagent-driven or inline execution before beginning implementation.
