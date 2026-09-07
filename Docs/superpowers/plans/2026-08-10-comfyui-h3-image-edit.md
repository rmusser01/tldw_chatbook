# ComfyUI MiniMax H3 Static Image Edit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one sanitized MiniMax H3 ComfyUI image-edit workflow that accepts exactly one staged image and one raw user instruction, returns exactly one persistent PNG through the existing Image Generation boundary, and remains correct across cancellation and Console remounts.

**Architecture:** Add a strict packaged-workflow `ComfyUIImageAdapter` under `Image_Generation`; extend the existing request, result, reference-image, adapter-registry, settings, attachment, and metadata contracts only where the approved design requires it. Keep protocol transport adapter-local, make the app own active edit operations and byte-free completion cleanup across fresh Console screens, and never import or route through `Video_Generation`.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Pillow, httpx through existing egress policy, stdlib `dataclasses`/`threading`/`asyncio`/`json`/`secrets`/`importlib.resources`, ComfyUI API-format JSON, setuptools wheel/sdist packaging, Backlog.md.

---

## Governing Contract

- Task: `backlog/tasks/task-3402 - H3-static-image-edit-through-Image_Generation.md`
- Design: `Docs/superpowers/specs/2026-08-10-comfyui-h3-image-edit-design.md`
- ADR required: yes
- ADR path: `backlog/decisions/052-comfyui-h3-image-edit-provider-boundary.md`
- Reason: TASK-3402 introduces an image-provider runtime, exact-origin trust, cancellation/lifecycle semantics, and a cross-module persistence contract. ADR-052 already records those decisions; do not create a duplicate ADR.

## Global Safety and Evidence Rules

- Treat the externally supplied API graph as read-only, private input. Never add it to the repository, name it in tracked files, print its path, print its embedded values, record its hash, or include it in a report or commit message.
- Produce the repository asset only in a validated private temporary directory, apply exactly the four approved source-to-sanitized changes, compare the graphs privately, and copy only the sanitized result into the package.
- The only tracked replacement instruction is `Apply the requested image edit.`; the only tracked replacement upload placeholder is `h3_edit_input.png`; the only tracked output prefix is `h3_edit`.
- Never stage with `git add .` or `git add -A`. Stage exact paths and inspect `git diff --cached --name-only` before every commit.
- Do not use the Video Generation adapter, configuration, workflow resource directory, store, metadata, publication gate, or playback path.
- Do not add a generic arbitrary-workflow setting or extract a shared ComfyUI transport abstraction.
- Do not add batch syntax, a separate negative-prompt input, style composition, context/LLM fallback, dimensions, model selection, CFG, or alternate image formats to the ComfyUI image command.
- Run only tests named by this plan and directly related to touched files. Do not run the full suite, a full collection, broad RuntimePolicy, or unrelated Video Generation tests.
- Before each test block, resolve the main repository's existing interpreter and validate it in that same foreground shell; do not assume variables persist between calls:

  ```bash
  task3402_git_common=$(git rev-parse --path-format=absolute --git-common-dir)
  test -n "$task3402_git_common"
  TASK3402_PYTHON="${task3402_git_common%/.git}/.venv/bin/python"
  test -x "$TASK3402_PYTHON"
  ```
- Every new guard needs a focused mutation proof: break the guarded line or branch, run the exact named test and observe RED, then restore it exactly and rerun GREEN.
- User copy and logs use stable phases and exception types only. No raw exception text, traceback in user copy, instruction, path, original filename, upload name, prompt ID, descriptor, response body, loader/model filename, or media bytes.
- Live UAT uses the user-configured trusted origin, isolated local state, a synthetic temporary PNG, and an unrecorded neutral instruction. Tracked evidence is prompt-free, host-free, descriptor-free, credential-free, and media-free.

## Required Execution Preflight

Before Task 1 changes any implementation or test file, the implementing agent must read these files completely:

```bash
backlog task 3402 --plain
sed -n '1,760p' Docs/superpowers/specs/2026-08-10-comfyui-h3-image-edit-design.md
sed -n '1,320p' backlog/decisions/052-comfyui-h3-image-edit-provider-boundary.md
sed -n '1,1785p' backlog/docs/lessons-testing-evidence.md
sed -n '1,640p' backlog/docs/lessons-live-verification.md
sed -n '1,320p' backlog/docs/lessons-backlog-hygiene.md
```

Expected: TASK-3402 is In Progress, ADR-052 is Accepted, and the executor understands the repository's mutation-evidence, live-profile-isolation, exact-path staging, and Backlog CLI traps.

The external graph is handed only to the Task 1 executor as runtime input from the user's existing file attachment. The primary agent must either execute the private sanitization block itself or provide the exact path directly to that one executor's foreground shell. The path must never be placed in the plan, task, reviewer prompt, ignored report, environment file, fixture, terminal transcript excerpt, commit message, or Git object. If the executor does not have the attachment at runtime, it stops Task 1 and asks the user to reattach it; it must not search broad user directories or guess a path.

## Mandatory Review Gate After Every Implementation Task

Tasks 1–6 are sequential because later tasks depend on earlier contracts. After each task's GREEN/static checks and exact-path commit, do not start the next task until both reviews below approve the complete task diff:

1. **Spec-compliance review:** a fresh reviewer receives only the task number, its implementation-base and HEAD commits, this plan, the approved design, ADR-052, TASK-3402, and the task's ignored evidence report. It verifies every required behavior, RED/GREEN/mutation evidence, scope, privacy, and focused-test restriction.
2. **Code-quality review:** a separate fresh reviewer receives the same bounded diff and checks correctness, lifecycle/concurrency behavior, security/privacy, maintainability, and non-vacuous tests. It must inspect real signatures at crossed seams rather than trusting matching fakes.

Validated findings are fixed by the implementing agent with a new focused RED→GREEN cycle and a separate exact-path commit. Re-run both reviews on the full task range after fixes. Stop and ask the user if a task exceeds three review rounds or retains an Important finding. This gate applies equally to Subagent-Driven and Inline Execution.

## File and Responsibility Map

### Packaged workflow and distribution

- Create `tldw_chatbook/Image_Generation/workflows/minimax_h3_image_edit.json` — sole packaged H3 image-edit graph. Its logical packaged workflow key is the extensionless `minimax_h3_image_edit`; the `.json` suffix belongs only to the resource filename and is not accepted as a second loader key.
- Create `Tests/Image_Generation/test_comfyui_workflow_assets.py` — independent exact node/class/link/literal/output inventory and hygiene assertions.
- Create `Tests/Image_Generation/test_comfyui_workflow_distribution.py` — wheel/sdist inventory plus fresh-wheel installed-resource load.
- Modify `pyproject.toml` — include only `Image_Generation/workflows/*.json` as Image Generation package data.
- Modify `MANIFEST.in` — include the same resource in the sdist.

### Image Generation contracts and runtime

- Modify `tldw_chatbook/Image_Generation/adapters/base.py` — optional request cancellation event and backward-compatible result effective parameters.
- Modify `tldw_chatbook/Image_Generation/exceptions.py` — typed cancellation and phase-coded ComfyUI image-edit failure.
- Modify `tldw_chatbook/Image_Generation/capabilities.py` — backward-compatible `required` reference capability and ComfyUI capability.
- Modify `tldw_chatbook/Image_Generation/request_validation.py` — required-reference enforcement and bounded MIME/signature/decode/mode/dimension/pixel validation.
- Modify `tldw_chatbook/Image_Generation/worker.py` — thread cancellation into requests and preserve the single validation choke point.
- Modify `tldw_chatbook/Image_Generation/adapter_registry.py` — lazy ComfyUI registration and runtime reset participation.
- Modify `tldw_chatbook/Image_Generation/config.py` — independent `[image_generation.comfyui]` settings and config+registry runtime reset.
- Modify `tldw_chatbook/Image_Generation/listing.py` — local-only explicit-opt-in configurability and reference-input capability.
- Modify `tldw_chatbook/config.py` — commented independent ComfyUI Image Generation defaults/disclosure; no LAN host default.

### Strict ComfyUI adapter

- Create `tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py` — confined resource loader, topology preparation, bounded same-origin API exchange, output validation, effective metadata, timeout, and prompt-scoped cancellation.
- Modify `tldw_chatbook/Image_Generation/adapters/__init__.py` only if the package's public adapter export convention requires it; lazy registry loading remains authoritative.
- Create `Tests/Image_Generation/test_comfyui_image_adapter.py` — local preparation, remote preflight, transport, cancellation, output, privacy, and immutability tests.

### Canonical F9 settings

- Modify `tldw_chatbook/UI/Screens/settings_image_gen_defaults.py` — ComfyUI backend row, curated fields, local probe/validation, packaged-default placeholders, retention/transmission disclosure data.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py` — render/save the fields and reset Image Generation config+registry only after successful persistence.
- Modify `Tests/UI/test_settings_image_gen_defaults.py` — pure schema/diff/configuration checks.
- Modify `Tests/UI/test_settings_image_gen_panel.py` — rendered fields, explicit consent, disclosure, successful-save reset, failed-save no-reset, and compact layout checks.

### Console source ownership, persistence, and lifecycle

- Modify `tldw_chatbook/Chat/attachment_core.py` — non-persisted random `attachment_id` default.
- Modify `tldw_chatbook/Chat/console_chat_store.py` — exact-ID pending consumption and idempotent persisted-message merge through the existing persistence read boundary.
- Modify `tldw_chatbook/Chat/console_generate_image.py` — ComfyUI count-one rule, same cancellation event, typed cancellation re-raise, allowlisted result metadata mapping.
- Create `tldw_chatbook/Chat/console_image_edit_operations.py` — app-owned per-session operation registry and byte-free completion-cleanup records; no Textual widgets and no media bytes.
- Modify `tldw_chatbook/app.py` — construct/own the registry for the app lifetime.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py` — H3-specific preparation before generic prompt work, in-memory reference construction, Stop/unmount behavior, shielded durable append, identity-gated attachment/draft cleanup, remount reconciliation, sanitized error mapping, and Regenerate refusal.
- Modify `Tests/Chat/test_attachment_core.py` — opaque ID creation/nonserialization characterization.
- Modify `Tests/Chat/test_console_generation_store.py` — exact pending consumption and exact persisted-message hydration.
- Modify `Tests/Chat/test_console_generate_image.py` — batch count, cancellation, event identity, and metadata allowlist.
- Modify `Tests/Chat/test_console_generation_actions.py` — real command and Regenerate behavior.
- Modify `Tests/UI/test_console_pending_attachment_stash.py` — completion/stash ordering across fresh screen stores.
- Create `Tests/Chat/test_console_h3_image_edit.py` — focused end-to-end Console operation, cancellation, persistence, remount, draft, privacy, and no-video-path contract.

### UAT and closeout

- Create `Docs/superpowers/qa/2026-08-10-comfyui-h3-image-edit-uat.md` — structural, sanitized real-server evidence only.
- Modify TASK-3402 — final AC, Implementation Notes, exact focused evidence, ADR link, and Done status only after every gate and live UAT passes.
- Write ignored execution reports under `.superpowers/sdd/2026-08-10-comfyui-h3-image-edit/`; do not stage them.

---

### Task 1: Package the Sanitized Workflow and Prove Distribution

**Files:**
- Create: `tldw_chatbook/Image_Generation/workflows/minimax_h3_image_edit.json`
- Create: `Tests/Image_Generation/test_comfyui_workflow_assets.py`
- Create: `Tests/Image_Generation/test_comfyui_workflow_distribution.py`
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`

**Interfaces:**
- Consumes: the approved exact node/class/link and four-change allowlist in design §5.
- Produces: one package-resource-confined logical key, `minimax_h3_image_edit`, mapped internally to the sole resource filename `minimax_h3_image_edit.json`, loadable from source, wheel, and sdist.

- [ ] **Step 1: Reconfirm provenance boundaries without revealing source identity**

In one foreground shell, resolve the external source from the active user attachment without echoing it. Verify it is a non-symlink regular file, record no hash, and search Git only using the repository-owned destination key and prohibited raw-export patterns—not the private source basename.

```bash
git ls-files | rg 'tldw_chatbook/Image_Generation/workflows/' || true
git log --all --name-only --pretty=format: | rg '(^|/)(raw|original|source|backup).*h3.*\.json$' && exit 1 || true
git status --short
```

Expected: no private source artifact is tracked; only the task status/plan docs may be dirty before implementation begins.

Record the current `HEAD` as one exact line, `implementation_base=<full-commit-hash>`, in the ignored Task 1 report. Validate the value with `git cat-file -e "${task3402_base}^{commit}"` immediately after writing it. Do not substitute a guessed hash later; Task 7 reads this exact recorded value for its whole-implementation scope audit.

- [ ] **Step 2: Write independent asset topology tests**

Create `test_comfyui_workflow_assets.py` with constants independently transcribed from design §5.2, not derived from the graph under test:

- exact 18-node ID → `class_type` map;
- exact 24 destination-input → source-link map;
- exactly one output-class node, `165: SaveImage`;
- nodes `154` and `166` absent;
- exact neutral literals at `114.image`, `133.prompt`, and `165.filename_prefix`;
- controlled literals limited to `114.image`, `125.sampler_name`, `126.steps`, `131.noise_seed`, `133.prompt`, and `165.filename_prefix`;
- node 165 linked only to `149[0]`, which restores dimensions from node 150's original-source path;
- no unexpected node, output, control, title/class decoy, source prompt, path-like provenance field, or second workflow resource.

Do not embed any private source value in the tests. Assert the resource directory inventory equals `{"minimax_h3_image_edit.json"}`.

- [ ] **Step 3: Write the distribution test**

Build wheel and sdist into `tmp_path`, never the repository. Assert both archives contain exactly one `tldw_chatbook/Image_Generation/workflows/*.json` member and it has the stable filename. Install the wheel into a second `tmp_path`, remove the checkout from import resolution, import the installed adapter, call its confined loader, and assert the exact topology constants.

- [ ] **Step 4: Run the focused asset/distribution tests and capture RED**

```bash
"$TASK3402_PYTHON" -B -m pytest \
  Tests/Image_Generation/test_comfyui_workflow_assets.py \
  Tests/Image_Generation/test_comfyui_workflow_distribution.py -q
```

Expected RED: resource missing from source/wheel/sdist and the adapter loader absent. Record the exact failure set in the ignored Task 1 report.

- [ ] **Step 5: Sanitize only in a validated private temporary directory**

In one foreground shell, obtain the external path only through the Required Execution Preflight handoff, assign it to `TASK3402_SOURCE_GRAPH`, and validate the variable, exact regular-file type, and non-symlink source before any copy. Do not echo it. Copy the external source to a non-repository `mktemp -d` directory, parse it, apply exactly:

1. delete top-level nodes `154` and `166`;
2. replace `/114/inputs/image` with `h3_edit_input.png`;
3. replace `/133/inputs/prompt` with `Apply the requested image edit.`;
4. replace `/165/inputs/filename_prefix` with `h3_edit`.

Canonicalize private source and sanitized copies after removing only those four allowed deltas and compare them byte-for-byte. Emit only PASS/FAIL. Validate the sanitized graph against the independent test constants before copying it to the destination. Re-read the external source and prove it was unchanged without printing or persisting a hash. Remove the validated temp directory. Never create a raw copy inside the worktree.

- [ ] **Step 6: Add package-data declarations and the minimal confined loader seam**

Add `"tldw_chatbook.Image_Generation" = ["workflows/*.json"]` to setuptools package data and the matching `recursive-include` to `MANIFEST.in`. Add only the minimal loader needed by the distribution test to the new adapter module: `importlib.resources.files(...)`, accept only the logical key `minimax_h3_image_edit`, map it internally to `minimax_h3_image_edit.json`, reject the suffixed filename/path separators/arbitrary keys, parse one known resource, validate it as a JSON object, and return a deep copy.

- [ ] **Step 7: Verify GREEN and mutate the inventory guards**

Run the same two files. Then, one at a time, add an unexpected node, change the node-165 link, add a second JSON resource, and remove the sdist include; each mutation must fail its named test. Restore exactly and rerun GREEN.

- [ ] **Step 8: Static checks and commit**

```bash
"$TASK3402_PYTHON" -B -m ruff check \
  Tests/Image_Generation/test_comfyui_workflow_assets.py \
  Tests/Image_Generation/test_comfyui_workflow_distribution.py \
  tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py
git diff --check
git add -- MANIFEST.in pyproject.toml \
  tldw_chatbook/Image_Generation/workflows/minimax_h3_image_edit.json \
  tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py \
  Tests/Image_Generation/test_comfyui_workflow_assets.py \
  Tests/Image_Generation/test_comfyui_workflow_distribution.py
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: package sanitized H3 image edit workflow"
```

Expected staged list: exactly the six named paths. No raw graph, private source identity, build directory, wheel, sdist, or egg-info remains in the worktree.

---

### Task 2: Extend the Image Generation Contracts with Strict Reference and Cancellation Semantics

**Files:**
- Modify: `tldw_chatbook/Image_Generation/adapters/base.py`
- Modify: `tldw_chatbook/Image_Generation/exceptions.py`
- Modify: `tldw_chatbook/Image_Generation/capabilities.py`
- Modify: `tldw_chatbook/Image_Generation/request_validation.py`
- Modify: `tldw_chatbook/Image_Generation/worker.py`
- Modify: `Tests/Image_Generation/test_contracts.py`
- Modify: `Tests/Image_Generation/test_capabilities.py`
- Modify: `Tests/Image_Generation/test_request_validation.py`
- Modify: `Tests/Image_Generation/test_worker.py`

**Interfaces:**
- Consumes: existing request/result/reference-image and validation choke point.
- Produces: backward-compatible fields used by the strict adapter and Console, without changing existing adapter constructors.

- [ ] **Step 1: Write contract RED tests**

Add tests proving:

- `ImageGenRequest.cancel_event` defaults to `None` and `build_request` preserves the exact supplied `threading.Event` object;
- `ImageGenResult.effective_params` defaults to `None` and accepts only a mapping at the type boundary;
- `ReferenceImageCapability.required` defaults to `False` for FAL/Gemini and resolves `True` only for ComfyUI;
- `ImageGenerationCancelled` is an `ImageGenerationError` subtype;
- a direct `run_generation()` call for ComfyUI with no reference is refused before adapter construction;
- required reference validation happens after canonical backend resolution, so aliases/disabled backends cannot bypass it;
- existing optional-reference backends and no-reference text-to-image requests retain behavior.

- [ ] **Step 2: Add non-vacuous image-byte validation RED tests**

Use small generated in-memory fixtures. Cover MIME/signature mismatch, truncated decode, decompression-bomb handling, unsupported mode, zero/oversized dimensions, pixel cap, empty content, `temp_path` source for ComfyUI, and a sentinel `file_path` that must never be opened. Assert failure precedes registry adapter construction and any network fake.

- [ ] **Step 3: Run the four focused files and capture RED**

```bash
"$TASK3402_PYTHON" -B -m pytest \
  Tests/Image_Generation/test_contracts.py \
  Tests/Image_Generation/test_capabilities.py \
  Tests/Image_Generation/test_request_validation.py \
  Tests/Image_Generation/test_worker.py -q
```

Expected RED: missing fields/types, ComfyUI capability absent, and required/decode boundary not enforced.

- [ ] **Step 4: Implement the minimal backward-compatible contract**

- Add `cancel_event: threading.Event | None = None` after existing request defaults.
- Add `effective_params: Mapping[str, JSONScalar] | None = None` after existing result defaults; define/export `JSONScalar = str | int | float | bool | None` without forcing existing adapters to pass it.
- Add `ImageGenerationCancelled` and a phase-coded `ComfyUIImageEditError` carrying only a closed phase token plus stable guidance; do not store raw response bodies or sensitive identifiers.
- Add `required: bool = False` to `ReferenceImageCapability`, add `comfyui` to the capability owner, and leave FAL/Gemini optional.
- Extend `build_request()` to preserve the exact event.
- In `run_generation()`, resolve backend, enforce the required-reference case, perform full shared validation, then load the adapter. Do not instantiate the adapter before validation.
- Decode in-memory reference bytes with Pillow under decompression-bomb protection, verify signature/MIME/mode/dimensions/pixels, and close all image objects. For ComfyUI require `content` and reject `temp_path`.

- [ ] **Step 5: Verify GREEN and mutations**

Run the same command. Mutation-check required-reference enforcement, exact event identity, MIME/signature agreement, pixel cap, and adapter-before-validation ordering. Restore and rerun.

- [ ] **Step 6: Static checks and commit**

```bash
"$TASK3402_PYTHON" -B -m ruff check \
  tldw_chatbook/Image_Generation/adapters/base.py \
  tldw_chatbook/Image_Generation/exceptions.py \
  tldw_chatbook/Image_Generation/capabilities.py \
  tldw_chatbook/Image_Generation/request_validation.py \
  tldw_chatbook/Image_Generation/worker.py \
  Tests/Image_Generation/test_contracts.py \
  Tests/Image_Generation/test_capabilities.py \
  Tests/Image_Generation/test_request_validation.py \
  Tests/Image_Generation/test_worker.py
git diff --check
git add -- tldw_chatbook/Image_Generation/adapters/base.py \
  tldw_chatbook/Image_Generation/exceptions.py \
  tldw_chatbook/Image_Generation/capabilities.py \
  tldw_chatbook/Image_Generation/request_validation.py \
  tldw_chatbook/Image_Generation/worker.py \
  Tests/Image_Generation/test_contracts.py \
  Tests/Image_Generation/test_capabilities.py \
  Tests/Image_Generation/test_request_validation.py \
  Tests/Image_Generation/test_worker.py
git diff --cached --check
git commit -m "feat: add required image edit request contracts"
```

---

### Task 3: Add Independent ComfyUI Image Settings and One Runtime Snapshot

**Files:**
- Modify: `tldw_chatbook/Image_Generation/config.py`
- Modify: `tldw_chatbook/Image_Generation/adapter_registry.py`
- Modify: `tldw_chatbook/Image_Generation/listing.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_image_gen_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/Image_Generation/test_config_loader.py`
- Modify: `Tests/Image_Generation/test_adapter_registry.py`
- Modify: `Tests/Image_Generation/test_listing.py`
- Modify: `Tests/UI/test_settings_image_gen_defaults.py`
- Modify: `Tests/UI/test_settings_image_gen_panel.py`

**Interfaces:**
- Consumes: ADR-052's independent image-provider configuration boundary.
- Produces: explicit-opt-in ComfyUI Image Generation configuration, catalog entry, canonical F9 editor, and post-save config+registry refresh.

- [ ] **Step 1: Write config/registry/listing RED tests**

Cover the exact `[image_generation.comfyui]` fields:

- `base_url` default `http://127.0.0.1:8188`;
- request/connect timeout, poll interval, total deadline;
- optional seed, steps, sampler, with blank/omitted values remaining `None`;
- normalized base URL rejects userinfo, query, fragment, non-HTTP(S), and malformed ports;
- ComfyUI is registered lazily but absent from default `enabled_backends` and never becomes the default backend automatically;
- local listing configurability uses enabled state + valid normalized URL + packaged-resource availability and makes no network call;
- Image config never reads `[video_generation.comfyui]` values;
- one `reset_image_generation_runtime()` clears config cache then registry;
- failed persistence performs neither reset; successful persistence performs exactly one reset.

- [ ] **Step 2: Write Settings RED tests**

Extend the pure schema and mounted panel tests for:

- canonical ComfyUI row and label;
- curated base URL, timeout, poll, deadline, optional seed/steps/sampler fields;
- cleared optional fields omitted from TOML and rendered as “Use packaged workflow”;
- no workflow path or model selector;
- explicit transmission/retention disclosure visible at 90×24 and neighbours inside screen bounds;
- saving a private-network URL is the one explicit trust action;
- only successful save resets config+registry;
- changing config after save causes the subsequent worker dispatch and capability listing to use the same new cached adapter snapshot.

- [ ] **Step 3: Run the five focused files and capture RED**

```bash
"$TASK3402_PYTHON" -B -m pytest \
  Tests/Image_Generation/test_config_loader.py \
  Tests/Image_Generation/test_adapter_registry.py \
  Tests/Image_Generation/test_listing.py \
  Tests/UI/test_settings_image_gen_defaults.py \
  Tests/UI/test_settings_image_gen_panel.py -q
```

- [ ] **Step 4: Implement config and runtime reset**

Add ComfyUI fields to `_NON_SECRET`, `_BACKEND_NAMES`, and `ImageGenerationConfig`; coerce positive timeout/interval values but reject invalid user-supplied optional generation controls rather than clamping them. Normalize the exact origin once. Implement `reset_image_generation_runtime()` with a lazy registry import to avoid a config↔registry import cycle. Add the lazy adapter spec to `DEFAULT_ADAPTERS`. Keep listing local-only.

Update the default config template with a commented `[image_generation.comfyui]` example and retention/transmission disclosure. Do not include the user's LAN origin.

- [ ] **Step 5: Implement F9 schema/render/save**

Add ComfyUI to `BACKEND_IDS`, labels, configured checks, field schema, validation/diffing, and the mounted Image Generation panel. The probe may perform a bounded explicit user-triggered `/object_info` check, but ordinary listing/rendering/saving does not probe. Saving a normalized base URL is consent; there is no second trust checkbox. Call `reset_image_generation_runtime()` only after all sections/deletions persist successfully.

- [ ] **Step 6: Verify GREEN and mutations**

Run the same command. Mutation-check default enablement, video-setting isolation, listing no-network, failed-save no-reset, successful-save registry reset, and disclosure geometry. Restore and rerun.

- [ ] **Step 7: Static checks and commit**

```bash
"$TASK3402_PYTHON" -B -m ruff check \
  tldw_chatbook/Image_Generation/config.py \
  tldw_chatbook/Image_Generation/adapter_registry.py \
  tldw_chatbook/Image_Generation/listing.py \
  tldw_chatbook/UI/Screens/settings_image_gen_defaults.py \
  Tests/Image_Generation/test_config_loader.py \
  Tests/Image_Generation/test_adapter_registry.py \
  Tests/Image_Generation/test_listing.py \
  Tests/UI/test_settings_image_gen_defaults.py \
  Tests/UI/test_settings_image_gen_panel.py
"$TASK3402_PYTHON" -B -m ruff check --select E9,F63,F7,F82 \
  tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py
git diff --check
git add -- tldw_chatbook/Image_Generation/config.py \
  tldw_chatbook/Image_Generation/adapter_registry.py \
  tldw_chatbook/Image_Generation/listing.py tldw_chatbook/config.py \
  tldw_chatbook/UI/Screens/settings_image_gen_defaults.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  Tests/Image_Generation/test_config_loader.py \
  Tests/Image_Generation/test_adapter_registry.py \
  Tests/Image_Generation/test_listing.py \
  Tests/UI/test_settings_image_gen_defaults.py \
  Tests/UI/test_settings_image_gen_panel.py
git diff --cached --check
git commit -m "feat: configure ComfyUI image generation independently"
```

---

### Task 4: Implement the Strict Packaged-Workflow ComfyUI Image Adapter

**Files:**
- Modify: `tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py`
- Modify: `Tests/Image_Generation/test_comfyui_image_adapter.py`
- Re-run: `Tests/Image_Generation/test_comfyui_workflow_assets.py`
- Re-run: `Tests/Image_Generation/test_comfyui_workflow_distribution.py`

**Interfaces:**
- Consumes: validated `ImageGenRequest`, one confined graph, independent config, and existing egress policy.
- Produces: one validated `ImageGenResult` PNG with exact effective metadata, or a sanitized typed phase failure/cancellation.

- [ ] **Step 1: Write local preparation RED tests**

Test the real packaged graph and targeted malformed copies:

- exact node/class/direct-link inventory and one output node;
- reject missing/duplicate/wrong-class expected nodes, unexpected output, direct-link drift, and title decoys;
- deep-copy immutability across success and failure;
- inject only `114.image`, `125.sampler_name`, `126.steps`, `131.noise_seed`, and `133.prompt`;
- keep node 165 prefix and every graph-owned literal unchanged;
- programmatic seed/steps/sampler override settings defaults; unset values retain validated graph literals;
- `seed=-1` resolves exactly once into the accepted non-negative ComfyUI range;
- reject negative prompt, CFG, model, non-PNG format, width, height, and nonempty unsupported extras before `/object_info` or upload;
- effective metadata exactly matches the same prepared graph later queued.

- [ ] **Step 2: Write remote preflight RED tests**

Use a scripted fake transport that captures only methods/paths and bounded synthetic JSON. Require `/object_info` before upload and validate every required class/input, loader/model choice, sampler, and SaveImage PNG contract. Missing class/choice/schema must fail with phase `remote_schema_preflight` and zero upload/prompt calls. Oversized declared and streamed JSON must fail before `json.loads`.

- [ ] **Step 3: Write exchange/output RED tests**

Cover:

- opaque random upload name with extension from validated MIME and no original filename;
- exact-origin endpoint construction and redirects disabled for every call;
- request base URL trusted hostname is the only private-origin exception;
- safe upload response injection into node 114;
- prompt submission and opaque prompt-ID validation without logging it;
- monotonic bounded polling, preview/unrelated outputs nonterminal, terminal errors sanitized;
- node 165 only, exactly one image descriptor, `type=output`, safe PNG filename and relative subfolder;
- descriptor values used only as `/view` query data;
- declared and streamed PNG cap no greater than `inline_max_bytes`;
- normalized `image/png`, signature, decode, mode, and preserved dimensions;
- result fields: bytes/type/length, resolved seed, no guessed model, exact allowlisted effective params;
- all sentinel instruction/path/name/descriptor/body values absent from logs and exception/user guidance.

- [ ] **Step 4: Write cancellation/timeout RED tests**

Prove:

- event checked before each phase and poll waiting uses `cancel_event.wait(interval)`;
- cancellation before prompt ID performs no queue delete;
- cancellation or deadline after prompt ID attempts exactly one `POST /queue` with `{"delete": [prompt_id]}`;
- no global interrupt endpoint;
- delete refusal/failure cannot mask cancellation/timeout;
- final event check after validated PNG linearizes cancel-vs-success;
- cancellation raises `ImageGenerationCancelled`; timeout raises sanitized failure;
- deadline arithmetic uses monotonic remaining time and no network timeout overshoots the total deadline.

- [ ] **Step 5: Run adapter + asset/distribution files and capture RED**

```bash
"$TASK3402_PYTHON" -B -m pytest \
  Tests/Image_Generation/test_comfyui_image_adapter.py \
  Tests/Image_Generation/test_comfyui_workflow_assets.py \
  Tests/Image_Generation/test_comfyui_workflow_distribution.py -q
```

- [ ] **Step 6: Implement preparation and bounded transport**

Use small private helpers inside the adapter module, not a new transport package:

- `_load_packaged_workflow()` and `_prepare_workflow(request, upload_name=None)`;
- `_read_bounded_json(response, allow_empty=False)` with 32 MiB declared/actual cap before `json.loads`;
- `_request_json(...)` and `_stream_png(...)` that use existing egress checks, exact normalized origin, explicit trusted hostname, and `follow_redirects=False`;
- `_validate_object_info(prepared, schema)` before upload;
- `_upload_reference`, `_submit_prompt`, `_poll_history`, `_delete_pending_prompt_once`, `_select_output_descriptor`, `_download_output`;
- one phase-normalization boundary that never formats raw exception strings into user copy/logs.

Keep graph/output objects local to the synchronous call. Do not cache request graphs, prompt IDs, descriptors, or bytes on the adapter.

- [ ] **Step 7: Verify GREEN and required mutations**

Run the same command. One at a time mutate: pre-upload preflight, exact output node, same-origin enforcement, redirect disabling, JSON declared cap, JSON actual cap, PNG actual cap, preview nonterminal handling, one-time queue deletion, final cancellation check, and graph deep copy. Each named test must fail. Restore and rerun GREEN.

- [ ] **Step 8: Static checks and commit**

```bash
"$TASK3402_PYTHON" -B -m ruff check \
  tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py \
  Tests/Image_Generation/test_comfyui_image_adapter.py \
  Tests/Image_Generation/test_comfyui_workflow_assets.py \
  Tests/Image_Generation/test_comfyui_workflow_distribution.py
git diff --check
git add -- tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py \
  Tests/Image_Generation/test_comfyui_image_adapter.py
git diff --cached --check
git commit -m "feat: execute strict ComfyUI H3 image edits"
```

---

### Task 5: Preserve Exact Attachment Ownership and Image Metadata

**Files:**
- Modify: `tldw_chatbook/Chat/attachment_core.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_generate_image.py`
- Modify: `Tests/Chat/test_attachment_core.py`
- Modify: `Tests/Chat/test_console_generation_store.py`
- Modify: `Tests/Chat/test_console_generate_image.py`

**Interfaces:**
- Consumes: validated adapter result and existing durable generation-message append.
- Produces: exact source identity, cancellation propagation, and allowlisted `GenerationVariantMeta.params` without changing current adapters.

- [ ] **Step 1: Write attachment/store RED tests**

Prove every independently constructed `PendingAttachment` gets a unique opaque UUID string, copies/stashes preserve it, and no persistence/serialization row receives it. Add `consume_pending_attachment(session_id, attachment_id)` tests for exact removal, identity mismatch no-op, ordering preservation, additional attachments retained, missing session behavior, and concurrent replacement characterization.

Add an idempotent persisted-generation-message merge test using the real persistence read contract: exact message ID, PNG attachment, aligned metadata, no duplicate when already present, no unrelated message mutation.

- [ ] **Step 2: Write batch/result RED tests**

Prove:

- `backend="comfyui"` accepts only `count == 1` before request construction;
- the same cancellation event reaches `build_request` and adapter generation;
- `ImageGenerationCancelled` is re-raised before the generic per-variant catch;
- only scalar values for `operation`, `workflow_key`, `width`, `height`, `steps`, `sampler`, `format` enter `GenerationVariantMeta.params`;
- unknown or nonscalar effective parameters cause a typed refusal rather than persistence;
- validated `result.content_type`, not metadata, supplies the attachment MIME;
- existing adapters with `effective_params=None` still emit `params={}`.

- [ ] **Step 3: Run the three focused files and capture RED**

```bash
"$TASK3402_PYTHON" -B -m pytest \
  Tests/Chat/test_attachment_core.py \
  Tests/Chat/test_console_generation_store.py \
  Tests/Chat/test_console_generate_image.py -q
```

- [ ] **Step 4: Implement minimal identity and mapping seams**

Add `attachment_id: str = field(default_factory=lambda: str(uuid4()))` as nonpersisted in-memory state. Implement exact list search/removal in the store without clearing or replacing the list. Add a narrow store hydration/merge method only if the existing restore API cannot idempotently insert one exact persisted message; reuse persistence reads and message-ID identity.

Extend `run_generation_batch()` with `reference_image`, `cancel_event`, and strict ComfyUI count behavior. Catch `ImageGenerationCancelled` first and re-raise. Validate the allowlisted effective mapping with exact scalar types (`bool` remains allowed as JSON scalar, but never silently coerce containers/objects).

- [ ] **Step 5: Verify GREEN and mutations**

Mutation-check UUID default, exact-remove vs clear-all, cancellation re-raise, count-one guard before build, unknown metadata rejection, and validated MIME authority. Restore and rerun.

- [ ] **Step 6: Static checks and commit**

```bash
"$TASK3402_PYTHON" -B -m ruff check \
  tldw_chatbook/Chat/attachment_core.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_generate_image.py \
  Tests/Chat/test_attachment_core.py \
  Tests/Chat/test_console_generation_store.py \
  Tests/Chat/test_console_generate_image.py
git diff --check
git add -- tldw_chatbook/Chat/attachment_core.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/console_generate_image.py \
  Tests/Chat/test_attachment_core.py \
  Tests/Chat/test_console_generation_store.py \
  Tests/Chat/test_console_generate_image.py
git diff --cached --check
git commit -m "feat: preserve H3 image edit attachment ownership"
```

---

### Task 6: Implement App-Owned H3 Console Lifecycle and Remount Reconciliation

**Files:**
- Create: `tldw_chatbook/Chat/console_image_edit_operations.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `Tests/Chat/test_console_h3_image_edit.py`
- Modify: `Tests/Chat/test_console_generation_actions.py`
- Modify: `Tests/UI/test_console_pending_attachment_stash.py`

**Interfaces:**
- Consumes: exact attachment identity, strict batch helper, durable generation append, app lifetime, and fresh-screen stash/restore lifecycle.
- Produces: one H3 edit per session across remounts, typed Stop/unmount cancellation, exact-once durable success, no stale UI, and identity-gated cleanup.

- [ ] **Step 1: Write pure operation-registry RED tests**

Define expected immutable records before production:

- active operation: session ID key, opaque generation token, attachment ID, captured draft, exact `threading.Event`, owned async task;
- completion cleanup: session ID, generation token, persisted message ID, attachment ID, captured draft only.

Test duplicate begin refusal, generation-checked removal, Stop event set, unmount terminal marking without blocking, session deletion cleanup, byte/path rejection in completion records, and a later operation surviving an old task's `finally`.

- [ ] **Step 2: Write real command-preparation RED tests**

Drive `_console_command_generate_image` with actual grammar and store:

- backend resolves before generic `prepare_generation_request`;
- one image + nonempty raw instruction accepted;
- zero/multiple/non-image/missing-memory-data refused before build/network;
- style token rejected and no style/template/context/LLM helper invoked;
- global batch default ignored and exact count one dispatched;
- raw instruction remains byte-for-byte the prompt;
- `ResolvedReferenceImage` uses attachment ID, `filename=None`, in-memory bytes, validated MIME/dimensions, and never opens `file_path`;
- unsupported programmatic values fail before upload;
- command draft remains during work; failure/cancellation preserves it.

- [ ] **Step 3: Write cancellation/commit linearization RED tests**

Use deterministic barriers around adapter final event check and durable append:

- Stop before final check: typed cancellation, no card, no consume, no error log;
- success wins final check: shielded durable append completes exactly once even if the app-owned operation task is cancelled immediately afterward;
- app-owned operation cancellation sets the same event, awaits the real runner child, then re-raises;
- application shutdown cancels and drains every registered operation before screen/database teardown;
- persistence failure retains attachment/draft and produces sanitized phase copy;
- consume mismatch/exception after commit does not roll back or misreport success;
- UI sync is outside shield, terminal/generation checked, and cancellable;
- unmount sets event and returns without waiting for blocked HTTP;
- a fresh screen sees active/stopping state and cannot start a duplicate.

- [ ] **Step 4: Write remount/stash reconciliation RED tests**

Use two real fresh `ChatScreen`/`ConsoleChatStore` instances and an app-owned registry:

- success before old-screen stash, after stash but before new adoption, and after new adoption;
- completion filters exact attachment from app stash immediately;
- new store rehydrates exact persisted message/PNG/metadata if absent, no-op if present, consumes only initiating attachment, preserves others;
- captured draft clears only when unchanged; replacement draft survives;
- record ack occurs only after message presence and cleanup;
- success arriving after new mount schedules reconciliation on that live screen;
- session deletion drops record;
- no source resurrection, missing durable result, duplicate message, or stale-screen UI update.

- [ ] **Step 5: Write Regenerate and privacy RED tests**

`operation="edit"` must refuse before variant capacity/in-flight/network checks with fixed restage guidance. Other generated images retain normal Regenerate behavior. Capture every log/user message with sentinel instruction/path/filename/descriptor/body values and assert none appear; assert component/phase/error_type are present. Patch Video Generation imports/store calls to raise if invoked.

- [ ] **Step 6: Run the three focused files and capture RED**

```bash
"$TASK3402_PYTHON" -B -m pytest \
  Tests/Chat/test_console_h3_image_edit.py \
  Tests/Chat/test_console_generation_actions.py \
  Tests/UI/test_console_pending_attachment_stash.py -q
```

- [ ] **Step 7: Implement the app-owned registry and screen orchestration**

Keep `console_image_edit_operations.py` UI-agnostic. The registry owns active tasks and completion records; it does not own image bytes. Construct it once in `TldwCli.__init__`. In `ChatScreen`:

1. parse and resolve backend/config;
2. branch on `comfyui` before generic preparation;
3. validate exact attachment/instruction/style/count and build the in-memory reference;
4. register the app-owned operation or refuse a duplicate, refresh Stop, and return from the Textual handler;
5. have the registry-owned task create and shield the real runner child containing blocking batch work and the durable success append;
6. on success, append durable PNG+metadata, create/filter completion cleanup, exact-consume source, and clear unchanged draft;
7. on cancellation of the app-owned operation, set the exact event, drain the shielded runner to settlement, then re-raise;
8. cancel and drain all registered operations during application shutdown before Textual teardown;
9. perform current-live-screen/session/generation-checked UI settlement after generation-matched active removal;
10. reconcile byte-free completion records during restore/adoption and on late completion.

The existing generic image path remains unchanged except for shared helper signatures introduced in Task 5.

- [ ] **Step 8: Verify GREEN and required mutations**

Run the same command. Mutation-check backend-before-generic preparation, app-owned duplicate gate, exact event identity, registry-owned shielding and runner drain, application-shutdown drain, generation-matched registry removal, exact stash filtering, idempotent hydration, attachment identity, unchanged-draft condition, outer terminal UI gate, and Regenerate early refusal. Restore and rerun.

- [ ] **Step 9: Static checks and commit**

```bash
"$TASK3402_PYTHON" -B -m ruff check \
  tldw_chatbook/Chat/console_image_edit_operations.py \
  Tests/Chat/test_console_h3_image_edit.py \
  Tests/Chat/test_console_generation_actions.py \
  Tests/UI/test_console_pending_attachment_stash.py
"$TASK3402_PYTHON" -B -m ruff check --select E9,F63,F7,F82 \
  tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py
git diff --check
git add -- tldw_chatbook/Chat/console_image_edit_operations.py \
  tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_h3_image_edit.py \
  Tests/Chat/test_console_generation_actions.py \
  Tests/UI/test_console_pending_attachment_stash.py
git diff --cached --check
git commit -m "feat: run H3 image edits across Console remounts"
```

---

### Task 7: Run the Authorized Final Gates and Live UAT

**Files:**
- Create: `Docs/superpowers/qa/2026-08-10-comfyui-h3-image-edit-uat.md`
- Modify: `backlog/tasks/task-3402 - H3-static-image-edit-through-Image_Generation.md`
- Create ignored: `.superpowers/sdd/2026-08-10-comfyui-h3-image-edit/final-report.md`

**Interfaces:**
- Consumes: all committed Tasks 1–6 and the user-configured trusted server.
- Produces: focused automated/static evidence, sanitized live structural evidence, completed task notes, and no local test/UAT residue.

- [ ] **Step 1: Re-read governing documents and audit scope**

```bash
backlog task 3402 --plain
sed -n '1,240p' backlog/decisions/052-comfyui-h3-image-edit-provider-boundary.md
task3402_base=$(sed -n 's/^implementation_base=//p' \
  .superpowers/sdd/2026-08-10-comfyui-h3-image-edit/task-1-report.md)
test -n "$task3402_base"
git cat-file -e "${task3402_base}^{commit}"
git merge-base --is-ancestor "$task3402_base" HEAD
git diff --name-only "$task3402_base"..HEAD
git status --short
```

Expected: only paths authorized by this plan; workflow source diff contains exactly the sanitized destination, never a raw export.

- [ ] **Step 2: Run the final focused automated gate only**

```bash
"$TASK3402_PYTHON" -B -m pytest \
  Tests/Image_Generation/test_comfyui_workflow_assets.py \
  Tests/Image_Generation/test_comfyui_workflow_distribution.py \
  Tests/Image_Generation/test_contracts.py \
  Tests/Image_Generation/test_capabilities.py \
  Tests/Image_Generation/test_request_validation.py \
  Tests/Image_Generation/test_worker.py \
  Tests/Image_Generation/test_config_loader.py \
  Tests/Image_Generation/test_adapter_registry.py \
  Tests/Image_Generation/test_listing.py \
  Tests/Image_Generation/test_comfyui_image_adapter.py \
  Tests/UI/test_settings_image_gen_defaults.py \
  Tests/UI/test_settings_image_gen_panel.py \
  Tests/Chat/test_attachment_core.py \
  Tests/Chat/test_console_generation_store.py \
  Tests/Chat/test_console_generate_image.py \
  Tests/Chat/test_console_h3_image_edit.py \
  Tests/Chat/test_console_generation_actions.py \
  Tests/UI/test_console_pending_attachment_stash.py -q
```

Do not broaden this command. Record exact counts, skips, and warnings. Any failure blocks UAT/Done.

- [ ] **Step 3: Run exact static analysis and compilation**

Run full Ruff only on changed/new small modules and tests. For baseline-large `app.py`, `chat_screen.py`, `settings_screen.py`, and `config.py`, run the repository-authorized fatal subset `E9,F63,F7,F82` and document that exact scope. Compile every changed Python file to a `TemporaryDirectory` destination so no `__pycache__` is left in the worktree. Run `git diff --check`.

- [ ] **Step 4: Build/install packaging proof outside the repository**

Repeat the distribution test in a fresh pytest temp root. Inspect wheel and sdist members: exactly one Image Generation workflow JSON; no claim about unrelated Video Generation resources. Install the wheel into a fresh temp target and load the graph only through the installed adapter's confined loader. Confirm no `dist/`, `build/`, wheel, sdist, or egg-info residue in the worktree.

- [ ] **Step 5: Run privacy/provenance scans**

Without naming the external source, scan tracked paths, staged diff, all branch history, plan/spec/task/UAT, fixtures, and ignored report for:

- raw/original/backup workflow artifacts;
- absolute external workflow paths;
- source identity/hashes;
- media, build, prompt-ID, descriptor, upload-name, host, or credential residue;
- forbidden `Video_Generation` imports in the new Image Generation and Console operation modules;
- the approved neutral filler appearing exactly where intended.

Never print a match containing private source material; fail closed and inspect privately if a broad scan signals one.

- [ ] **Step 6: Live preflight and edit through the real adapter**

Use a validated temporary root and synthetic PNG. Read the origin from isolated Image Generation config; never hardcode or print it. Verify `/object_info` through the adapter's bounded transport and required class/schema/loader/sampler/SaveImage contract. Run one neutral unrecorded instruction through `worker.run_generation()`/`ComfyUIImageAdapter`, not a hand-built HTTP script.

Require:

- exactly one validated `image/png` result;
- source dimensions preserved;
- node 165 selected;
- resolved seed/steps/sampler/workflow key/operation/format metadata present;
- no Video Generation import/store call;
- no raw prompt, host, prompt ID, descriptor, model filename, or response body emitted.

- [ ] **Step 7: Persist through the normal image boundary**

Using isolated temporary persistence, call the real `ConsoleChatStore.append_generation_message(..., persist=True)` path with the adapter result and mapped metadata. Rehydrate it, assert exact message/attachment/metadata identity and one PNG variant, then remove isolated local persistence and download buffers. Do not persist the synthetic source. State honestly that server-side input/output retention is operator-managed.

- [ ] **Step 8: Write sanitized UAT evidence**

Create the tracked UAT document with only:

- date and configured-trusted-origin class (no host);
- required-class/schema preflight pass;
- one node-165 PNG pass;
- dimensions preserved (numeric dimensions are allowed only if they describe the synthetic fixture, not private media);
- effective metadata key presence, not opaque values;
- normal Image Generation persistence/rehydration pass;
- explicit no-video-path pass;
- local cleanup pass;
- operator-managed server retention statement.

The ignored report may contain exact test counts and commit hashes, but not source identity, server identity, prompt text, descriptors, or media.

- [ ] **Step 9: Final whole-task review and fixes**

Request one implementation/spec reviewer to inspect the complete implementation-base..HEAD diff against TASK-3402, ADR-052, design, this plan, and UAT. Review Critical/Important/Minor findings. Fix validated findings with fresh focused RED→GREEN cycles and separate commits. Repeat review up to three rounds; do not wave through an unresolved Important finding.

- [ ] **Step 10: Complete Backlog hygiene and commit closeout**

Only after every AC and DoD item passes:

1. prepare concise final notes with approach, trade-offs, exact modified files/groups, ADR-052, exact focused tests/static scopes, live-UAT result, commit hashes, and the explicit no-full-suite deviation authorized by AC #6;
2. use one Backlog CLI call to install those notes, check AC 1–6, and set TASK-3402 Done: `backlog task edit 3402 --notes "$task3402_notes" --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 -s Done`;
3. state whether a real generalizable lesson surfaced; add an incident-based lesson only if one did;
4. re-read `backlog task 3402 --plain` and confirm all six ACs checked;
5. stage only the task and tracked UAT document;
6. commit `docs: complete ComfyUI H3 image edit task`;
7. verify clean normal status (ignored report omitted), exact commit scope, and no prohibited artifacts.

---

## Final Definition-of-Done Gate

- [ ] TASK-3402 is Done via CLI and all six AC checkboxes are checked.
- [ ] ADR-052 is linked from task plan and notes; no duplicate ADR exists.
- [ ] One sanitized workflow resource ships in source, wheel, and sdist; the external source never entered Git.
- [ ] Every supported control is applied or graph-validated; every unsupported control fails before upload.
- [ ] Bounded exact-origin JSON/upload/prompt/history/queue/view exchange and node-165 PNG validation pass.
- [ ] The exact source attachment and unchanged draft are consumed only after exact-once durable success.
- [ ] Typed cancellation, app-owned drain, remount reconciliation, and Regenerate refusal pass non-vacuous mutation checks.
- [ ] Canonical F9 settings are independent, explicit opt-in, disclose transmission/retention, and reset one Image Generation runtime snapshot only after successful save.
- [ ] Only the plan's focused test files and exact static targets were run and reported.
- [ ] Live UAT passes through the real adapter and normal Image Generation persistence boundary.
- [ ] Tracked/ignored evidence is source-private, prompt-free, host-free, descriptor-free, credential-free, and media-free.
- [ ] No build, temp, download, media, `__pycache__`, or UAT residue remains.
- [ ] Whole-task review has no unresolved Critical, Important, or Minor finding.
