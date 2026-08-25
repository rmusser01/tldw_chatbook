# Research Workspace Extended Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`; apply `superpowers:test-driven-development`
> before production changes and `superpowers:verification-before-completion`
> before commits. Do not delegate unless the user explicitly requests
> subagents. Apply `impeccable` immediately before UI implementation tasks.

**Goal:** Complete the audited Research Workspace namespace with honest More
outputs, work products, workspace/help controls, search, diagnostics, and
owner links while keeping unavailable and Planned controls non-deceptive.

**Architecture:** Extend the existing `ResearchCapability` projection with a
single action registry for all ten outputs, five work products, menus, and
owner links. Extended output adapters activate only when both a real generator
and canonical owner pass readiness: server workspace artifacts/Data Tables/
Slides/audio owners or Local TTS/Chatbook owners. The UI renders one command
per action, progressively discloses relevant options, and navigates management
actions to their real screens.

**Tech Stack:** Python 3.11+, Textual 8.x, existing Research Studio service,
TLDW Data Tables/Slides/Audio/workspace APIs, Local TTS and Chatbooks,
workspace registry/overlay, pending handoffs, pytest.

**Spec:**
`Docs/superpowers/specs/2026-08-23-research-workspace-design.md`

**Backlog:** `TASK-21512` (depends on `TASK-21507` through `TASK-21511`)

## Global constraints

- `More outputs…` has Learn, Analyze, and Present groups. Primary outputs may
  appear for orientation but resolve to the same action ID/control.
- The ten output labels are Summary, Flashcards, Quiz, Report, Compare Sources,
  Mind Map, Timeline, Data Table, Slides, and Audio Summary.
- Work products are Executive Brief, Literature Matrix, Corpus Gap Finder,
  Evidence-Bound Hypotheses, and Research Proposal Pack.
- Research Dossier, Competitive Market Memo, and Technical Project Spec are
  text-labeled `Planned`; they are not buttons, focus targets, commands, or
  dispatchable action IDs.
- Coarse server categories never activate a specific output by themselves.
  Generator readiness and canonical-owner readiness must both be true.
- Unsupported controls preserve selected authority and show owner, reason,
  impact, and recovery. They never execute a substitute.
- Research does not recreate Settings, Library, Artifacts, Study, Console,
  MCP, ACP, Lab, or Logs management UI.
- Disabled text must meet the project's measured 3:1 contrast rule and carry a
  readable reason; color is supplemental.
- F1 and footer hints come from the real active binding set. Do not bind
  reserved/terminal-convention keys.
- Unknown or stale capability fails closed and increments context revision.
- No full-suite run/claim without explicit user approval.

## ADR check

ADR required: no new ADR

ADR path:
`backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 and the approved design already classify these controls and
their owner boundaries. If implementation would require a new generator,
canonical owner, service contract, or persistence boundary not listed below,
stop and create a new ADR before proceeding.

## Extended owner truth table

| Action | Local | Server |
| --- | --- | --- |
| Mind Map | unavailable: DB tables exist but no working owner/editor | server grounded generation -> workspace artifact `artifact_type="mindmap"` |
| Timeline | unavailable: no Local canonical owner | server grounded generation -> workspace artifact `artifact_type="timeline"` |
| Data Table | unavailable: no Local Data Table owner | Data Tables API `uuid/version`; `artifact_type="data_table"` reference, not duplicated rows |
| Slides | unavailable: no Local Slides owner | Slides API `presentation_id/version`; `artifact_type="slides"` reference |
| Audio Summary | Research summary generation -> Local TTS history `history_id` + output membership | server summary generation -> audio speech job/history artifact + `artifact_type="audio_overview"` reference |
| Executive Brief | Research generation -> Local Chatbook + output membership | server grounded generation -> `artifact_type="report"` workspace artifact |
| Literature Matrix | unavailable: requires a structured table owner | server Data Tables API + `artifact_type="data_table"` reference |
| Corpus Gap Finder | unavailable: requires a structured table owner | server Data Tables API + `artifact_type="data_table"` reference |
| Evidence-Bound Hypotheses | Research generation -> Local Chatbook + output membership | server grounded generation -> `artifact_type="report"` workspace artifact |
| Research Proposal Pack | Research generation -> Local Chatbook + output membership | server grounded generation -> `artifact_type="report"` workspace artifact |

Server reference artifacts contain owner kind/ID/version, title, provenance,
and status only; canonical slide/table/audio payloads remain in native owners.
If server Audio does not return a stable inspectable owner ID, Audio Summary
remains unavailable in Server mode rather than treating downloaded bytes as
persistence.

## Task 1: Complete the capability/action registry

**Files:**

- Extend: `tldw_chatbook/Research_Workspace/studio_models.py`
- Extend: `tldw_chatbook/Research_Workspace/studio_mapping.py`
- Create: `tldw_chatbook/Research_Workspace/action_registry.py`
- Extend: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Test: `Tests/Research_Workspace/test_extended_capabilities.py`
- Test: `Tests/Research_Workspace/test_action_registry.py`

1. Add RED matrix tests for ten outputs, five work products, three inert
   Planned labels, two authorities, zero duplicate action IDs, unknown/stale
   capability, coarse-category false positives, and every owner/recovery field.
2. Extend `ResearchOutputKind` with `MIND_MAP`, `TIMELINE`, `DATA_TABLE`,
   `SLIDES`, and `AUDIO_SUMMARY`; define `ResearchWorkProductKind` with the five
   actionable IDs. Planned labels are presentation records without action IDs.
3. Add `ResearchActionDefinition(action_id, label, group, owner_by_authority,
   required_capabilities, option_schema_id, reopen_target)` and one immutable
   registry. Primary and More-output surfaces resolve the same definition.
4. Reuse the strict source-status and workspace-capability client delivered by
   TASK-21508. Do not add another probe path or infer support from broad server
   health.
5. Project specific readiness from exact probes:

   - Data Table: API capability plus supported selected server source kinds;
   - Slides: Slides health/generation route plus selected server source kinds;
   - Audio: TTS health/provider/voice plus stable history/job artifact owner;
   - Mind Map/Timeline/text work product: grounded server completion plus
     workspace artifact create/update capability;
   - Local Audio: Local TTS health/history plus Local membership;
   - Local textual work product: processing route plus Local Chatbook owner.

6. Capability cache keys include qualified workspace, capability revision,
   provider/model identity, and source readiness revision. A changed probe
   invalidates visible action projection and increments controller context.
7. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_extended_capabilities.py Tests/Research_Workspace/test_action_registry.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: project extended Research capabilities"
   ```

## Task 2: Add native Server structured/present/audio owner adapters

**Files:**

- Extend: `tldw_chatbook/Research_Workspace/server_studio.py`
- Reuse: `tldw_chatbook/Audio_Services_Interop/audio_services_scope_service.py`
- Modify: `Tests/Research_Workspace/test_server_studio.py`
- Test: `Tests/Audio_Services_Interop/test_audio_services_scope_service.py`

1. Add RED tests for all five extended outputs, selected source qualification,
   exact owner IDs/versions, owner reference artifact, job polling/cancel,
   source mismatch, unavailable native owner, client download not persistence,
   stale completion, and no Local calls.
2. Mind Map and Timeline reuse TASK-21510's immutable generation context and
   server grounded route, validate bounded Mermaid/text result, and persist the
   full content as a server workspace artifact of the exact type.
3. Data Table calls `TLDWAPIClient.generate_data_table` with qualified source
   inputs, stable workspace tag, columns/max rows/model options, and a durable
   job. Poll/cancel through existing job APIs; after completion create a
   workspace artifact reference containing table UUID/version only.
4. Slides calls `generate_presentation_from_rag` for the captured workspace
   query/source scope and selected visual style. Treat returned presentation
   ID/version as canonical; create a workspace artifact reference and expose
   existing export/render methods.
5. Audio Summary first generates a bounded cited summary script through the
   server route, then calls `AudioServicesScopeService(mode="server")` with
   provider/model/voice/speed/format. Activate only if the response yields a
   stable inspectable history/job artifact ID; associate via a workspace
   artifact reference.
6. Replace/new-version/delete/list always call the native owner first and
   update/remove its reference artifact second. A failed reference update
   preserves the canonical native item and returns retry-link recovery.
7. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_server_studio.py Tests/Audio_Services_Interop/test_audio_services_scope_service.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: add server extended Research outputs"
   ```

## Task 3: Add Local Audio Summary and actionable work-product adapters

**Files:**

- Create: `tldw_chatbook/Research_Workspace/work_product_templates.py`
- Extend: `tldw_chatbook/Research_Workspace/studio_generation.py`
- Extend: `tldw_chatbook/Research_Workspace/local_studio.py`
- Extend: `tldw_chatbook/Research_Workspace/server_studio.py`
- Test: `Tests/Research_Workspace/test_work_product_templates.py`
- Modify: `Tests/Research_Workspace/test_studio_generation.py`
- Modify: `Tests/Research_Workspace/test_local_studio.py`

1. Add RED tests for template labels/sections/minimum sources/citation policy,
   Local/Server owner mapping, structured table requirements, citation audit,
   schema-invalid results, audio owner persistence, TTS failure, overlay-free
   provenance, and three Planned labels absent from dispatch registry.
2. Define immutable templates matching the audited product semantics:

   - Executive Brief: Situation, Key Findings, Evidence, Risks, Recommended
     Actions; minimum one source; citations required.
   - Literature Matrix: source/method/sample/finding/limitations/
     contradictions table; minimum two sources.
   - Corpus Gap Finder: gap/type/evidence/sources/importance/follow-up table;
     minimum two sources.
   - Evidence-Bound Hypotheses: hypothesis/support/prediction/method/validity/
     confidence; minimum two sources.
   - Research Proposal Pack: question/literature/gaps/hypothesis/method/risks/
     source audit; minimum two sources.

3. Extend TASK-21510's immutable generator with the work-product templates and
   bounded structured-result validators; do not introduce a second generation
   coordinator. Local Executive Brief, Hypotheses, and Proposal persist as
   Chatbooks. Server equivalents persist as workspace artifacts. Literature
   Matrix and Corpus Gap Finder are Server-only Data Tables until a Local
   structured-table owner exists.
4. Extend `LocalStudioAdapter` for Audio Summary: generate the summary script,
   then call
   `AudioServicesScopeService.create_audio_speech(mode="local", request_data=...)`.
   Link returned `history_id` as `item_type="tts_audio"`, `role="output"` with
   bounded provenance; history remains the list/get/delete owner.
5. Planned templates are constant `PlannedOutputLabel` records with no action,
   command, key binding, click handler, or focus flag.
6. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_work_product_templates.py Tests/Research_Workspace/test_studio_generation.py Tests/Research_Workspace/test_local_studio.py
   ```

7. Commit:

   ```bash
   git commit -m "feat: add Research work products and local audio"
   ```

## Task 4: Build More outputs with one-command semantics and progressive options

**Files:**

- Create: `tldw_chatbook/UI/Research_Workspace_Modules/more_outputs_dialog.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/output_options.py`
- Extend: `tldw_chatbook/UI/Research_Workspace_Modules/studio_region.py`
- Extend: `tldw_chatbook/UI/Research_Workspace_Modules/studio_history.py`
- Modify: `tldw_chatbook/css/features/_research_workspace.tcss`
- Test: `Tests/UI/test_research_more_outputs.py`
- Test: `Tests/UI/test_research_output_options.py`
- Test: `Tests/UI/test_research_extended_history.py`

1. Add mounted RED tests for Learn/Analyze/Present grouping, one action per
   primary item, five extended outputs, five work products, reason/recovery,
   three non-focusable Planned labels, option visibility by type, generated
   receipts, reopen/export, and capability revision changes.
2. Render orientation rows for Summary/Flashcards/Quiz/Report/Compare by
   resolving the existing primary action ID; do not create secondary buttons,
   duplicate command-palette entries, or separate handlers.
3. Default rows show label, availability, owner/destination, and concise
   reason. Selecting an available row reveals only relevant options:

   - shared provider/model/generation/retrieval controls;
   - Data Table column hints/max rows/export CSV/JSON/XLSX;
   - Slides visual style/theme/source limit/export format;
   - Audio provider/model/voice preview/speed/format;
   - Flashcard deck target remains on the existing primary action.

4. Unavailable rows remain readable and focusable only when activation opens
   meaningful reason/recovery detail. Planned labels use `Static`, not disabled
   `Button`, so they are not in focus order.
5. Extend canonical history resolution for native owner/reference pairs. A
   broken reference names the surviving native owner and offers retry link;
   membership/reference metadata never reconstructs missing payload.
6. Apply the established terminal design system: flat workbench hierarchy,
   semantic state tokens, full recovery borders instead of decorative stripes,
   dimensionally stable focus, and measured readable disabled text.
7. Rebuild CSS; verify the complete enabled set is reachable at 160x40,
   120x30, 100x30, 84x24, 80x24, and 60x20.
8. Run:

   ```bash
   .venv/bin/python tldw_chatbook/css/build_css.py
   .venv/bin/python -m pytest -q Tests/UI/test_research_more_outputs.py Tests/UI/test_research_output_options.py Tests/UI/test_research_extended_history.py Tests/UI/test_research_workspace_geometry.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: add Research More outputs UI"
   ```

## Task 5: Complete workspace search, lifecycle menu, templates, and exports

**Files:**

- Create: `tldw_chatbook/Research_Workspace/workspace_templates.py`
- Create: `tldw_chatbook/Research_Workspace/bibtex_export.py`
- Extend: `tldw_chatbook/Research_Workspace/local_adapter.py`
- Extend: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/workspace_search.py`
- Extend: `tldw_chatbook/UI/Research_Workspace_Modules/workspace_menu.py`
- Extend: `tldw_chatbook/Research_Workspace/overlay_store.py`
- Test: `Tests/Research_Workspace/test_workspace_templates.py`
- Test: `Tests/Research_Workspace/test_bibtex_export.py`
- Test: `Tests/Research_Workspace/test_overlay_store.py`
- Test: `Tests/UI/test_research_workspace_search.py`
- Test: `Tests/UI/test_research_workspace_menu.py`

1. Add RED tests for current-authority search only, `/` focus at all widths,
   recent/pinned/archived, create/rename/duplicate/archive/restore/delete
   ownership, Local delete redirect, three templates, collections owner link,
   overlay-only banner/split state, import/export, BibTeX escaping/dedupe, and
   no cross-authority calls.
2. Implement paginated/windowed workspace search through the selected port.
   Wide shows the field; medium/narrow retain `/` to open the same search
   overlay. Results and caches use `QualifiedWorkspaceRef`.
3. Recent/pinned, banner, and split preferences are bounded overlay schema-v5
   presentation state. Migrate v1-v4 without inventing preference values.
   Archived is canonical. Local create/rename/duplicate/archive/restore use
   registry service; Local destructive delete routes to Settings. Server
   lifecycle calls `NotesScopeService` with exact version and permission
   capability.
4. Define Literature Review, Interview Analysis, and Product Brief templates as
   confirmed creation presets (name/description/banner/starter-note fields),
   then create through the selected authority owner. Do not ship fake example
   sources or silently ingest content.
5. Customize banner and split preferences stay in the private overlay. Menu
   copy says device-only. Collections navigates to its existing Library owner.
6. Import/export reuse TASK-21511's manifest/redaction/bundle paths. Export
   BibTeX builds stable cite keys from source metadata, escapes BibTeX special
   characters, labels incomplete entries, deduplicates by canonical source ref,
   and writes only after a user-selected destination.
7. Keep server-unavailable actions discoverable with owner/reason/recovery;
   Local Share remains absent except Copy to Server and Share.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_workspace_templates.py Tests/Research_Workspace/test_bibtex_export.py Tests/Research_Workspace/test_overlay_store.py Tests/UI/test_research_workspace_search.py Tests/UI/test_research_workspace_menu.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: complete Research workspace menus"
   ```

## Task 6: Add contextual Help, operation status, and real owner links

**Files:**

- Create: `tldw_chatbook/Research_Workspace/owner_links.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/help_menu.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/operation_status.py`
- Extend: `tldw_chatbook/UI/Research_Workspace_Modules/header_region.py`
- Extend: `tldw_chatbook/UI/Screens/research_workspace_screen.py`
- Extend: `tldw_chatbook/UI/Navigation/pending_handoff_store.py` only for
  owner intents that require an exact target
- Test: `Tests/Research_Workspace/test_owner_links.py`
- Test: `Tests/UI/test_research_help_menu.py`
- Test: `Tests/UI/test_research_operation_status.py`
- Test: `Tests/UI/test_research_workspace_accessibility.py`

1. Add RED tests for guided tour, keyboard help, storage/operation status,
   telemetry feature flag, link route/intent mapping, unavailable owner,
   context preservation, implemented-only footer hints, focus relocation,
   accessible names, async announcements, and readable blocked state.
2. Help shows first-source tour, keyboard shortcuts from the active binding
   registry, storage and operation status, and telemetry diagnostics only when
   its Settings feature flag is enabled. It does not mutate workspace state.
3. Aggregate durable ingest/Copy/server operation receipts plus active
   non-durable chat/generation into one status model. Each row names authority,
   workspace, owner, state, and recovery; no UI timer guesses completion.
4. Define exact owner links:

   ```text
   Create agent task / Open in Console -> console (`chat`) with qualified workspace handoff
   ACP history / runtime remediation   -> ACP (`acp`)
   Sandbox diagnostics                 -> Logs (`logs`)
   MCP remediation                     -> MCP (`mcp`)
   Provider remediation                -> Lab (`llm`)
   Grounded-answer remediation         -> current Sources readiness or Library (`library`)
   Manage Workspaces                   -> Settings (`settings`)
   Canonical output destinations       -> Artifacts (`artifacts`) or Study (`study`)
   ```

5. Use typed pending handoffs only when a destination already supports or this
   task adds an exact claim/acknowledge consumer. Otherwise navigate to the
   real owner route with explanatory copy; do not invent a pretend deep link.
6. Screen actions remain htop-style outside text inputs, do not shadow global
   keys, and advertise only implemented actions. F1 invokes the existing
   global help path with Research context; it is not rebound by the screen.
7. Prove hidden panes do not retain focus, status changes announce without
   stealing focus, arrow controls retain exact ASCII plus full accessible
   names, and all enabled controls remain keyboard reachable in the six target
   geometries.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_owner_links.py Tests/UI/test_research_help_menu.py Tests/UI/test_research_operation_status.py Tests/UI/test_research_workspace_accessibility.py Tests/UI/test_research_workspace_geometry.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: complete Research help and owner links"
   ```

## Task 7: Prove extended owner and UI parity and close TASK-21512

**Files:**

- Create: `Tests/integration/test_research_extended_server_outputs.py`
- Create: `Tests/integration/test_research_extended_local_outputs.py`
- Create: `Tests/integration/test_research_workspace_control_parity.py`
- Modify: `backlog/tasks/task-21512 - Add-capability-gated-Research-Workspace-extended-parity.md`

1. Add recorded-contract Server round trips for Mind Map, Timeline, Data Table,
   Slides, Audio Summary, and five work products, asserting native owner plus
   workspace reference and no false client-download persistence.
2. Add real temporary-owner Local round trips for Audio Summary, Executive
   Brief, Evidence-Bound Hypotheses, and Proposal Pack; assert the four
   unsupported Local extended outputs and two structured work products stay
   unavailable with recovery.
3. Add a control inventory test comparing every audited design label to exactly
   one classification/action/Planned presentation, and mount it through the
   production screen at all six geometries.
4. Perform one bounded visual defect pass at the six shipped terminal classes,
   batch-fix focus/clipping/contrast/hierarchy defects, then run one
   confirmation pass. Do not start an open-ended polish loop.
5. Run targeted Tasks 1-6 tests, these integration files, CSS build, and
   `git diff --check`. Do not claim the full suite.
6. Review the approved spec section-by-section, scan for placeholders, compare
   all method/type names to implementation, and verify no Planned label appears
   in bindings, commands, or handlers.
7. Check TASK-21512 acceptance criteria only with captured evidence, add brief
   Implementation Notes, and set Done only after the repository Definition of
   Done is satisfied.
8. Commit exact files only:

   ```bash
   git commit -m "test: prove extended Research workspace parity"
   ```
