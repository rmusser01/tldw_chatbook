# Task 3 — explicit approval controls and responsive card

Status: implementation review pending; TASK-34566 remains In Progress. ADR-221 implements the approved design with ADR-150/161 tokens and ADR-031 bindings. Base: 3ba1706c5f21567159fe51805234f75583614f0f. No merge, push, shared branch mutation or real-profile launch.

## Behavior and interface

ApprovalDraft is ephemeral UI state. It stages only a captured row's legal decision, distinguishes deliberate raw review from default/programmatic Deny, refuses incomplete/illegal maps and supplies complete exact verdict maps. It does not grant permission. Single requests expose Allow once, Deny, More options; More options and Escape do not commit, disclosure offers scope and explicit Apply. Multi/grouped requests expose counted immediate Allow all N once / Deny all N; bulk once never falls back to a temporary grant. Mixed maps require every raw row deliberately reviewed. All commits lock controls before emitting the existing ApprovalDecided message and preserve round ID and denial behavior.

set_batch retains legacy compatibility and accepts optional view/presentation_revision. Captured revision fences answerable same-round replacement; advisory summary patches retain revision and valid queued gestures. Changed payload, replaced round, old generation and duplicate commits remain refused. Settled finishing payloads are inert. Neutral summary receives Alt+A; Enter is consumed. Original captured argument_sets never enter hot equality or eager rendering; legacy summarized previews remain.

Controls are outside the scrolling request body. Responsive stacking measures actual button intrinsic width and token gutter rather than a fixed breakpoint. Compact body token6 reserves room at80x24; new decision token28 accommodates Until Chatbook exits plus Select chrome. No shared token values changed. Approval feature rules replace a single source marker at the original cascade slot and are consumed exactly once. Canonical ds-approval-card remains singly owned. CSS bundle and token-prefaced generated screen sheets rebuilt.

## Evidence and commands

Interpreter: C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe (3.12.10). Each UI/governance invocation used Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py MODULE [selection]. It installs real_profile_guard before HOME redirection, asserts checkout module origins, owns bootstrap/config/TEMP and refuses before pytest when startup admission fails. Recorded admitted launches report startup_allowed. Exact BASE uses an ignored helper copied from the same launcher with BASE checkout constant and existing collection bootstrap markers for two first-open nodes; no BASE product/test assertions changed.

Meaningful interaction RED: missing More options and counted bulk labels,2 failed3passed. Final interaction13passed (task3_test_approval_interaction_qualified.log); ownership25passed (task3_ownership_complete.log). Unchanged resync test test_unchanged_resync_keeps_a_queued_toolbar_action_valid now asserts exactly one emitted map with the current round, complete current key set and all approve_once values. Changed calls, replacement round, old generation, clear/finishing and duplicate cases retain rejection assertions. A first combined edit was rejected by automatic approval review as weakening ownership; the narrowly scoped revised assertion was approved with explicit immediate-submit product context and all counterexamples retained. No rejected edit was executed.

Compact RED reproduced action clipping: Deny had no painted hit at80x24 and More options exceeded request width. Assertions were retained. Full headless matrix:80x24,120x40,170x48 × Inspect open/closed × dark/light, plus resize/reuse/height journeys:1passed191.70s (task3_compact_frames.log); final actual Alt+A variant1passed198.41s (task3_test_console_approval_compact_layout_qualified.log). Focused nonempty composer sentinel after Alt+A/Enter:1passed1deselected58.32s, critical80x24 and representative120x40 (task3_focus_sentinel.log); draft and real store message snapshot unchanged, no approval emitted. Twelve SVG compositor frames in task-3-frames cover legacy compatibility payloads without captured view. They are headless receipts, not native inspection or paint timestamps.

Other targeted verification: batch geometry1passed57.03s; bundle sync5passed45.30s; final all referenced ds tokens1passed7deselected7.77s. Owned ruff check10files:All checks passed; formatcheck10files already formatted; git diff --check passes.

## Exact BASE failures and limits

First-open HEAD2failed4passed47.60s, exact BASE2failed4passed57.78s: NoMatches '#console-left-rail' in unchanged ChatScreen._adapt_console_workspace_to_width and native composer readiness failure before approval rendering. Worker card HEAD32passed1failed67.28s; exact BASE selected same failure1failed32deselected57.36s: approval batch did not finish rendering. Earlier wrong-cwd BASE launch was interrupted and excluded from comparison evidence.

Design token governance HEAD7passed1failed; exact BASE selected1failed7deselected8.30s. test_active_css_extension_is_in_hex_floor correctly rejects the injected hex but regex expects components/active.css while Windows diagnostic uses components\\active.css.

Component governance HEAD13passed5failed; exact BASE selected same5failed13deselected131.83s. Exact failures:
- test_canonical_classes_defined_only_in_owning_sheet: Windows backslash sheet keys do not match POSIX registry ownership.
- test_dimension_literal_ratchet: Windows core\\_variables.tcss path misses token-file exemption; pre-existing raw dimension paths components/_agentic_terminal(1), components/_settings_splash_theme(2), features/_workflows(11) also reported.
- test_python_style_ratchet: unchanged Widgets/Library/library_skill_work_pane.py:170 static width assignment.
- test_active_css_extension_is_in_dimension_floor: POSIX membership expected against backslash keys.
- test_pin_rejects_active_css_extension_without_writing_baseline: expected POSIX path differs from Windows stderr.
No token coverage, governed membership, regex or guard was relaxed. Known external RemoteRoot F821 and pytest/Pydantic/audioop deprecation noise remain. No full-suite claim. Native/browser presented-frame timestamps unavailable; no latency/p95 qualification. Actual Windows virtual dispatch root identity gap remains open under unchanged BASE; no runtime/security repair here.

## Self review and changed scope

Reviewed legal-value rejection, raw same-value deliberate overlay event, scope Apply, immediate counted bulk, stale owner fences, duplicate locking, finishing, exact denial/round semantics, hot equality, CSS cascade and painted bounds. Reused native Select, Button, Static, existing card pagination and generation button helpers; no dependency or UI framework added. Legacy previews remain fallback. Captured request facts are Task3; later Details original-body pages and Task5 settlement/grant/execution feedback remain separate.

Owned source: approval_controls.py, chat_approval_card.py, chat_task_cards.py, build_css.py; source CSS _agentic_terminal.tcss, _console_approvals.tcss, _variables.tcss; main generated bundle and seven generated screen sheets; interaction/ownership/card/compact/first-open tests and finite guarded launcher allowlist. Evidence consists of this report, selected logs and12SVG legacy frames. Task implementation notes and commit references will be recorded after final scope review. Independent review and final qualification remain required before TaskDone.

## Final captured request qualification

Controller clarified captured action/target/profile/location/current scope is Task3 request presentation, while deferred feedback means Task5 outcomes. RED child assertion: Read file absent from Legacy server label · fs_read (wrapper CP1252 decoding failed; inspected UTF8 child assertion directly). Implemented literal captured header; local/builtin identifying targets complete, other generic targets bounded256UTF8bytes, raw command referenced rather than duplicated. Current scope tracks Select changes. Original argument_sets not expanded. Legacy header remains fallback.

Actual _build_approval_payload fixture revision7/profile Writer/location Project scratch/action Read file/target notes/owner-target.md covers80x24 Inspect/dark and120x40 noInspect/light. Final painted header viewport and hit-testing plus all controls, Alt+A/Enter draft sentinel:1passed1deselected59.78s (task3_captured_painted_final.log); earlier captured run1passed63.95s. Two captured-*.svg frames supplement12 legacy frames. Header facts and scope are checked; original body sentinel absent. Final interaction13passed8.59s, ownership25passed11.58s; final owned ruff check+format10files and diffcheck pass. TaskCLI implementation notes saved through cached1.50.1 CLI; status verifiedInProgress. No ACs or DoD completion claimed.

Commit: explicit owned sources/tests/sourceCSS/generatedbundles/tasknotes plus report, selected receipts and14headless frames; hash reported separately below after commit.

Committed text logs and SVG whitespace-only template lines have trailing whitespace stripped for diffcheck; assertion content and geometry unchanged.

Implementation/evidence commit:363c4ce3244c92eef0dce0894d1d6ebd1d9baaae. Frame directory also contains three reviewer-rendered PNG derivatives included with14SVG receipts; these remain headless exports. Post-commit review handoff retains Task34566 InProgress.

## Independent review fix round1/5

FIX_BASE:2b2a0322c60d7629c350b8d2d6fa832d4c6eda29. All six Important findings addressed; Task34566 remains InProgress pending independent re-review and existing qualification limits.

1. Capture derives display targets only after existing MCP redaction of keys, nested mappings/sequences and secret-shaped values. Virtual argv also uses the existing CLI redactor. Canonical captured argument_sets remain exact originals for matching; display redaction grants no permission.
2. Captured/reused header preserves the existing path-precheck warning, including will fail even if approved; it remains literal. First80x24 warning snapshot passed initial viewport/hit assertions before the later variant failure. No warning or raw full-command contract removed.
3. Single Apply names Allow once / Until Chatbook exits / Remember these inputs / Remember this tool / Deny. Mixed Apply has a pinned complete allowed/denied count and tool-bound captured scope summary from ApprovalDraft.summary. Scope appears once per row before bounded request previews; row header no longer duplicates consent copy.
4. Identifying paths, URL and virtual command+argv targets remain complete. Generic no-identifying-key arguments are explicitly Parameters preview, with an omissions marker at256UTF8bytes. Full generic Arguments pages remain Task4; original bodies never expand in collapsed UI.
5. Narrow bulk controls use distinct Allow all once / Deny all labels and pinned associations with each exact count and outcome. Exact complete decision maps remain approve_once or deny respectively; one-time bulk never falls back to broader scope.
6. Task3 private additions were derived exactly from3ba1706c5f..FIX_BASE and removed from INDEX only. Local files remain. Durable receipts are curated under Docs/superpowers/qa/2026-10-05-console-approval-ux/task-3/: this report, verification excerpts, migration list and four headless SVGs. Redundant PNG derivatives and full logs remain private/untracked; SVG external font declarations stripped from curated copies.

Minor owned fixes: repaired UTF8 mojibake; removed header/row scope duplication; Escape returns to the still-current More options opener only with widget membership, generation, mounted and actionable checks. Provider import census exposed Task3 eager import: widget annotations and draft captured scopes now resolve lazily; no new module/runtime framework.

Focused meaningful RED: UI six regressions failed5.34s; capture secret redaction1failed1passed0.89s; tool-bound scope summary1failed3.79s. Initial GREEN UI17/19 left narrow geometry failures in an unstyled widget-only host; only new painted cases were changed to shipping APP_STYLESHEETS, retaining all bounds/hit assertions. Final interaction20passed15.44s; presentation18passed5.22s; card32passed1known-worker-node deselected9.24s; ownership25passed11.43s; bundle5passed38.66s; tokenrefs1passed7deselected7.33s; hex/spacing2passed6deselected14.06s; owned Ruff check/format6files pass.

Actual _build_approval_payload Console cases: initial local warning; MCP nested/key/shape secrets; two MCP paths differing after256bytes. Critical80x24 Inspect/dark and representative120x40 noInspect/light, exact painted action bounds/hit tests, literal captured facts, suffix discoverability, original-body absence and Alt+A/Enter nonempty draft/message snapshot:1passed1deselected68.10s. Four captured-variant SVGs are compositor exports, not native timing.

Superseding attribution: an early variant fixture incorrectly reused round+revision7 for a changed snapshot; distinct variant round IDs corrected the contract. With correct rounds, previous Tab gestures still left inherited viewport offset1 (header y10 vs viewport y11). An explicit viewport Home gesture inspects subsequent metadata; long target inspection then scrolls to its identifying suffix. This proves discoverability after inspection, not initial visibility of reused metadata. Initial warning80x24 evidence is separately recorded above. Speculative production reset was removed; no scroll/focus policy change is claimed. Same-round revision replacement, unchanged-summary/queued-action preservation, clear/finishing/duplicate and exact denial/round assertions remain.

Commands: primary Python -m Ruff check --no-cache and format --check --no-cache on the six changed source/test files; guarded private_control.py targeted interaction/presentation/ownership/card(-k not action_bar_is_actually_visible)/compact(-k focus_enter)/bundle/token selections. No full sweep. Retained exactBASE failures and native/Windows virtual dispatch gaps remain unqualified; warning noise remains. No storage/root/recovery guard repair, grant/backend changes or speculative timing claim.

Self-review: checked display redaction versus exact original retention, complete identifying targets versus marked excerpts, warning preservation across reuse and selection, distinct count/scope commits, duplicate/old-owner guards, literal UTF8 and private index-only migration. Old private report/log references in historical sections mean locally retained untracked diagnostics; durable current receipts and task notes point here.

Final ownership rerun after all fixes:25passed10.16s; final owned Ruff check/format6files and diffcheck pass. Exactly37private additions untracked; all local files verified present after index removal;211previous sibling/old .superpowers tracked paths remain. TaskCLI notes now link durable report and retainInProgress. No TaskDone claim. Fix source commit hash recorded after commit.

Fix implementation/artifact migration commit:e6ed025cfe1b878028160c360eec1c59fe50089c. Post-commit tracked checkout clean; local private report/evidence remain present and ignored. Independent re-review pending.


## Fix round 2 / 5 — compact mixed-scope summary (inline)

The collaboration harness refused both original-implementer followup and fresh replacement with agent thread limit reached. Executing-plans inline fallback now applies; independent re-review is unavailable and no full review certificate is claimed.

RED: guarded private_control.py Tests/UI/test_approval_interaction.py -k many_tool_scope_summary: 1 failed, 20 deselected. Actual Apply region was y=42 outside the 24-row screen. The test exercises ten distinct real captured tool rows with temporary choices and painted bounds/hit targets.

Changed ApprovalDraft.summary to offer a compact count-by-scope projection for the pinned area; the existing full per-tool summary contract stays available. Tool subjects, profiles and complete selected scope explanations remain in the bounded scrolling request rows. The pinned text keeps complete allowed/denied counts and selected scope totals; multiple profile names are explicitly reviewed in their rows rather than concatenated into an unbounded footer. Original arguments, verdict maps, review flags and ownership fences are unchanged.

GREEN focused many-tool/mixed/fallback/full-summary selection: 5 passed, 16 deselected. Final targeted whole interaction file: 21 passed; ownership file: 25 passed. Each retained the pre-existing pytest/Pydantic warnings. Changed-file Ruff check passed, format check reports three formatted files; diff-check passed. No CSS/shared token or authority/runtime change. No full sweep, perceived timing or native inspection claim.

Self-review: compact pinned counts group by existing decision strings; full scope and exact profile/subject remain visible in their existing row surfaces. The new test verifies actual styled painted Apply, all ten tool/profile row labels, selected row scopes, one complete exact approve_session map and current round. The author verified the fix; a fresh independent review cannot be obtained under the harness thread limit. Full Task3 DoD stays open alongside documented BASE/platform qualification.
