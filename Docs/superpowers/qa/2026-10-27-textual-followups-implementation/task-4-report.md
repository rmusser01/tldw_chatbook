# TASK-37034 implementation report

Status: implementation complete; task stays In Progress for root review and integration. Local behavior gates pass with explicit unchanged guard qualifications below. No remote, PR, CI, child-agent, full-suite, or primary-checkout operations were performed.

## Scope and ownership

Worktree: `C:/Users/GDesktop-1/.codex/worktrees/textual-settings-patterns/tldw_tui`; branch `codex/textual-task-37031-37034`; assigned base `c375f164e6955c68c90c3d5e75a6c936023dacbc`. All runs use `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8` from that worktree. The supplied runners establish native private TEMP/profile parents before pytest import, disable third-party plugin autoload, load only pytest_asyncio/pytest_timeout, and retain the unchanged 180-second per-test budget.

`STTSScreen` clears/revokes its old window owner before native screen recomposition and restores a still-attached owner if the native swap is refused. It now derives custom-footer registration and native action availability from both the current Playground view and the actual target under its attached, current `STTSWindow`. The existing current-view watcher removes hints immediately on navigation. A small typed `STTSWindow.ContentChanged` message follows actual native content removal and successful replacement/descendant readiness. The screen accepts only its exact attached window owner and re-reads current target presence; late messages contain no stale view/hint snapshot to apply. No on_mount override, global registry, provider/audio work, style, token or dependency was added.

Root also authorized the same-AC recomposition wrapper after its separate clean RED. Root explicitly authorized the additional existing `UI/STTS_Window.py` notification seam after the original RED. ADR required: no; ADR path: N/A. This directly implements existing widget-message ownership under `backlog/decisions/011-chatbook-workbench-ui-system.md` and advertised==working active-context truthfulness under `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, preserving `backlog/decisions/150-design-token-system-and-design-language.md` and `backlog/decisions/161-component-pattern-library.md`. The task plan was updated through the CLI before editing either product file.

## Test behavior and original RED

The first new production-CSS mounted test performs ordinary Playground rail Clear using `x`, focuses the actual Effects rail and presses Enter, reads the actual `#footer-key-quit` Static child content, checks all five absent hints and native bindings, presses the formerly inert keys, exercises Lab mode focus, returns through the actual Playground rail Enter, and verifies restored rail Clear and printable `grxps` typing in the actual TextArea. A narrow wrapper parks only the native target mount on the return: the real view worker and state machine run, hints/bindings stay absent while no pane exists, and the native mount resumes after release. The release is always set and native workers are drained in finally. No generation, recording, playback or provider service is invoked by this journey or by availability computation.

Original RED command (before product changes):

```powershell
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted.py task4-red01 Tests/UI/test_speech_footer_scope.py
```

Actual parent: `.test-tmp/textual-37031-37034/task4-red01/result.xml`, 1 case / 1 failure / 0 errors / 0 skips. Actual private child: `task4-red01/pytest/test_footer_and_rail_actions_f0/private-profile-process/pytest.xml`, 1 case / 1 failure / 0 errors / 0 skips. Ordinary rail Clear and real Effects Enter qualified before the sole failure: `assert 'g generate' not in painted`, with the actual footer child still containing all five Playground actions. The child pins source resolution to the assigned worktree. The first RED used record_property and therefore also records an observer-only xunit2 warning; final tests append user_properties directly instead.

## Additional observer failures and admitted repairs

All intermediate failures remain in their original run directories and are excluded from final all-green claims:

- `task4-existing-baseline01`: 6 parent cases, 5 passed / 1 failed / 0 errors / 0 skips, stopped with -x at old `test_screen_level_random_text_mirror_reaches_the_mounted_pane`.
- `task4-rail-baseline01`, `task4-effects-baseline01`, `task4-mount-baseline01`: each 1 parent case / 1 failure / 0 errors / 0 skips, stopped with -x at `test_the_body_owns_view_switching`, `test_the_rail_entry_is_reachable`, and `test_speech_rail_exposes_and_opens_voice_profiles`, respectively.
- Each of those source-profile refusals follows `_build_test_app -> app_factory.build_test_app_config:125 -> config.load_settings:2214 -> config_participants.wrapped:505/operation:436 -> raw_participants._scope:945/_raw_participant:357/_register_raw_participant:307 -> RecoveryRequired(raw_source_selection_changed)`. Root approved supported bootstrap_profile opt-ins for exactly the four inspected modules; none deliberately changes selected profiles. Existing fixture reset/admission behavior and substantive assertions remain intact; no blanket plugin/mark was used.
- `task4-green01`: preliminary new journey parent 1 passed, actual private child 1 passed. Superseded by the final-source runs below.
- `task4-behavior01`: 21 parent cases, 19 passed / 2 failed / 0 errors / 0 skips. All 14 actual private children passed (new journey plus 13 Recognition button controls). The only failures were the two old shortcut tests waiting for the pane before awaiting the native Studio load worker, before pressing their first action key.
- Root authorized one custody-controlled comparison of the two exact BASE product files. `task4-shortcuts-original01` reproduced the same two pre-key short-wait failures, 7 parent cases / 5 passed / 2 failed / 0 errors / 0 skips. The process ran serially with no competing App/source readers. Exact current native bytes were preserved and restored in finally; `.test-tmp/textual-37031-37034/task4-static/original-custody.json` records original Git/native and restored hashes. Old shortcut readiness now awaits the real window native mount event and the already-existing native workers before querying the pane; key assertions and all time/attempt allowances are unchanged.
- `task4-controls01`: 13 parent cases / 12 passed / 1 failed / 0 errors / 0 skips, stopped with -x. All seven shortcut cases and five rail cases passed. The existing `test_the_playground_view_mounts_a_playground` then raised NoMatches after two fixed pilot.pause calls, before the Studio worker completed. Root approved the same native readiness synchronization in that exact assertion. This rail failure is source-qualified readiness evidence, not a separately original-runtime-proven failure. The separate existing MRO test now also waits actual target readiness before requiring combined hints, preserving its exact-one-rail and footer assertions.

Commands are retained in each run's `command.json`; complete raw output and actual XML are retained under the unique run directories. No missing/rotated receipt is reconstructed.

## Final validation

Final GREEN command on final native product bytes:

```powershell
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted-stream.py task4-final02 Tests/UI/test_speech_footer_scope.py Tests/UI/test_stts_capability_state.py::test_speech_surfaces_share_fresh_local_dependency_snapshot Tests/Architecture/test_on_mount_super_guard.py Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size Tests/Performance/test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget -rP
```

Actual `task4-final02/result.xml`: **9 parent cases / 9 passed / 0 failures / 0 errors / 0 skips**. Its two actual private children each contain **1 passed / 0 failures / 0 errors / 0 skips**:

- `task4-final02/pytest/test_footer_and_rail_actions_f0/private-profile-process/pytest.xml`
- `task4-final02/pytest/test_recompose_hides_cached_hi0/private-profile-process/pytest.xml`

Both child runtime-source hash sets were independently compared with the current native file bytes after completion and match. The final private scenarios cover Effects Enter/ordinary and returned rail Clear/printable input, held native return mount, and actual screen recomposition with both pre-removal and detached old-owner messages. The recomposition scenario gates and then forwards the real native removal and Studio load; it never replaces their state transitions or results. All releases and worker settlement run in finally.

The additional recomposition TDD command, before adding STTSScreen.recompose, was:

```powershell
& C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/textual-37031-37034/run-targeted-stream.py task4-recompose-red01 Tests/UI/test_speech_footer_scope.py::test_recompose_hides_cached_hints_and_rejects_old_window
```

Actual parent and actual child each recorded **1 failure / 0 errors / 0 skips**, exactly `Recompose retained cached hint` for `g generate` in the actual custom-footer child after a queued old-owner message while native removal was held. No product mutation was used to manufacture this RED.

Final module budgets remain unchanged and pass: boot **680/686** modules; preimport **558/558** modules, **424406/425347** LOC and fattest route **130690/135111** LOC. Preimport has zero module headroom and 941 LOC headroom; no allowance was raised.

Relevant old behavior evidence, with exact selectors/commands in the manifest and every named run's command.json:

| Cases | Actual qualifying execution |
| --- | --- |
| Seven supported/reserved/mirrored shortcut cases | task4-controls01: all 7 passed after native readiness synchronization |
| Seven rail/navigation/dirty-draft cases | Five passed in task4-controls01; mounted-target and dirty-draft controls passed in task4-final01 |
| Three Effects placeholder cases | Two passed in task4-final01; native-drained rail reachability passed in task4-amended01 |
| Seven capability/MRO cases | Five passed in task4-final01; exact-one-mount passed in task4-amended01; fixture-consistent fresh snapshot passed in task4-final02 |
| Thirteen Recognition button-delivery cases | All 13 actual private children passed in task4-behavior01; no child errors/skips |

These are qualified scoped results, not a claim that the red intermediate batches passed as a whole. Root explicitly directed against repeating already-passed heavy cases without a new concern. The later production addition is confined to STTSScreen.recompose; the final AST proof shows all other previously tested method bodies unchanged, and the new native recompose scenario directly exercises that addition. The final two new scenarios plus the 37 old behavior cases cover 39 distinct behavior nodes across these executions.

Fresh unchanged controls also passed in task4-final01: native MRO runtime count, all four select-option source guards and all four unavailable-global-footer widths. The four BaseAppScreen/Wizard mount guards were freshly green again in task4-final02. No all-suite or unexecuted behavior claim is made.

Additional qualification of task4-final01 and task4-amended01:

- task4-final01 has 30 parent cases / 25 passed / 5 failed / 0 errors / 0 skips; its actual private child passed. Three old observers first failed on native mount readiness (Effects teardown SelectCurrent `#label`, fresh snapshot, single-Mount target). The admitted native settlement fixes passed in subsequent runs. This is observer/teardown repair, not a claimed production Select fix.
- task4-amended01 has 3 parent cases / 2 passed / 1 failed / 0 errors / 0 skips. The fresh-snapshot test then reached a previously hidden stale expectation: actual Local 4/5 vs expected 4/4. Existing product capability definitions already count OmniVoice as a fifth capability; the fixture omits both onnxruntime and tokenizers. Root authorized changing only the expected summary to 4/5 and adding explicit `Local OmniVoice: missing` inspector evidence, keeping all four ready-row assertions and the partial-readiness fixture. Unchanged capability files are byte/source-qualified in source-static-final.json. No fifth-pane row or product presentation change was added.
- The broad unchanged `Tests/UI/test_on_mount_mro_convention.py::test_no_screen_reintroduces_super_on_mount` still reports five pre-existing sites: console_endpoint_template_modal.py:575, response_rules_modal.py:176, buddy_character_review.py:186, buddy_conversation_modal.py:138, buddy_management_modal.py:372. `task4-static/additional-guard-identity.json` proves the guard and all five offender files LF-normalized BASE-identical, with native/Git SHA256. This is **not** attributed to slash normalization; _ALLOWED_OFFENDERS is empty. No unrelated source or guard was changed.
- The unchanged 8-second import elapsed guard failed in task4-final01 at **15.108830100041814 s**, heavy modules `[]`. Root authorized one exact two-product-file original-source custody comparison using the same test and environment. `task4-import-original01` recorded **1 parent failure / 0 errors / 0 skips**, at **16.361436699982733 s**, heavy modules `[]`. The exact current native bytes were restored in finally; import-original-custody.json proves original Git/native and restored hashes. This qualifies the failure beyond this Task4 change; it does not claim a passing timing gate or substitute for root's fresh current-head/hosted gates. No threshold/guard/environment allowance was changed and no blind candidate rerun was performed.

`task4-static/all-xml-manifest.json` lists every actual parent/private-child XML path, SHA256, node name and result for all 14 retained run directories, along with exact underlying pytest commands. It distinguishes wrappers from actual child reports; it does not infer child counts from wrapper dots.

## Runtime identities and native hashes

The final-source mounted child asserts each loaded module resolves beneath the assigned worktree and writes these native byte SHA256 identities into its actual XML properties:

| Runtime source | SHA256 |
| --- | --- |
| UI/Screens/stts_screen.py | 37013085399f0e4acb614ddac2476b4b4295585924433f642856af0400b68cb9 |
| UI/STTS_Window.py | 47a5aa9bc4cedf3c17633e386cf1cd309e203bafc4412f2cb85fa358ec571268 |
| UI/Speech/speech_playground_pane.py | 27158c125c1cbebfefc1dddb446767aead5371db0101256ebaf4620dcc71ad82 |
| Widgets/AppFooterStatus.py | ad2db731205f9b290afa164c99c1ff2d1c69fdd80a38ccc561f4cb288db46a0f |
| Utils/platform_files.py | b1c43ddecc259efebdc3b37f6d4c767819fead2372bf9d4cd7d8a05db03429e3 |
| Utils/windows_files.py (actual WindowsOS class) | af62e79f163c4f43bebcac238dd315f177719b22e12c87437039b520363a29e8 |

Actual final-source Effects child text: `←/→ [ ] Move mode focus | Enter Go | F1 help · F6 next pane · Ctrl+P palette · Ctrl+Q quit`. Actual returned child text: `g generate | r random text | x clear | p play | s stop | ←/→ [ ] Move mode focus | Enter Go | F1 help · F6 next pane · Ctrl+P palette · Ctrl+Q quit`.

## Static, style, and prior guard qualification

`task4-static/source-static-final.json` compares final method ASTs against assigned BASE: only `STTSScreen._lab_footer_registration`, `_playground`, `_sync_rail_active` changed; `_speech_content_changed`, `check_action`, and `recompose` were added; 20 other screen methods are identical. In STTSWindow only `_mount_view_unlocked` changed and the nested message constructor was added; 85 other methods are identical. Import/BINDINGS ownership and existing source/import limits are preserved.

Normal Ruff reports exactly the same seven inherited E721 findings as BASE (two screen, five window), with no new findings. `ruff check --ignore E721` across all seven touched Python files passes (`task4-static/ruff-focused-final.log`). These exact-type checks are intentionally preserved rather than changed during this bounded UI repair. Ruff format passes on the screen, new test, shortcut tests, Effects tests and capability tests. Existing STTS_Window.py and rail-navigation formatting debts remain format-check failures in both BASE and current source (`task4-static/source-static.json`, `format-final.log`); no unrelated whole-file reformat was applied. A small mechanical formatting change to an already-owned shortcut assertion preserves its AST. Scoped git diff --check passes.

No stylesheet, token, or styles assignment changed. `task4-static/unchanged-visual-source.json` independently proves 87 visual governance/input/style sources LF-normalized identical to assigned BASE. The two actual native guard source hashes match prior evidence: design-token guard `86ee4a2965a4cab738235e2e2dc57bcd248fb4ae70ebb366bb8d219f7d264a0b`; component-pattern guard `5ce3470d06dcaca40c641af6fdbcfd25a6825457c56e8f711477626f94e8451f`. Prior receipt `task1-static/guard-baseline-identity.json` SHA256 independently rechecked as `3934894d69fdfb481602386e60664451045813298683a07135b39f5ea3640c33`.

Per common protocol/root ruling, these unchanged visual suites were not repeated. Prior actual evidence remains 30 parent cases / 25 passed / 5 failed / 0 errors / 0 skips and 26 actual private children / 21 passed / 5 failed / 0 errors / 0 skips. The exact inherited Windows slash-assumption failures are:

- Tests/UI/test_design_token_governance.py::test_active_css_extension_is_in_hex_floor
- Tests/UI/test_component_pattern_governance.py::test_canonical_classes_defined_only_in_owning_sheet
- Tests/UI/test_component_pattern_governance.py::test_dimension_literal_ratchet
- Tests/UI/test_component_pattern_governance.py::test_active_css_extension_is_in_dimension_floor
- Tests/UI/test_component_pattern_governance.py::test_pin_rejects_active_css_extension_without_writing_baseline

This is explicit unchanged-source qualification, not a new green execution claim. Relevant current mount/keyboard/source/import controls are recorded above. Inherited Pydantic Field extra-keyword and pytest-asyncio loop-scope warnings remain; applicable old App runs also reported splash invalid-escape and pydub audioop deprecations. Logs preserve pre-existing app-factory missing-service messages and optional sounddevice/WebRTC notices. Output is not claimed pristine.

## Self-review, limits, and delivery

Self-review verified BaseAppScreen.recompose and native Widget.recompose do not call save_state/restore_state/export_state; STTSScreen.save_state remains unchanged. The native self-await refusal guard still runs through super, with still-attached owner recovery in finally. Current-view/live-axis persistence beyond existing screen construction semantics is explicitly unclaimed; root rejected optional axis snapshot expansion as outside this AC. Self-review checked exact owner rejection, current target derivation for late notifications, no provider/audio side effects from availability, actual custom child text, native absent/restored bindings, retained printable routing, initial deferred mount and return interval, MRO hook preservation, no new source/import allowances, and exact owned-file staging. Full app/server/audio-device operation is outside this task's no-audio mounted qualification; full-suite/CI/whole-branch independent review and task closure belong to root.

No new generalized lesson was invented: the source-profile/native readiness incidents reuse the already-read lessons for private profile selection and worker-mounted widget readiness. Documentation was updated in Docs/Architecture/speech.md. The task remains In Progress with root responsible for AC/status completion and integration.

Implementation commit: `1be8428cd7dffbd5c7c7de870454544b4306d1e3` (`fix(speech): scope Playground footer actions to live targets`). Exactly nine task-owned versioned files were committed. The assigned report and raw evidence remain local plan artifacts; root-owned scripts/ui_pr_gate_census.txt and other task/sibling files were not staged or changed. Post-commit status contains only the root-owned census modification and shared .test-tmp artifacts. Task status remains In Progress.
