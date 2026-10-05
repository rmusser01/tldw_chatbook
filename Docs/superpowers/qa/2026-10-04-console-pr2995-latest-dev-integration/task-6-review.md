### Spec Compliance

- ❌ **Issues found:** the new unpublished-message scanner exemption does not remain fail-closed after publication through `setattr`. This misses the brief’s narrow publication/alias requirement; see Important finding I1 at `Tests/Chat/test_console_fork_transition_census.py:369`.
- The remaining inspected Task 6 source corrections match the brief: canonical mutation ownership, physical display-worker custody, typed restoration initialization, exact rollback contracts, fixture-only baseline repairs, and formatting/comment-only size corrections. Reviewed source: `71cc9db79fd802a0fc1c656f3186c415e2e844a9`; pinned dev: `f1f80847a410525ce1b00d27ea0e8be824a98422`; reviewed BASE: `b265d5bd2c2f5a28e4f56c2115f6247988d29ac2`.
- ⚠️ **Cannot verify from this scoped diff:** unchanged Tasks 1–5 behavior and the earlier whole-feature conclusions are carried by the supplied source mappings, not independently exercised again. Current-head Qodo, required CI/PerfGuard (including the historical native timeout), fresh dev ancestry and ADR219 allocation, publication, and merge remain root gates. This is no merge-success claim.

### Strengths

- The six-route tests establish eligible sources, pause after actual field assignment, verify same-session refusal and other-session eligibility, then check balanced owners after success and exceptions. Validation precedence and preparation→promotion nesting have separate controls. Evidence: immutable package lines 640–1035; source wrappers at package lines 2899–3556; `task-6-public-boundaries-all-routes-red.json` records 12 actual failures, followed by the 81-case qualification and five validation/lock controls.
- Display-name preparation acquires a token before live mutation; acceptance or explicit abandonment releases that exact session/token. The coordinator retains the child task across pre-start cancellation, sibling failure and repeated cancellation, drains its physical owner, and preserves the primary exception. Evidence: package lines 3193–3336 and 4006–4135; focused lifetime check of `chat_screen.py:5612` and `console_chat_store.py:13543`, `:13818`, `:13863`, `:16467`. No new persistence operation or retry was introduced.
- Fresh project controls move into the typed restore/create boundary, with invalid input rejected before publication and the durable/legacy default retained. The new-chat caller supplies `ProjectInstructionControlState.new_session()` directly. Evidence: package lines 2617–2898 and constructor tests at package lines 819–867.
- The rollback guard pins the three submission branch owners, guards and arguments, with adversarial changes for each; typed settings-drain provenance has explicit registry controls. Evidence: package lines 1753–1860 and 1980–2083. Bidirectional route equality and prior protected-path assertions remain.
- Preservation summaries account for the integration rather than silently normalizing semantic changes: the initial 1,838 non-overlap blobs are exact; the four initial non-equivalent Python paths are exactly the four canonical diffs; later upstream/owned exceptions are named. The final manifest records 11,572 historical QA blobs exact, 163/164 owned Python paths unchanged, the navigation exception, and the exact diagnostic union. Evidence: `task-6-preservation.json`, `task-6-boundary-preservation.json`, `task-6-latestdev-final-preservation.json`, `task-6-reader-final-preservation.json`.
- Provider and Reader fixture changes retain their original assertions/bodies. Navigation now proves a single content Console, runtime/top-screen ownership, the active/visible first session, and actual typed `half` before the original public navigation assertions. Evidence: package lines 2084–2367, 7727–7963 and 7971–8326; `task-6-provider-assertion-proof.json`, `task-6-reader-navigation-assertion-proof.json`, `task-6-reader-repair-proof.json`.
- Reader/Skills formatting and comment repairs preserve strict ASTs and diagnostics. Reader stays at 967 lines; Skills returns to 3142 without raising either cap. The rewritten comment preserves task-8/15457/15790, missing-widget/no-retry behavior and canvas post-recompose ordering; the asynchronous keyring initialization remains. Evidence: package lines 8441–9224 and `task-6-reader-repair-proof.json`.
- Upstream round-robin sharding, fail-closed aggregation, BaseAppScreen opt-in behavior, lazy Reader freshness and exact-identity recheck are retained. The 557 pre-import threshold is the upstream ADR097/212 exception, with no additional feature allowance. Evidence: package lines 4208–4757, 5032–5109, 6523–7686; `task-6-reader-budget-and-doc-proof.json`.

### Issues

#### Critical (Must Fix)

- None found in this scoped review.

#### Important (Should Fix)

- **I1 — `Tests/Chat/test_console_fork_transition_census.py:369`: `setattr` publication bypasses the new escape check.** `_detached_receiver_before` skips every `setattr` call. A newly constructed message published as the third argument therefore remains classified as detached, so a subsequent unfenced fork-field write disappears from `_mutation_events`. The same happens through an alias. This is a regression in the census safeguard added by this task, not a demonstrated current production race.

  Focused cache-free probe using the actual extracted scanner helpers:

  ```python
  def mutate(self, session_id):
      message = ConsoleChatMessage()
      setattr(self, "published", message)
      message.content = "changed"
  ```

  Actual scanner output: `()`. Replacing publication with `self.publish(message)` returns `((4, None),)`. `alias = message; setattr(self, "published", alias)` also returns `()`. Publishing into an owner through `setattr` must invalidate the message and its aliases just as an ordinary publication call does. Narrow the exemption to genuine mutation of a detached receiver; inspect the assigned value for escape. Add direct/alias publication controls and retain a control allowing scalar `setattr` on an unpublished message. No broad scanner rewrite is needed.

#### Minor (Nice to Have)

- **Retained warning — `task-6-latestdev-baseapp.log`, documented in `task-6-latestdev-fd-disposition.md:3`:** the mounted BaseAppScreen owners passed 43 assertions but reported FD growth 206 (14→220, limit 200). The nested TldwCli fixture is only a plausible owner; no descriptor attribution was established. This is inherited qualification noise, not a Task 6 blocker or proof of a production leak. Preserve it for a separately scoped ownership diagnosis; do not change the sentinel or suppress it.
- **Intentional budget notices — `task-6-reader-startup.log`:** three ADR097 headroom warnings remain (imports 681/686, UI-ready 1033/1033, pre-import 557/557). These are designed pass diagnostics, not failures. The unchanged limits and honest warnings are appropriate.

### Focused Checks Performed

- Read the authorized task/review briefs, all three phase reports, the supplied immutable diff in sequential sections, and linked preservation/exception summaries. Checked package SHA256 equals `9d640baf899b44ccc0ffc9e45e1b9bb180f7391f872f2f69b25c2df528601110`.
- Named outside-diff risk: a cancelled coordinator could abandon a token while physical persistence remains. Read the cut-off beginning of `_coordinate_console_settings_submission`/`persist_display_name` and the exact serialized persistence, acceptance, abandon and fork-owner helpers cited above. Their lifetime contract supports the new cleanup.
- Named focused correctness doubt: blanket `setattr` handling could hide publication. Ran one small AST-only reproduction with the repository Python and `-B`; it imports no application/test module, creates no cache/profile, and produces the I1 results above. Two preliminary harness setup attempts failed before the probe ran (missing synthesized AST location, then older system Python annotations); neither is test evidence.
- Inspected existing exit/log receipts for 81 boundary/caller/neighbor cases, 43 census cases, five validation/lock controls, three repaired Reader nodes, 25 startup cases, the public navigation node, diagnostic inventory, UI census, fatal Ruff, formatter and whitespace. Confirmed preserved RED outputs for the 12 actual fork races and three immutable-dev Reader failures. No passing suite was replayed.
- No Git commands, subagents, source/index/cache mutations, installs, profile changes or budget/guard changes. Only this review report was written.

### Assessment

**Task quality: Needs fixes.**

The production integration changes have clear ownership and substantial targeted evidence. The new scanner exemption must handle `setattr` publication before its claimed fail-closed protection is trustworthy; no other blocking finding was identified in this task scope.
