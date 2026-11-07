# Narrow ordinary/plugin reference integration review

Date: 2026-11-05. Independent source review only. This supplements, and does not replace, `independent-integration-review.md`. Product paths below are relative to this directory's `integration/` export; line numbers identify the pre-fix source read for this review. No tests were rerun, no subagents were used, and no product, test, Git, profile, PR, or CI state was changed.

**SPEC: CHANGES REQUIRED. QUALITY: CHANGES REQUIRED / HOLD. Narrow finding counts: C0 / I1 / M0.** This is an additional Important finding to the initial checkpoint finding: cumulative known findings before either fix is reviewed are C0 / I2 / M0. The proposed composition described below is appropriate in principle; implementation and RED/GREEN evidence remain pending. Neither proposal approval nor earlier passing cohorts close this finding.

## Important: plugin reference expansion reads files before ordinary admission

**Locations:** `tldw_chatbook/Chat/console_chat_controller.py:10168–10178`, `tldw_chatbook/Chat/console_plugin_context.py:398–429`; the second expansion is `console_chat_controller.py:10318–10344` through `console_reference_preparation.py:376–402`.

`_prepare_native_references()` builds a fresh resolver and runs `_expand()` through bare `asyncio.to_thread()`. Its caller has not yet run `prepare_agent_sources()` or `admit_ordinary_turn()` (controller lines 10270–10297). A manual draft such as `@visible.txt` therefore reaches the actual file-expansion callback before the ordinary execution owner and worktree claim exist. The controller then discards the first expanded text and records and expands the same raw draft again through the original, owned preparation receiver. This is a real unowned file effect, not merely a missing test-side owner lookup.

The bounded `ordinary-first-failure-receipt.json` records exit 1 in 9.4565466 seconds with 9,108 source files unchanged (`4cbf0df2704411f4926fb08a784e6eab04518b9aaac155d536b1bdcefc50854d` before and after). The failing case is `test_conflicting_actual_reference_refuses_other_turn_before_row_or_model`. `ordinary-reference-stack.log` identifies two paths into the observed `_expand_path`: first via plugin `_expand` line 405, where `current_execution_owner` is `None`; then via `console_preparation_reads` and `console_reference_preparation`, under custody. In a conflicting turn, the first path can read before the ordinary conflict refusal. Bare `to_thread` also lacks the original preparation receiver's tracked worker-return and cancellation lifetime.

There are connected semantic defects: the early helper suppresses literal native command/leading plugin skill arguments and removes a leading manual-rule selector, while the later controller rescans every raw reference without those filters. Further, `reference_offsets` come from the first expansion while provider text comes from the second. A file that changes length between those reads can move embedded skill substitution to the wrong coordinates. The unconditional reset at controller line 10318 also discards the helper's preserved resumed executed text. These belong to the same split-expansion defect, rather than separate finding counts.

**Scope qualification:** the stock early helper passes bare `run_git_reference`; the stock runner refuses when allowed roots are absent. The evidence establishes pre-admission file expansion; it does not establish that this stock early path launches Git. The later original receiver supplies the authority-bound Git runner and native producer.

## Smallest complete composition

The writer's communicated Ruling 14 proposal addresses the right boundary: retain `_prepare_native_references` as pure metadata/selector/continuation preparation, returning a finite selected candidate tuple in its private result. Pass that exact tuple to ordinary admission and to the original `_ReferenceRead` receiver. Produce expanded text, records, and offsets in one pass inside that receiver's existing activation and worker custody. A private preparation result can carry offsets without changing the public `ReferenceExpansion` model. Preserve the helper's resumed executed text, invocation source, and offsets rather than resetting them.

This must change the actual receiver as well as its caller. `_ReferenceRead.expand()` currently passes the whole text into `expand_references()`, which reparses all candidates; `self.candidates` only affects no-root refusal records. Filtering only the admission tuple is insufficient. Looping over the captured candidate spans, expanding each selected raw token under the same captured authority, and stitching the untouched original slices is a small compatible way to enforce the selection and calculate offsets from those exact returned bytes. Construct the resolver and Git runner once per expansion under the existing activation. Keep before/after source checks and original worker-return/finish behavior.

Deleting only the early call loses selector/literal/continuation semantics. Moving the entire helper below admission leaves duplicate reads and a separate untracked worker. Adding a new owner or receipt is unnecessary: ordinary admission already captures `_ReferenceRead` and reserves its footprint; `prepare_reference_expansion()` reuses that request, verifies exact text/candidates, and retains the original worker and operation through host assignment and `finally` completion.

## Semantic risks to verify in the resulting delta

- Use the same finite candidates at admission and expansion, with no full-text reparse that reinstates suppressed tokens. Automatic/machine/resumed sends and an empty selected tuple must not start a reference read.
- Preserve literal command and leading plugin-skill arguments, and exclude only the leading manual-rule selector while expanding its eligible remainder. Keep native selector refusal semantics.
- Derive embedded skill invocations from the raw invocation source, never newly inserted reference bodies. Offsets must describe the same expansion supplied to the provider; failure/raw fallback must clear offsets. The existing skill substitution code translates raw coordinates using these offsets.
- On actual pause/resume, reuse the expanded draft and offsets without re-reading, while preserving existing fresh plugin admission and disabled-plugin refusal. User echo remains raw; reference status/refusal records remain truthful.
- Keep the captured authority, original ordinary owner, source checks, worker-return drain, and cancellation cleanup. If classification dependencies move, inspect the actual source-identity receiver records; do not weaken guards or substitute a new ownership path to accommodate a fixture.
- Preserve no-root refusal, Git handling, and non-native fallback behavior. Do not infer stock Git execution from the file-read finding.

Relevant existing plugin regression cases cover embedded skills after a reference, reference bodies that look like skill invocations, real preparation resume including plugin disablement, manual-rule remainders on initial/resumed sends, and literal plugin arguments. Some resolver monkeypatches use a no-argument lambda; the original authoritative receiver passes `authority=...`. Such fixtures may need to target the real compatible seam, but this does not explain away the independently observed pre-admission read.

## Coverage and required evidence

Read connected controller preparation and substitution callers, the extracted plugin helper, ordinary admission/reference capture, the original reference preparation receiver and read lifetime, reference parsing/resolver/Git callback contracts, and the relevant plugin tests. This was a narrow source investigation, not a repeat whole-diff review or executable verification.

The writer should supply bounded evidence for the actual conflicting-reference path (no read before admission; exactly one owned read), original return/cancellation custody, and the plugin literal/manual-rule/embedded-skill/real-resume cases above, followed by the precise final delta and receipts. The old interrupted/terminated ordinary cohort is not drainage or success evidence. The one bounded failure does not establish the status of every ordinary-tree case. Final source/receiver-identity review and all unresolved evidence holds from the initial report remain open.

The checkpoint Ruling 13 proposal was communicated during this narrow review; its implementation and read-only provenance semantics are outside this supplement and require their own scoped delta review before closing the original I1.
