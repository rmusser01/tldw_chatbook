# PR3346 legacy export fix (AC19)

Owned scope: `tldw_chatbook/DB/prompt_export.py` and new `Tests/Prompts_DB/test_prompt_export_mixins.py`. Parent owns Task38544, docs/manifests/pins/publication. No commit, push or GitHub mutation by this implementer.

The public legacy exporter now includes artifact type in its internal filtered read and explicitly refuses any selected Mix-in before keyword fetch, tempfile/directory creation, file opening or ZIP writing. Its fixed feedback contains no names/bodies: “Mix-ins are unsupported; use Library or Chatbook export.” It returns the existing `"None"` output sentinel, without silently filtering records or producing partial output.

The duplicated keyword-filter branch was simplified while preserving normalization, joined active Prompt/active keyword predicates, unfiltered fallback for blank filters and the live defining dependency seam. Ordinary Prompt/Recipe columns/templates, public signatures, options, errors, typed Library exports and Chatbook codecs are unchanged. The helper is241 lines under the unchanged242 ceiling. Existing ADR286decision2 governs; no new ADR required.

## Verification

`commands.json` contains exact commands, exit codes and results. `fix.patch` contains both owned files. `source-proof.json` pins the frozen bytes.

- Initial RED before product edit:8 failed /3 passed. Refusal cases observed forbidden keyword work; ordinary compatibility cases already passed.
- Final frozen current export command:27 passed,62 deselected (7.04s). Sixteen refusal cases cover CSV/Markdown, unfiltered/shared/mixin-only/blank filters and associated-keyword on/off. All use a real SQLite database; only forbidden downstream side-effect seams are guarded. Three exact-content ordinary Prompt/Recipe controls cover CSV and Basic/Detailed Markdown with a Mix-in in the same database. Existing export tests retain empty-result, minimal-column, template, collision and Unicode coverage.
- Final regressions with only the exact b110 baseline helper substituted in memory:16 failed /3 passed (1.52s). Without keyword collection the old helper reaches forbidden output allocation; with it the old helper reaches forbidden keyword access. `baseline-probe.py` preserves this bounded comparison without reverting any shared checkout file.
- Typed Markdown/Chatbook plus the two exact prompt_export size nodes:116 passed (8.24s).
- Ruff baseline3/current3 identical `(file, code, message)` multiplicities; no introduced diagnostics and no new-test diagnostics. Existing helper debt is F841/DTZ005 at the unchanged unused timestamp, and BLE001 at the preserved broad error boundary. `lint-delta.json` records the comparison; the whole helper is not claimed lint-clean.
- Both files pass formatting. Tracked product whitespace returns0; new-file no-index whitespace returns1 (expected diff-present status) with no whitespace diagnostics. The first evidence-recording script incorrectly expected0 for both; its bookkeeping assertion was corrected without changing source or tests.

All databases/exports used repository private pytest/bootstrap fixtures; successful temporary export paths are removed by tests. Disposable pytest fixture directories under this evidence root are excluded from the top-level delivery inventory. No user profile, mounted app, external provider or full test sweep was used. Source/test were frozen and the integration reviewer notified before final private evidence writing.

## Review state

The confirmed core finding is implemented and covered. Root/integration reviewer decides closure after independent scoped re-review. Only the two owned source/test paths were changed by this implementer; the Task38544 status change belongs to the parent. No further tests or source changes planned.

Not checked: full suite, mounted UI/performance, real cloud cache behavior or cross-OS execution.
Risk: selected Mix-ins deliberately refuse this legacy export; use the existing typed Library/Chatbook export.
