# f095 UI16 first-send import repair

Starting head f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80. Approved scope: restore required warmup/import contracts and repay real module cost; no ratchet raises or source-authority shortcuts. No commits/push/Qodo by this agent.

## Causality and patch

The hosted lean dependency environment loads six during html5lib's step. Local optional imports had already loaded six, hiding that cost. Add six as its own step before html5lib; existing third-party package oracle is unchanged. The prior removal of app_served based on actual native boot residency was incomplete: ConsoleHarness and automatic Library send do not share that mounted boot state. Restore app_served in the warmup list. Both original exact no-extra-UI-import assertions remain unchanged, including the deliberate two-module Library exception.

Repay the added warmup cost by deferring Web_Scraping.deep_search_citations from web_tool_impls to actual deep-search footer formatting. The existing global name deep_search_citations_footer remains a callable with the same parameter/default/return semantics; it forwards verification unchanged to summarize_for_footer with no catches or empty fast path. No existing web tool function/class body changed (independent AST comparison recorded), including permission/SSRF/pipeline/cancellation/truncation/citation processing. New annotation uses the equivalent modern Mapping|None spelling to avoid new lint. Remove only the now-cold explicit warmup/historical-live-inventory citation entry; retain its first-use behavior and tests.

## Targeted evidence

Two new local regressions fail before product edit: missing six step and citation module resident on tool import (red.log). After edit,12 exact focused controls pass in25.89s (green.log/xml): all three original hosted failures; dependency order and cold-first-use positive; catalog import-list control; existing deep-search happy path and real footer formatting; current tool-global monkeypatch seam; silent/unknown/misquoted/uncited footer semantics. There are no network calls in the footer controls; pipeline boundaries use existing fixtures.

lean-html-order.json independently reproduces the lean dependency issue in fresh python -S interpreters with only the existing venv site path added. No app imports or sys.modules removal/seeding: previous sequence loads html5lib+six in one step, current actual inventory sequence loads webencodings,six,html5lib as separate package roots. This is a focused dependency proof, not a claim of a complete recreated hosted environment. The first private probe incorrectly counted new six submodules as new package roots; corrected to the same root-set difference used by the production guard before final evidence.

Scoped default lint30->30, zero added (lint-source.json). All original web tool function/class ASTs identical (unchanged-web-tool-bodies.json). New delegate range-format clean and git diff --check passes. Annotation-only cleanup followed the12-case run; final guards will exercise the settled source.

## Owned repository paths

- tldw_chatbook/Chat/console_first_send_modules.py
- tldw_chatbook/Tools/web_tool_impls.py
- Tests/UI/test_console_first_send_preimport.py
- Tests/Tools/test_web_deep_search.py

## Final coordinated performance qualification

After core confirmed no active app tests and frozen source, all five original performance guards passed in31.22s. Driver is unchanged observation-only /private/tmp/mixins-perf-probe.py, label f095-firstsend; log and four raw census returns are enumerated in artifact-files.json.

- Actual native UI-ready1031/1031 modules (zero headroom).
- Screen preimport554/558 modules,426096/426199 LOC (103LOC headroom); largest Library132405/135111LOC.
- First-send warmup93/94 modules,57485diagnosticLOC (one module headroom).
- CSS original parsed rule-count guard passes<=268 and original exact boot-worker inventory passes. No fresh exact CSS count claimed because pytest captures successful print output.

No ratchet constant, source snapshot, first-send import assertion or permission gate changed. All owned runtime processes stopped. Final source hashes and unchanged web-body proof are recorded; no further edits planned. Independent source review is parent-assigned and separately recorded.
