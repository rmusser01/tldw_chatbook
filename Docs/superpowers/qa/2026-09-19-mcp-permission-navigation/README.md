# Tool permission navigation — TASK-32867

The two “Change in Permissions” controls retain the tool/profile shown by their
actual mounted button. Refresh and close invalidate controls before asynchronous
removal. Retired, hidden, disabled and covered-screen controls cannot navigate;
live controls remain retryable. Existing routing, permission policy and styling
are unchanged. Audit navigation remains in separate draft PR2724 / TASK-32837.

Implementation follows ADR-150 and ADR-161; no new ADR is required. Branch
`codex/mcp-navigation-action-review` starts at merged dev `6095db2f5f` (PR2734).
PR2731 and PR2734 are merged; this follow-up needs its own final visual approval.

## Verification

- **75 targeted tests pass:** 23 new navigation regressions plus 37 permission
  write/revoke regressions ([60-case run](green-003.txt)); 15 existing routing,
  blocked-preview and panel cases ([routing run](routing-001.txt)).
  [Case results](test-results.json). No full suite ran.
- Initial tests reproduced 13 navigation failures; two additional initial failures
  were an invalid attempt to mutate a frozen fixture. Independent review then
  found unavailable-owner handling, reproduced by six additional failing cases.
- [Independent re-review](independent-review.txt) has no remaining blockers.
  [All seven preflight guards](preflight.txt) pass. New test and runner pass Ruff;
  the production file retains its 13 existing diagnostics with no additions
  ([comparison](static-analysis.json)). Edited production lines and new files
  satisfy Ruff formatting ([receipt](formatting.json)).
- Existing pytest cleanup warnings concern old unrelated temporary directories.

## Native and visual evidence

[Gallery](GALLERY.md): twelve inspected captures cover both source buttons and
the selected Permissions destination, dark/light at 120×40 and 170×48. Eight
keyboard journeys use the real inspector/workbench/service with a disposable
saved discovery fixture; both controls select `local:review-docs::search` and
leave its permission record unchanged. Full button labels, keyboard focus,
compositor bounds and center hit ownership are checked before Enter.

Exported SVGs normalize whitespace-only lines; their rendered content is unchanged.

The fixture is disconnected and its tool is Off. Test Tool displays the real
permission refusal; no server connects or tool executes. The native runner blocks
outbound sockets. [Native receipt](native/result.json) records all eight routes;
[lifecycle](lifecycle.json) checks exit 0, absent process, released instance lock,
ten healthy private databases, zero conversations/messages/execution records,
unchanged default-profile files, and nine matching source hashes.

Two preliminary native attempts failed in the harness: “off” was used instead of
the store's canonical “deny”, then Enter preceded table focus settling. The final
runner uses deny, reloads the fixture, and settles focus before input. An earlier
green attempt also caught an invalid `is_displayed` property; final code uses the
supported Textual display/visibility APIs. Final tests and native replay use the
same production source hash.

The captures retain existing built-in server guidance and compact scrolling;
those broader issues have separate drafts (including PR2722 and PR2718). This
repair does not claim to complete the MCP or wider design-system review.

## Reproduce the native check

Prepare an unused profile beneath canonical `/tmp` with `home/`, `config/`,
`data/` and a contained config/database layout. In a native tmux TTY run:

```sh
.venv/bin/python Docs/superpowers/qa/2026-09-19-mcp-permission-navigation/native_check.py ROOT TMUX_SOCKET SESSION
```

The shared argument parser validates private paths and output ownership before
startup. The runner's docstring describes outputs and exit codes.
