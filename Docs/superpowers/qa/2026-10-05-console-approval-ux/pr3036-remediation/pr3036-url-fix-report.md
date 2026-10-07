# PR3036 URL display redaction remediation

Authorized checkout: `C:/Users/GDesktop-1/.codex/worktrees/console-approval-pr/tldw_tui`.
Branch: `codex/console-approval-ux`. HEAD observed at report creation: `93f639956cc3b66e0bb5e7ea31259a2eeeedca55`.
Review base: `93f639956cc3b66e0bb5e7ea31259a2eeeedca55`.

The concrete review finding is fixed at the shared scalar redaction boundary. `_redact_value` preserves whole credential-shape masking first, then delegates complete scheme-bearing/scheme-relative string values to the existing `redact_url`. Nested mappings, lists and tuples already use that boundary. `redact_url` preserves the exact input spelling when no credential is redacted and masks a malformed URL wholly on a parser `ValueError`.

Only redaction.py and three existing test files changed. No dependency, provider authority, matching input, scope, grant, owner, persistence, launcher, card, controller, CSS, root-identity, or metadata change was made by this agent. Existing shared redaction consumers receive the same credential masking; enforcement/matching callers do not receive rewritten arguments. Captured original mappings remain separately copied by the existing capture owner.

ADR required: no new ADR.
ADR path: backlog/decisions/221-console-approval-interaction-and-feedback.md.
Reason: direct repair of existing complete redacted display/original input contracts; no storage or authority boundary change.

## RED evidence

All commands ran from the checkout above, with the existing canonical private launcher protecting the original home before redirecting config/home/data/temp. The launcher printed origin verification and `startup_allowed` each time. Commands (PowerShell, exact pipeline form):

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/MCP/test_redaction.py' -k 'recursively_redacts_url or preserves_nonsecret_urls or malformed_url' 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-red-unit.log'; exit $LASTEXITCODE
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/Chat/test_approval_presentation.py' -k 'captured_url_targets' 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-red-capture.log'; exit $LASTEXITCODE
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/UI/test_approval_details.py' -k 'complete_details_pages_redact' 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-red-details.log'; exit $LASTEXITCODE
```

- Unit: 10 failed, 1 passed, 12 deselected, 1 warning (1.33s pytest duration). Four nested URI fixtures leaked the synthetic query credential, four harmless strings were normalized unnecessarily by direct redact_url, and two malformed fixtures raised ValueError instead of returning safe display content.
- Capture: 1 failed, 18 deselected, 1 warning (1.26s), at the real captured long target comparison showing `api_key=synthetic-0` surviving.
- Details: 1 failed, 19 deselected, 1 warning (4.05s), at the reconstructed all-pages URL secret absence assertion. The fixture uses actual capture and includes nested URI values after a large body, so the complete continuation is checked.

## Final GREEN evidence

The four targeted files passed on the final formatted sources: 141 tests total, each process with its own private profile. Exact commands:

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/MCP/test_redaction.py' 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-final-unit.log'; exit $LASTEXITCODE
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/MCP/test_redaction_value_shapes.py' 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-final-shapes.log'; exit $LASTEXITCODE
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/Chat/test_approval_presentation.py' 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-final-capture.log'; exit $LASTEXITCODE
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' 'Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py' 'Tests/UI/test_approval_details.py' 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-final-details.log'; exit $LASTEXITCODE
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -m ruff check --no-cache tldw_chatbook/MCP/redaction.py Tests/MCP/test_redaction.py Tests/Chat/test_approval_presentation.py Tests/UI/test_approval_details.py 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-final-lint.log'; exit $LASTEXITCODE
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' -m ruff format --no-cache --check tldw_chatbook/MCP/redaction.py Tests/MCP/test_redaction.py Tests/Chat/test_approval_presentation.py Tests/UI/test_approval_details.py 2>&1 | Tee-Object -FilePath '.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr3036-url-final-format.log'; exit $LASTEXITCODE
git -c safe.directory=C:/Users/GDesktop-1/.codex/worktrees/console-approval-pr/tldw_tui diff --check -- tldw_chatbook/MCP/redaction.py Tests/MCP/test_redaction.py Tests/Chat/test_approval_presentation.py Tests/UI/test_approval_details.py
```

- Unit: 23 passed, 1 warning (4.49s).
- Existing credential shapes: 79 passed, 1 warning (1.48s). Existing whole secret-key, URI-userinfo, credential-shape, bytes, CLI, and innocent-value behavior is retained.
- Capture: 19 passed, 1 warning (4.26s).
- Details: 20 passed, 1 warning (13.31s). Full reconstructed targets/arguments are redacted and nested originals retain the synthetic source values.
- Ruff check: exit 0, All checks passed. Formatter: exit 0, four files already formatted. Scoped Git whitespace check: exit 0.

Earlier GREEN logs before format remain as `pr3036-url-green-*.log`. An initial read-only format check without `--no-cache` could not initialize the sandbox-external `.ruff_cache`; the no-cache check ran correctly and reported three formatting changes. Formatting corrected new blank lines/new expression wrapping plus existing expression wrapping in the owned helper. Final lint/format and tests passed. No guard was weakened to get those results.

## Source hashes (SHA256)

- `tldw_chatbook/MCP/redaction.py`: `1fa57eef60261317ab322502c932e521d4955ff3b4458df88f9fb30611b08b82`
- `Tests/MCP/test_redaction.py`: `659e617a7d9c646fd449fc6a03ffdbbfc0fce3c051be532b68cf93624ec47f5f`
- `Tests/Chat/test_approval_presentation.py`: `38ae23ea1a51f994ff00b7b505df327a8f41c7eb2b7954c9dd2035ec47cc5c1d`
- `Tests/UI/test_approval_details.py`: `9e205ec9634701336659258abbdf356340a2026b2ba067e62edb0e7352f60d0e`
- `Tests/MCP/test_redaction_value_shapes.py`: `57c04cb839388f8bacd98ad0ff7bef32a64b90debc2556b99d3acc0bd09e1f22`
- `Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py`: `f5262bca70fb62a8db59467e938d7080c4bb90b07b5581a61087ada33b83ba3c`

## Evidence log hashes (SHA256)

- `pr3036-url-final-capture.log`: `e03ce6e018c009708b9c56e6158632d98b3f555966784c43759828c0b8caa11b`
- `pr3036-url-final-details.log`: `6c86f4770b3301a12ba246a1ef00b52fd52679626494d7ac6f48f4a5fe6e9a76`
- `pr3036-url-final-format.log`: `55b5a2cb0bdd55641f1b6dad60805a4c29520e41d507781e75aa8ec8d6ed9a10`
- `pr3036-url-final-lint.log`: `a4443afdcfb6d7363adb285762515ccf7cf50473b1a05c20c1a50f6bed4d26b0`
- `pr3036-url-final-shapes.log`: `165611fe23001743813c97fbf49056fa6cfd504a37a8b85d1e2c75092cbd9f18`
- `pr3036-url-final-unit.log`: `a6f8ebbb59b7874054a31276b849ffb0e119a4a28120700b30cd169952aa355a`
- `pr3036-url-green-capture.log`: `e41da274cf4c7958f488a4b6ce73b9ecfb955b19a416f615430acc7b74890cc9`
- `pr3036-url-green-details.log`: `13fd510367924196c0d2eb89a1b181816d29da4d1207b7b9d68dbd183038b08f`
- `pr3036-url-green-shapes.log`: `dc5230daf26a4f7611cfc6dcfdf126a7573089fe37bbd3851c458b5fe66b6557`
- `pr3036-url-green-unit.log`: `c73f9b1d65119af27091995f41551d8c55f77257147c31d687befec30258b98b`
- `pr3036-url-red-capture.log`: `e468a772aa7af013bae0ac36b1f8cb6618e4f645179068aab725f08f4284cb8d`
- `pr3036-url-red-details.log`: `4633f30924c125186fe1ddcd6279f32974dcc28b62cc5ebbcb5bbe8464b2a82c`
- `pr3036-url-red-unit.log`: `e6048cc6d760f5edeb9616e902baa55d4adf25306b8bb774e6b8f84a756840ce`

## Limits and self-review

Self-reviewed the complete owned diff and the scalar/mapping/sequence/CLI and direct-URL caller paths. The change has no I/O and adds no imports/dependencies. Scheme recognition is anchored to complete URI values, so ordinary prose is not sent through URI normalization. Non-secret URLs retain their exact existing percent escaping, separators, blank flags and fragment; malformed authorities are wholly masked because safely identifying their credential boundaries is unreliable.

Existing pytest-asyncio unset-loop-scope and Pydantic Field deprecation warnings are preserved in the raw receipts. Startup optional audio dependency messages appeared in failing fixture setup output. No warning was hidden or repaired in unrelated files.

No full test suite was run, no real application/profile was launched, and no native/browser paint, latency, Windows dispatch/root-pin or full-task qualification claim is made. Root owns CI/integration/review and task metadata. This agent did not stage, commit or publish.
