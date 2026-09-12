# Provider, catalog and Speech UI review

Reviewed application HEAD `1edcbb1aee9bacfd7573c53f70b22d51b27f6300` against
`a36fc6133c69f77b8b14596a59918de34261f8ef` on 2026-09-09. Scope: AllTalk and Higgs
backend changes, legacy voice catalogs, Speech Playground/configured Kokoro
engine/language choices, Speech settings defaults, and their targeted tests.
The accepted Kokoro runtime/lifecycle/language review was not repeated.

## Actionable finding

P2 — Update the stale fresh-AllTalk UI expectation in
`Tests/UI/test_speech_playground_pane_lifecycle.py:3787`. The
`test_legacy_provider_defaults_and_labels_are_preserved` AllTalk parameter still
expects `female_01.wav` / `Female 01`. The actual mounted selector now correctly
shows `alloy`. Its exact targeted control fails at line 3814: `assert 'alloy' ==
'female_01.wav'` (1 failed, 5 deselected, 2.83s, exit 1). Update this fresh-default
expectation to `alloy` / `Alloy`; retain historical saved-value migration
semantics. Root independently reproduced and accepted this correction in the
broader targeted run.

## No additional product findings

- AllTalk's six aliases and opaque explicit custom IDs agree with its selected
  OpenAI endpoint boundary. Native `/api/voices` speaker names are not OpenAI
  request voice IDs. `default` resolves the captured backend configuration;
  network authority and custom ID forwarding remain exact. An endpoint that
  rejects an explicit custom ID still produces an error rather than silently
  choosing a voice.
- The initialization request uses the actual awaited httpx API. Connection and
  non-200 statuses are logged without server body/voice/path leakage. The HTTP
  log suppression context unwinds on cancellation. Initialization remains
  advisory discovery; it does not establish synthesis health or guarantee that
  the configured server accepts custom IDs.
- Higgs forwards configured torch dtype through the pinned V2 `torch_dtype`
  keyword and retains the older `dtype` constructor signature. The pinned V2
  constructor is keyword-compatible with the production argument selection.
- Historical AllTalk migration retains `female_01.wav` as the old-default
  sentinel. The changed fresh settings default does not rewrite that sentinel
  or the migration implementation. Existing saved custom values continue to
  migrate as exact values. Kokoro's fresh Lab switch inherits its configured
  engine; explicit switch changes and non-English voice/language requests are
  covered through the mounted production pane and fake service.

## Evidence and limits

The inspected local source files exactly match AllTalk
`f16117e95b540e9bbbd8247b49ca6c6b1350b172` and Higgs
`05a145bb490501b534563bf51bf2f7aa2326b271`. The pinned files were read, not imported
as servers or engines. Full source hashes, exact pins and the stale-test receipt
are in `provider-code-review-provenance.json` beside this report.

One bounded fake selection completed with **36 passed, 1 inherited pydub
audioop deprecation warning in 9.51s**, exit 0. The selection includes new provider
contract tests, new mounted Kokoro defaults/language UI tests, existing AllTalk
endpoint authority/privacy tests, catalog compatibility, and saved-value /
idempotent migration tests. Log: `provider-review-targeted-tests.log`.

No model inference, ASR, audio playback, real server, downloads, package changes,
or repository edits were performed. No further duplicate rerun was started
after root requested that. The reported green selection does not replace root's
separate live provider/content or broader integration evidence.
