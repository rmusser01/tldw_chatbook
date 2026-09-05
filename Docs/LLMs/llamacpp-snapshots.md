# llama.cpp prompt-cache snapshots

The Models → llama.cpp manager manually saves and restores processed model
context. It does **not** save, reopen, or restore Chatbook conversations. Normal
chat requests are unchanged, and successful Restore does not guarantee that a
later request will reuse the cache.

**Live Models UAT passed with b10816 and Gemma 4 plus its BF16 vision projector.**
The confirmed-Restore/readiness race is fixed, and the normal UI demonstrated text
and same-image reuse, default/configurable retention and confirmed Delete with
no diagnostic overrides. This is evidence for that exact configuration, not a
guarantee for other models or releases. See the
[UAT record](../superpowers/reviews/2026-09-05-llamacpp-slot-snapshots-uat.md).

## Before launching

Enable snapshots in canonical **F9 Settings → LLM and Providers**, using the
category Save/Revert workflow, or in Models → **Details & preferences** using
Apply. Enable/disable changes apply to the **next launch**. Keep-count changes
apply to later Save operations on an already-running enabled launch.

Launch a local llama-server through Chatbook. The first supported management
envelope is plain HTTP on numeric IPv4/IPv6 loopback, with normal endpoint paths.
Environment proxies and redirects are disabled for management requests. Remote,
TLS, prefixed, router, or unknown-option launches can retain ordinary launcher
support while snapshot controls explain why management is unavailable. Do not
supply `--slots`, `--no-slots`, or `--slot-save-path`: Chatbook owns these options.

The runtime must support slot persistence and expose complete compatibility
evidence. Runtime, model, projector, effective settings, and observed slot context
must match. Automatic settings whose effective values cannot be observed disable
Save/Restore. Explicit CPU settings such as `--flash-attn off --fit off --device
none --n-gpu-layers 0 --parallel 1 --ctx-size 8192` avoid those automatic choices.
For vision, provide the matching `--mmproj` and explicit `--mmproj-device none`
with `--no-mmproj-offload`. These are examples, not universal model recommendations.
SWA models require `--swa-full` for prefix reuse; Chatbook does not add it
automatically. Unknown or incompatible settings remain disabled rather than guessed.

Private storage currently requires POSIX ownership and permission semantics.
Windows ACL equivalence is not claimed. An unavailable snapshot store does not
prevent ordinary snapshot-disabled launches.

## Using the manager

Process a request, select its idle slot, then choose **Save**. Names are generated
from timestamps; filenames are not user-entered. Select a saved row and destination
slot for **Restore**, then confirm replacement of processed context. A failed
Restore may clear destination cache, but does not delete the saved source or
change conversations. **Delete** permanently removes a selected saved snapshot
after confirmation; there is no automatic backup before Restore.

The default is the **newest 10 across all models in this profile**, not 10 per
model. The keep count accepts integers from 1 through 1000. Retention prunes only
after a new Save commits successfully. It is a count limit, **not a byte quota**;
large binaries and temporary working copies can require substantial extra space.
Cleanup failures and residual bytes are reported separately from Save success.
Details shows the profile storage location only when expanded, as read-only text.

Refresh observes current slots; observations are not a reservation or continuous
monitor. Readiness probes have a five-second aggregate bound. Save/Restore show
preparation and elapsed time and allow up to ten minutes after submission.
Navigation away does not cancel an app-owned operation.

An **unknown outcome** means the server may still be writing or restoring.
Do not retry automatically: use **Stop Server** to settle that generation before
starting again. Working files stay retained until server and local file work
are settled. Catalog browsing and confirmed deletion remain available. Corrupt
or missing saved files are rejected before a Restore request is sent.

If persisted snapshot preferences are malformed, Models and F9 remain usable but
affected mutation/preference controls are disabled. Correct
`[llamacpp_snapshots] enabled` (boolean) and `keep_count` (integer 1–1000) in
Advanced Config, then Models Reload or F9 Revert. No silent default is accepted.
F9 detects changes saved by another surface and asks you to reload rather than
overwriting a stale draft.

## Opt-in local verification

No model, projector, image, or executable is downloaded by the harness. Supply
existing files you are permitted to use. Use the reviewed llama.cpp revision
`427291b5b34cd914a31b3fd3b61a68f6184f4b9f`, a compatible vision GGUF/projector,
and two byte-distinct PNG/JPEG/WebP images. Assets must fit an 8192-token CPU
context; choose a small model suitable for the available memory and time.

```bash
export TLDW_LLAMA_SNAPSHOT_SERVER="/path/to/llama-server"
export TLDW_LLAMA_SNAPSHOT_MODEL="/path/to/vision-model.gguf"
export TLDW_LLAMA_SNAPSHOT_MMPROJ="/path/to/mmproj.gguf"
export TLDW_LLAMA_SNAPSHOT_IMAGE_A="/path/to/image-a.png"
export TLDW_LLAMA_SNAPSHOT_IMAGE_B="/path/to/image-b.png"
TLDW_LLAMA_SNAPSHOT_LIVE=1 python -m pytest Tests/LLM_Management/test_snapshot_live.py -q
```

Without the exact opt-in, the live case skips before creating processes or
sockets. Opting in with missing inputs fails; missing counters fail, never become
zero. Run from the repository through pytest so root conftest isolates config
and data and installs the loopback-only network guard. No paid/optional/slow
marker is required. The per-test 4800-second budget accommodates local CPU
startup and the production ten-minute mutation limits. Every owned process is
stopped and reaped in `finally`; binaries remain in temporary test storage.

The harness drives the real Models Start/Save/Stop/Restore controls, using the
production app-owned service/store/client. It sends fixed benign ordinary
OpenAI-compatible requests without `id_slot`, not modified Chatbook chat routing.
One slot, explicit settings, and `--cache-ram 0` isolate slot persistence from the
server's separate RAM prompt cache. It verifies text reuse and compares image
controls: cold A, native in-memory A→B, separately restored A→A, and separately
restored A→B. Prompt totals must match across matching requests; restored A→A
must exceed both cold A and native A→B cache counts, while restored A→B must not
exceed native A→B. Distinct image-byte SHA-256 identities are required.

This boundary depends on the pinned runtime contract: media IDs hash incoming
bytes and common-prefix comparison stops before the first differing whole media
chunk. It does not guess image token counts from a text-only template. A changed
runtime contract or missing evidence fails the gate. The harness reports only
`timings.cache_n` and `cache_n + prompt_n` for named controls; use pytest `-rP`
to show successful captured counters, or `--junitxml` to retain the sanitized
`snapshot_live_counters` property. Fixture-counter tests are not real-server proof.
No audio reuse is claimed.
