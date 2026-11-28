# Hosted macOS first-send comparison

Run TASK-34406.31's frozen baseline and candidate in one GitHub-hosted macOS
job. The exact source commits are ancestors of the workflow commit. Use the
original unprofiled B/C/C/B/B/C/C/B order with eight fresh private profiles.

ADR required: no.
ADR path: existing 230 (Console Send preparation and I/O ownership), 097 (boot
budget ratchets), and 126 (local storage ownership and retirement).
Reason: this is a test-fixture transport change. Product trees, storage,
permissions, and preparation contracts are frozen.

1. Stage the original reviewed driver, launcher, source maps, observer, and
   qualification checks. Bind portable paths and seal the hosted helper set.
2. Preserve complete checked and settled config equality using the reviewed
   maps. Carry only the expected header, schema, and settings fields from
   previous runs.
3. Use the unchanged DeepSeek adapter and redirect only its expected HTTPS
   POST to a real loopback HTTP server with fixed SSE fixtures. The request
   still goes through the original observer and per-app budget. The fixture
   requires the expected prompt, model, stream, auth marker and schema count.
   No live provider credentials are read or supplied.
4. Require the original continuous quiet window and host overlap receipt,
   all eight original per-run audits and a closed stub with eight HTTP 200
   responses before paired statistics. Keep partial artifacts on failure.
5. Report Enter-to-observed-POST and post-save timings for this runner. Keep
   the historical live-provider cohorts and source refs intact.
