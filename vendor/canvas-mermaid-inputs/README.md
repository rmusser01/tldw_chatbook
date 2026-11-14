# Pinned Canvas Mermaid build inputs

This directory contains the eight original source inputs declared by
`tldw_chatbook/Canvas/mermaid/inputs.json`. Normal CI and local preflight
read them offline and verify the original sizes, SHA-256 digests, Mermaid
archive inventory and selected source hashes before rebuilding. These
19,636,664 bytes are build inputs, not installed application package data.

The Mermaid tarball includes its MIT license; `unicode-license.txt` is
the original pinned Unicode license. Source URLs and provenance are in
the existing input manifest. No npm installation or upstream code runs.

Verify with the manifest-pinned Python 3.12.11:

```sh
python scripts/check_canvas_mermaid_assets.py
```

To reacquire the same pinned inputs, explicitly download into a **new**
directory and then verify that directory:

```sh
python scripts/check_canvas_mermaid_assets.py --fetch-inputs /tmp/canvas-input-refresh
python scripts/check_canvas_mermaid_assets.py --input-dir /tmp/canvas-input-refresh
```

The fetch command refuses an existing destination and validates every
download against the unchanged manifest. It does not update pins, assets,
profiles or admission policy. Review the refreshed bytes before replacing
this directory. Changing semantic inputs requires a new immutable runtime
profile and ADR-124 qualification; do not repin an existing profile.

An explicit `--input-dir` or `TLDW_CANVAS_MERMAID_INPUT_DIR` can use another
verified cache. Missing files or wrong hashes fail; checks never fall back
to the network or skip reproduction. See ADR-238 for the build boundary.
