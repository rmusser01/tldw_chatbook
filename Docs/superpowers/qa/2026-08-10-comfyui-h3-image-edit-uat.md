# ComfyUI H3 image-edit UAT

Date: 2026-08-11

## Result

Pass. The configured trusted origin was a private, non-loopback origin. Its host,
credentials, request identifiers, server descriptors, response bodies, and the edit
instruction are intentionally omitted.

- The real adapter completed the required class and schema preflight before upload.
- The sanitized packaged workflow returned exactly one PNG through canonical node 165.
- A synthetic 512×512 source produced one validated 512×512 result.
- Effective metadata contained the resolved seed plus the steps, sampler, workflow key,
  dimensions, operation, and format keys.
- The normal Image Generation persistence boundary durably stored and rehydrated the
  exact message, single PNG attachment, and generation metadata.
- The generation path did not import Video Generation, and persistence did not call the
  video store.
- Synthetic source bytes were not persisted. Temporary configuration, database,
  downloaded output, and harness files were removed after verification.

ComfyUI-side uploaded inputs and saved outputs remain subject to the server operator's
retention and cleanup policy; the application does not claim portable server cleanup.
