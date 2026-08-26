# ComfyUI MiniMax H3 UAT

- Date: 2026-08-09. Origin class: configured trusted host; no credentials recorded.
- Required class names and availability: Base — `BasicGuider`, `BasicScheduler`, `CLIPLoader`, `ComfyMathExpression`, `CreateVideo`, `KSamplerSelect`, `MiniMaxH3ImageToVideo`, `PrimitiveFloat`, `RandomNoise`, `SamplerCustomAdvanced`, `SaveVideo`, `UNETLoader`, `VAEDecode`, `VAEDecodeAudio`, and `VAELoader` available. Spectrum — all Base classes plus `SpectrumApplyMiniMaxH3` available. `SaveVideo` advertised MP4; MiniMax H3 width and height advertised step 32.
- Packaged workflow filename: `minimax_h3_t2v.json`. Submission accepted; prompt-id suffix `f026f076`.
- Base history output: success; node id `92`; collection key `images`; descriptor field names `filename`, `subfolder`, and `type`.
- Base HTTP result: content type `video/mp4`; byte length 1,137,440.
- Base ffprobe: format `mov,mp4,m4a,3gp,3g2,mj2`; duration 5.167000 seconds; 864×480; frame rate `24/1`; stream types `video` and `audio`.
- Packaged workflow filename: `minimax_h3_t2v_spectrum.json`. Submission accepted; prompt-id suffix `008bacea`.
- Spectrum history output: success; node id `92`; collection key `images`; descriptor field names `filename`, `subfolder`, and `type`.
- Spectrum HTTP result: content type `video/mp4`; byte length 939,055.
- Spectrum ffprobe: format `mov,mp4,m4a,3gp,3g2,mj2`; duration 5.167000 seconds; 864×480; frame rate `24/1`; stream types `video` and `audio`.
- No media, source path, raw export, or prompt text was committed.
