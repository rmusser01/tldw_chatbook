# Kokoro language frontend repair

Task: TASK-32149

ADR required: yes
ADR path: [ADR-142](../../../backlog/decisions/142-kokoro-east-asian-phonemization-across-engines.md)
Reason: Select optional model-aligned Japanese/Mandarin phonemizers across the
existing engines while keeping dependency and runtime ownership explicit.

1. Preserve the completed fourteen-tuple baseline and exact French/Mandarin
   setup failures, Japanese ONNX character descriptions, and missing UniDic data.
2. Add targeted failing language-helper, actual backend-routing and cancellation
   regressions. Normalize French and prepare Japanese/Mandarin phonemes lazily
   inside the retained ONNX worker; preserve explicit phoneme calls.
3. Report missing Japanese dictionary data with actionable fixed guidance in
   both engines. Provision only the dictionary in the isolated test environment.
4. Rerun the affected languages with real playback and complete ASR. Retain
   spelling/script variants and unresolved pronunciation differences separately.
5. Run affected lifecycle, language and UI checks; record source and runtime
   identities, review the change, and link ADR-142 in task implementation notes.
6. Close review findings with failure-first regressions: classify imports made
   lazily by frontend construction/use as missing language dependencies, and
   prevent PyTorch model entry after Stop during pipeline phonemization via the
   official per-call model argument. Preserve shared pipeline/model identity,
   upstream chunking, retained ownership and successor generation.
