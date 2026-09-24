# Lab — Models, speech, and evaluation runs

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Lab is the nav label for Models, speech, and evaluation runs. Opening it
lands on a screen titled **"Models"** ("Manage providers, models, and
endpoints.") — its default sub-area; the other sub-areas (speech,
evaluation runs) are reached from within that screen.

## Getting there

- Press **F2**, click **F2 Lab** in the nav bar, or press **Ctrl+P** →
  "Tab Navigation: Switch to Lab".
  There is no hotkey digit for Lab.

## Skill eval

Skill eval answers "how good is this skill?" — it scores a skill as the
subject under test (triggering, instruction fitness, simulated behavior)
without ever executing it. To run one, pick the **Evals** chip on the Lab
mode strip, then:

1. Press **"+ New skill eval"** in the library rail. This creates a draft
   skill-eval bench.
2. Pick the **subject**: either a skill from the store picker (each entry
   shows its trust tier) or, for a skill that lives outside the store, type
   its directory path in the input below the picker — whichever you set
   last wins. Then pick a **depth** (quick / standard / deep) and the
   **generator** and **judge** models. The panel shows the estimated LLM
   call count for the chosen depth before anything is spent (quick 0,
   standard 16 with a worst case of 32, deep 67 with a worst case of 84 —
   the maximum counts each judge cell's one retry).
3. Press **Run**. Progress ticks per completed call; **Stop run** (armed
   only while a run is in flight) keeps the partial results. Every pick —
   subject, depth, and both models — is saved as you make it, so
   navigating away and back loses nothing; **Escape** on the panel closes
   it. If no eval models exist yet, the panel says so and names the two
   ways to create one ("+ New target" in a bench editor, or "Create
   sample bench" once).

When the run finishes it appears as a group in the library rail; selecting
it opens the skill-eval report: the composite score with letter grade and
confidence label (Estimated / Assessed / Certified by depth), per-dimension
bars, layer statistics (judge rubrics and trigger F1; simulation activation,
consistency and failure rates with confidence intervals), any
anti-pattern findings with their remediation text, and — expanding the old
artifact count — one read-only line per captured judge/simulation cell.
**Run again** reruns the bench from its saved configuration, and when an
earlier run of the same bench exists, **Compare with previous** shows the
composite and per-dimension deltas between the two reports. A preflight check blocks
the run early if the chosen models aren't configured. Skill bodies longer
than 8,000 characters are truncated in the prompts and the report says so.

## Speech: the OmniVoice (ONNX) provider

The speech sub-area offers **OmniVoice (Local)** as a TTS provider: a
CPU-only ONNX int8hq engine for multilingual speech with zero-shot voice
cloning, independent of the audio.cpp runtime.

- **Install**: `pip install ".[omnivoice_tts]"` (onnxruntime + tokenizers),
  then fetch the model in the model browser — the consent step states the
  license stack: the LM weights are **CC-BY-NC** (non-commercial use; no
  impersonation) and the audio tokenizer is under the Boson Higgs Audio 2
  Community License. Or point `[OmniVoiceSettings] model_root` at a local
  copy of the `ct03/omnivoice-onnx-int8hq` tree.
- **Latency**: the engine runs at roughly 3–7× real time on desktop CPUs —
  a 10 s clip can take well over half a minute. It is a batch provider:
  expect progress reporting per diffusion step, not streaming audio.
- **Quality/speed dial**: **Diffusion steps** (default 32; lower is faster,
  slightly rougher) and **Guidance (CFG)** are per-request knobs in the
  speech playground.
- **Voice cloning**: cloning needs the reference clip *and its exact
  transcript*, so clone voices are managed as profiles in the Voice
  Cloning window (which asks for the transcript). A bare reference upload
  in the playground cannot clone.
