# First-run wizard: OmniVoice Voice option — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OmniVoice as a fourth service in the first-run wizard's Voice step — install its model in-wizard, test it locally with the same seed that gets saved, and save it as the default TTS provider.

**Architecture:** Pure install plumbing next to the OmniVoice catalog (`omnivoice_setup_state`, preflight/provision wrappers); pure OmniVoice helpers in the Voice step's state module (seed choice, save event, sample through the app's shared TTS service); a per-request `seed` in the engine and request admission; and an OmniVoice panel in `VoiceSetupStep` reusing the Speech step's install worker shape and the shared `ModelInstallModal`/`ModelInstallProgress`.

**Tech Stack:** Python 3.12, Textual 8, pytest/pytest-asyncio, onnxruntime (OmniVoice engine), existing `Model_Artifacts` acquisition service.

**Spec:** `Docs/superpowers/specs/2026-09-25-wizard-omnivoice-voice-option-design.md`

**Worktree:** `.worktrees/wizard-omnivoice` on branch `feat/wizard-omnivoice-tts` (off `origin/dev`). Run everything from there. Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=$PWD`. Many `Tests/UI` files trip the ADR-126 storage gate (`RecoveryRequired`) when run *alone*; run UI/Wizard tests together with `Tests/UI/test_app_quit_guard.py` first on the command line, and treat a `RecoveryRequired` failure as environmental only after confirming it fails identically on a pristine `origin/dev` worktree.

## Global Constraints

- Service label is exactly `OmniVoice`; step subtitle: `Hear replies read aloud — optional. PocketTTS or OmniVoice run locally, no account needed; skip with Next if you don't want voice.`
- Provider/model/voice/format saved: `provider_id="omnivoice"`, `model_id="omnivoice-int8hq"`, `voice_id="default"`, `response_format="wav"`.
- Seed rule: keep an existing `[OmniVoiceSettings] seed` (int, `0 <= seed < 2**31`); otherwise generate one random int in `[0, 2**31)` at first use in the step; the SAME seed is used for the Test-and-Hear sample and the save.
- The OmniVoice save event's settings contain only `OMNIVOICE_SEED` — never endpoint or credential keys.
- Next is never blocked by install state; `commit()` refuses only when "Use as default" is checked while OmniVoice is not `ready`.
- Model status is read off the UI thread; install uses `@work(thread=True, group="setup-voice-omnivoice-install", exclusive=True, exit_on_error=False)`.
- Copy strings (verbatim):
  - engine missing: `OmniVoice needs its local engine: pip install "tldw_chatbook[omnivoice_tts]", then run setup again from Settings ▸ Diagnostics ▸ Run Setup Wizard.`
  - model missing: `Downloads the OmniVoice model (1.1 GB, one time).`
  - ready (stable seed): `OmniVoice is installed — runs offline on this computer.`
  - ready (seed not stable — only if Task 1 says so): `OmniVoice is installed — runs offline on this computer. Replies may vary in voice until you create a voice profile in Voice Cloning.`
  - checking: `Checking the OmniVoice model…`
  - generating: `Generating locally (first run loads the model)…`
  - sample failed: `Couldn't play a test sample — you can still save and test later in Speech Lab.`
  - default without model: `Install the OmniVoice model first, or uncheck Use as default.`

## Review Focus

1. Re-running setup on a machine that already has `[OmniVoiceSettings] seed` must keep that seed (a re-run never changes the user's voice) — Task 4 `test_choose_seed_keeps_existing_valid_seed`.
2. A configured `model_root` that points at a missing or partial directory must read as `model_missing`, not crash the step — Task 3 `test_setup_state_broken_model_root_is_model_missing`.
3. Pressing Install twice quickly must start only one preflight — Task 5 `test_install_double_press_runs_one_preflight`.
4. Leaving the step mid-download and coming back must re-read the model state (not show a stale "installing" forever, not cancel the download) — Task 5 `test_reshow_rereads_state_without_cancelling_install`.
5. A sample response that is not WAV audio or exceeds the byte bound must be refused, not played — Task 4 `test_sample_rejects_non_wav_and_oversize`.

---

### Task 1: Measure whether a fixed seed keeps OmniVoice's voice stable (spike)

Throwaway measurement that decides the ready-state copy. No production code.

**Files:**
- Create (scratch, not committed): `$SCRATCH/seed_probe.py` where `$SCRATCH` is the session scratchpad
- Modify: `Docs/superpowers/specs/2026-09-25-wizard-omnivoice-voice-option-design.md` (record the result in "Voice stability")

**Interfaces:**
- Produces: a decision `SEED_STABLE = True|False` recorded in the spec; Task 4 uses it to pick `OMNIVOICE_READY_COPY`.

- [ ] **Step 1: Download the model to scratch (pinned revision)**

```bash
SCRATCH=<session scratchpad>; mkdir -p $SCRATCH/omnivoice-onnx
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('ct03/omnivoice-onnx-int8hq', revision='65c840ba966f4b50cd6bd73f234fb1eba72f9a16',
  local_dir='$SCRATCH/omnivoice-onnx',
  allow_patterns=['omnivoice_lm_int8_hq/*','audio_tokenizer_decoder_int8/*','audio_tokenizer_encoder_int8/*','tokenizer.json','config.json'])"
```

- [ ] **Step 2: Write the probe**

```python
# $SCRATCH/seed_probe.py — synthesize 4 different sentences with one seed and
# with no seed; report median F0 per clip and the Whisper transcript.
import asyncio, io, sys, numpy as np, soundfile as sf
from faster_whisper import WhisperModel
from tldw_chatbook.TTS.backends.omnivoice import OmniVoiceOnnxTTSBackend

ROOT = sys.argv[1]
SENTENCES = [
    "Good morning, here is your daily summary.",
    "The meeting has been moved to three o'clock on Thursday.",
    "I found four articles that match your search.",
    "Would you like me to read the next section aloud?",
]

def f0(wav: np.ndarray, sr: int) -> float:
    fr, out = int(0.04 * sr), []
    for i in range(0, len(wav) - fr, fr // 2):
        w = wav[i:i + fr]
        if np.sqrt(np.mean(w ** 2)) < 0.02 * np.abs(wav).max():
            continue
        ac = np.correlate(w, w, "full")[fr - 1:]
        lo, hi = int(sr / 400), int(sr / 60)
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3 * ac[0]:
            out.append(sr / lag)
    return float(np.median(out))

async def run(seed):
    backend = OmniVoiceOnnxTTSBackend({"OMNIVOICE_MODEL_ROOT": ROOT, "OMNIVOICE_NUM_STEPS": 32,
                                       "OMNIVOICE_LANGUAGE": "en", "OMNIVOICE_SEED": seed})
    asr, rows = WhisperModel("base", device="cpu", compute_type="int8"), []
    for text in SENTENCES:
        data = b"".join([c async for c in backend.generate_speech_stream(text=text, response_format="wav")])
        wav, sr = sf.read(io.BytesIO(data))
        segs, _ = asr.transcribe(io.BytesIO(data), language="en")
        rows.append((f0(wav, sr), " ".join(s.text.strip() for s in segs)))
    await backend.close()
    return rows

for seed in (1234567, None):
    rows = asyncio.run(run(seed))
    f0s = [r[0] for r in rows]
    spread = (max(f0s) - min(f0s)) / np.mean(f0s)
    print(f"seed={seed} f0={[round(x) for x in f0s]} spread={spread:.1%}")
    for r in rows: print("   ", r[1])
```

- [ ] **Step 3: Run it**

Run: `cd <worktree> && PYTHONPATH=$PWD <venv-python> $SCRATCH/seed_probe.py $SCRATCH/omnivoice-onnx`
Expected: two lines of F0 values + spread and 8 transcripts.

- [ ] **Step 4: Decide and record**

Rule: `SEED_STABLE = True` iff the seeded spread is ≤ 12% AND at most half the unseeded spread (the seed clearly narrows the voice), and every transcript matches its sentence. Append to the spec's "Voice stability" section a dated "Measurement" paragraph with both F0 lists, both spreads, and the decision. Listen to two seeded clips if in doubt; the numbers decide.

- [ ] **Step 5: Commit the spec update**

```bash
git add Docs/superpowers/specs/2026-09-25-wizard-omnivoice-voice-option-design.md
git commit -m "docs: record OmniVoice seed-stability measurement for the wizard"
```

Keep `$SCRATCH/omnivoice-onnx` until Task 6's live UAT, then delete it.

---

### Task 2: Per-request seed through the engine, admission and Settings

**Files:**
- Modify: `tldw_chatbook/TTS/backends/omnivoice.py` (`generate_speech_stream`, `_synthesize_codes`, public `managed_model_root`)
- Modify: `tldw_chatbook/TTS/effective_settings.py` (`_validated_options`)
- Modify: `tldw_chatbook/TTS/legacy_catalogs.py` (`LEGACY_REQUEST_OPTION_KEYS["omnivoice"]`)
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py` (`_TTS_SETTING_BINDINGS` OmniVoice list)
- Modify: `Tests/TTS/test_legacy_bridge.py` (pinned omnivoice option tuple)
- Test: `Tests/TTS/test_omnivoice_review_fixes.py` (append)

**Interfaces:**
- Produces: `managed_model_root() -> Path | None` (public; `_managed_root` stays as an alias); `extra_params["seed"]` (int, `0 <= seed < 2**31`) overrides `OMNIVOICE_SEED` for one request; admission accepts option `seed`; Settings binding `OMNIVOICE_SEED` → `[OmniVoiceSettings] seed`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/TTS/test_omnivoice_review_fixes.py`)

```python
async def test_request_seed_overrides_configured_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch, extra_config={"OMNIVOICE_SEED": 7})
    seen: list = []

    def capture(lm, prompt_ids, target_len, *, config, **_):
        seen.append(config.seed)
        return np.zeros((8, target_len), dtype=np.int64)

    monkeypatch.setattr(omnivoice_module, "run_diffusion_sampling", capture)
    request = SimpleNamespace(
        input="hi", voice="", response_format="wav", speed=1.0,
        extra_params={"seed": 424242},
    )
    [c async for c in backend.generate_speech_stream(request)]
    [c async for c in backend.generate_speech_stream(text="hi", voice="")]
    assert seen == [424242, 7]


@pytest.mark.parametrize("bad", [-1, 2**31, True, 1.5, "12"])
async def test_invalid_request_seed_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch, extra_config={"OMNIVOICE_SEED": 7})
    seen: list = []
    monkeypatch.setattr(
        omnivoice_module, "run_diffusion_sampling",
        lambda lm, p, t, *, config, **_: seen.append(config.seed) or np.zeros((8, t), dtype=np.int64),
    )
    request = SimpleNamespace(input="hi", voice="", response_format="wav", speed=1.0,
                              extra_params={"seed": bad})
    [c async for c in backend.generate_speech_stream(request)]
    assert seen == [7]


def test_admission_accepts_seed_in_range() -> None:
    from tldw_chatbook.TTS.effective_settings import (
        TTSEffectiveResolutionError, _validated_options,
    )
    assert dict(_validated_options("omnivoice", {"seed": 0}, None)) == {"seed": 0}
    assert dict(_validated_options("omnivoice", {"seed": 2**31 - 1}, None)) == {"seed": 2**31 - 1}
    for bad in (-1, 2**31, True, 3.0):
        with pytest.raises(TTSEffectiveResolutionError):
            _validated_options("omnivoice", {"seed": bad}, None)


def test_seed_setting_binding_targets_omnivoice_section() -> None:
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import _TTS_SETTING_BINDINGS
    binding = _TTS_SETTING_BINDINGS["OMNIVOICE_SEED"]
    assert binding.destinations == (("OmniVoiceSettings", "seed"),)
    assert binding.provider_id == "omnivoice"


def test_managed_model_root_is_public() -> None:
    assert omnivoice_module.managed_model_root is not None
    assert omnivoice_module._managed_root is omnivoice_module.managed_model_root
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/TTS/test_omnivoice_review_fixes.py -k "seed or managed_model_root_is_public"`
Expected: FAIL (seed ignored → `[7, 7]`; admission raises `unsupported_selection`; `KeyError: 'OMNIVOICE_SEED'`; `AttributeError: managed_model_root`).

- [ ] **Step 3: Implement**

In `tldw_chatbook/TTS/backends/omnivoice.py`:

```python
# rename the function and keep the old private name as an alias
def managed_model_root() -> Path | None:
    """Return the active managed OmniVoice artifact root, if installed.
    ... (existing docstring body of _managed_root) ...
    """
    # (existing _managed_root body, unchanged)


_managed_root = managed_model_root  # existing callers/tests
```

In `generate_speech_stream`, after the `max_reference_duration` block:

```python
        seed_override = extra.get("seed")
        if not (
            type(seed_override) is int and 0 <= seed_override < _SEED_LIMIT
        ):
            seed_override = None
```

with a module constant next to `_MIN_TIMEOUT_S`:

```python
_SEED_LIMIT = 2**31  # per-request seeds are non-negative 31-bit ints
```

Pass it through: add `seed_override,` as the last positional argument in the `self._run_off_thread(cancel_event, self._synthesize_codes, ...)` call (after `language`), add `seed_override: int | None = None,` as the last parameter of `_synthesize_codes`, and in the `OmniVoiceSamplerConfig(...)` call:

```python
            seed=(
                seed_override
                if seed_override is not None
                else self._optional_int("OMNIVOICE_SEED")
            ),
```

In `tldw_chatbook/TTS/effective_settings.py`, inside `_validated_options`, directly after the existing `if key == "num_steps":` block:

```python
        if key == "seed":
            # OmniVoice per-request sampling seed (non-negative 31-bit int).
            if type(option) is not int or not 0 <= option < 2**31:
                raise TTSEffectiveResolutionError(
                    code="invalid_selection",
                    axis="provider_options",
                    source=source,
                )
            normalized[key] = option
            continue
```

In `tldw_chatbook/TTS/legacy_catalogs.py`, the omnivoice entry of `LEGACY_REQUEST_OPTION_KEYS` becomes:

```python
        "omnivoice": (
            "language",
            "num_steps",
            "guidance_scale",
            "max_reference_duration",
            "seed",
        ),
```

In `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`, add `"seed",` to the tuple of names in the `**{f"OMNIVOICE_{name.upper()}": ...}` comprehension (after `"language"`).

In `Tests/TTS/test_legacy_bridge.py`, the omnivoice entry of the pinned options tuple gets `"seed",` appended after `"max_reference_duration",`.

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/TTS/test_omnivoice_*.py Tests/TTS/test_legacy_bridge.py Tests/TTS/test_global_tts_settings_events.py`
Expected: all pass except `test_openai_production_manager_authorizes_actual_effective_origin` (baseline failure on dev).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/backends/omnivoice.py tldw_chatbook/TTS/effective_settings.py tldw_chatbook/TTS/legacy_catalogs.py tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py Tests/TTS/test_omnivoice_review_fixes.py Tests/TTS/test_legacy_bridge.py
git commit -m "feat(omnivoice): per-request seed through engine, admission and Settings"
```

---

### Task 3: Install plumbing next to the OmniVoice catalog

**Files:**
- Modify: `tldw_chatbook/TTS/omnivoice_artifact_catalog.py`
- Modify: `tldw_chatbook/Utils/widget_helpers.py` (reuse `missing_omnivoice_modules` from the catalog)
- Test: `Tests/TTS/test_omnivoice_artifact_catalog.py` (append)

**Interfaces:**
- Consumes: `managed_model_root()`, `resolve_model_root`, `OmniVoiceModelError`, `OmniVoiceNotConfiguredError` from `tldw_chatbook.TTS.backends.omnivoice` (Task 2).
- Produces:
  - `OmniVoiceSetupState = Literal["engine_missing", "model_missing", "ready"]`
  - `missing_omnivoice_modules() -> list[str]`
  - `omnivoice_setup_state(model_root: str | None, *, missing_modules: Callable[[], list[str]] | None = None, managed_root: Callable[[], Path | None] | None = None) -> OmniVoiceSetupState`
  - `class OmniVoiceCatalog` with `descriptor(ref) -> ArtifactDescriptor`
  - `async def run_omnivoice_preflight(*, core=None, credential_resolver=None, free_bytes_probe=None) -> PreflightReport`
  - `async def run_omnivoice_provision(report, *, core=None, credential_resolver=None, free_bytes_probe=None, progress=None) -> Path`

- [ ] **Step 1: Write the failing tests** (append to `Tests/TTS/test_omnivoice_artifact_catalog.py`)

```python
from pathlib import Path

import pytest

from tldw_chatbook.TTS import omnivoice_artifact_catalog as cat


def _tree(root: Path) -> Path:
    for rel in cat.OMNIVOICE_ONNX_REQUIRED_PATHS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    return root


def test_setup_state_engine_missing_wins() -> None:
    state = cat.omnivoice_setup_state(
        None, missing_modules=lambda: ["onnxruntime"], managed_root=lambda: None
    )
    assert state == "engine_missing"


def test_setup_state_model_missing_without_any_root() -> None:
    assert cat.omnivoice_setup_state(
        None, missing_modules=list, managed_root=lambda: None
    ) == "model_missing"


def test_setup_state_ready_from_configured_root(tmp_path: Path) -> None:
    root = _tree(tmp_path / "m")
    assert cat.omnivoice_setup_state(
        str(root), missing_modules=list, managed_root=lambda: None
    ) == "ready"


def test_setup_state_ready_from_managed_root(tmp_path: Path) -> None:
    root = _tree(tmp_path / "m")
    assert cat.omnivoice_setup_state(
        "", missing_modules=list, managed_root=lambda: root
    ) == "ready"


def test_setup_state_broken_model_root_is_model_missing(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    for model_root in (str(tmp_path / "does-not-exist"), str(partial), "bad\x00root"):
        assert cat.omnivoice_setup_state(
            model_root, missing_modules=list, managed_root=lambda: None
        ) == "model_missing"


def test_catalog_serves_only_the_omnivoice_descriptor() -> None:
    catalog = cat.OmniVoiceCatalog()
    assert catalog.descriptor(cat.omnivoice_onnx_reference()).model_id == "omnivoice-onnx-int8hq"
    with pytest.raises(KeyError):
        catalog.descriptor(object())


async def test_wrappers_pass_the_pinned_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []

    class FakeAcquisition:
        def __init__(self, service, **kwargs):
            calls.append(("init", service))

        async def preflight(self, ref, catalog, *, sources):
            calls.append(("preflight", ref, type(catalog).__name__, sources))
            return "REPORT"

        async def provision(self, ref, grant, catalog, *, sources, progress=None):
            calls.append(("provision", ref, grant, sources, progress))
            return ref

    class FakeReport:
        def grant(self):
            return "GRANT"

    class FakeService:
        def artifact_path(self, ref):
            return Path("/managed/omnivoice")

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition

    monkeypatch.setattr(acquisition, "ArtifactAcquisitionService", FakeAcquisition)
    service = FakeService()
    ref = cat.omnivoice_onnx_reference()
    sources = cat.omnivoice_onnx_source_map()

    assert await cat.run_omnivoice_preflight(core=service, credential_resolver=object()) == "REPORT"
    path = await cat.run_omnivoice_provision(
        FakeReport(), core=service, credential_resolver=object(), progress=print
    )
    assert path == Path("/managed/omnivoice")
    assert calls[1] == ("preflight", ref, "OmniVoiceCatalog", sources)
    assert calls[3] == ("provision", ref, "GRANT", sources, print)
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/TTS/test_omnivoice_artifact_catalog.py`
Expected: FAIL with `AttributeError: ... has no attribute 'omnivoice_setup_state'`.

- [ ] **Step 3: Implement** (append to `tldw_chatbook/TTS/omnivoice_artifact_catalog.py`; add `from pathlib import Path`, `from typing import Callable, Literal` to the imports)

```python
OmniVoiceSetupState = Literal["engine_missing", "model_missing", "ready"]


def missing_omnivoice_modules() -> list[str]:
    """Return the OmniVoice runtime modules that are not installed.

    Probed with ``find_spec`` (no import).

    Returns:
        The missing module names, in install order.
    """
    from importlib.util import find_spec

    missing: list[str] = []
    for module_name in ("onnxruntime", "tokenizers"):
        try:
            if find_spec(module_name) is None:
                missing.append(module_name)
        except (ImportError, ValueError):
            missing.append(module_name)
    return missing


def omnivoice_setup_state(
    model_root: str | None,
    *,
    missing_modules: Callable[[], list[str]] | None = None,
    managed_root: Callable[[], Path | None] | None = None,
) -> OmniVoiceSetupState:
    """Report what OmniVoice still needs before it can speak here.

    Resolves the model exactly as the engine does (a configured
    ``model_root`` layout first, else the active managed artifact). Does
    filesystem work — call it off the UI thread.

    Args:
        model_root: ``[OmniVoiceSettings] model_root`` (blank/None = managed).
        missing_modules: Dependency probe (defaults to ``missing_omnivoice_modules``).
        managed_root: Managed-artifact lookup (defaults to the engine's).

    Returns:
        ``engine_missing``, ``model_missing`` or ``ready``.
    """
    from tldw_chatbook.TTS.backends.omnivoice import (
        OmniVoiceModelError,
        OmniVoiceNotConfiguredError,
        managed_model_root,
        resolve_model_root,
    )

    if (missing_modules or missing_omnivoice_modules)():
        return "engine_missing"
    try:
        resolve_model_root(
            {"OMNIVOICE_MODEL_ROOT": model_root or ""},
            managed_root or managed_model_root,
        )
    except (OmniVoiceModelError, OmniVoiceNotConfiguredError):
        return "model_missing"
    return "ready"


class OmniVoiceCatalog:
    """Catalog exposing only the curated OmniVoice ONNX descriptor."""

    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor:
        """Return the OmniVoice descriptor for its exact reference.

        Raises:
            KeyError: For any other reference.
        """
        if ref != omnivoice_onnx_reference():
            raise KeyError(ref)
        return omnivoice_onnx_descriptor()


def _acquisition(core, credential_resolver, free_bytes_probe):
    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactAcquisitionService,
        EnvConfigCredentialResolver,
    )
    from tldw_chatbook.Model_Artifacts.store import managed_service

    service = core if core is not None else managed_service()
    resolver = (
        credential_resolver
        if credential_resolver is not None
        else EnvConfigCredentialResolver()
    )
    return service, ArtifactAcquisitionService(
        service, credential_resolver=resolver, free_bytes_probe=free_bytes_probe
    )


async def run_omnivoice_preflight(
    *, core=None, credential_resolver=None, free_bytes_probe=None
) -> "PreflightReport":
    """Plan the OmniVoice model install (sizes, disk, consent data).

    Returns:
        The preflight report the consent dialog renders.
    """
    _service, acquisition = _acquisition(core, credential_resolver, free_bytes_probe)
    return await acquisition.preflight(
        omnivoice_onnx_reference(),
        OmniVoiceCatalog(),
        sources=omnivoice_onnx_source_map(),
    )


async def run_omnivoice_provision(
    report: "PreflightReport",
    *,
    core=None,
    credential_resolver=None,
    free_bytes_probe=None,
    progress: "Callable[[AcquisitionProgress], None] | None" = None,
) -> Path:
    """Download, verify and activate the OmniVoice model after consent.

    Returns:
        The installed artifact directory.
    """
    service, acquisition = _acquisition(core, credential_resolver, free_bytes_probe)
    installed = await acquisition.provision(
        omnivoice_onnx_reference(),
        report.grant(),
        OmniVoiceCatalog(),
        sources=omnivoice_onnx_source_map(),
        progress=progress,
    )
    return service.artifact_path(installed)
```

Under the existing `if TYPE_CHECKING:` block add `from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress, PreflightReport`.

In `tldw_chatbook/Utils/widget_helpers.py`, delete the body of `missing_omnivoice_modules` and replace the definition with a re-export so there is one implementation:

```python
from tldw_chatbook.TTS.omnivoice_artifact_catalog import (  # noqa: E402
    missing_omnivoice_modules,
)
```

(placed with the module's other imports; remove the old function).

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/UI/test_app_quit_guard.py Tests/TTS/test_omnivoice_artifact_catalog.py Tests/UI/test_voice_cloning_window_omnivoice.py`
Expected: PASS (the voice-cloning alert tests still pass through the re-export).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/omnivoice_artifact_catalog.py tldw_chatbook/Utils/widget_helpers.py Tests/TTS/test_omnivoice_artifact_catalog.py
git commit -m "feat(omnivoice): setup-state probe and install wrappers for the wizard"
```

---

### Task 4: Pure Voice-step helpers for OmniVoice (+ resume schema)

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/first_run_voice_step_state.py`
- Modify: `tldw_chatbook/UI/Wizards/first_run_setup_state.py` (`_SETUP_DRAFT_FIELD_TYPES[STEP_VOICE]`)
- Test: `Tests/Wizards/test_first_run_voice_step_state.py` (append)

**Interfaces:**
- Consumes: Task 2's seed option; Task 1's `SEED_STABLE` decision.
- Produces (all in `first_run_voice_step_state`):
  - `VOICE_PRESET_OMNIVOICE = "omnivoice"`, `OMNIVOICE_MODEL_ID = "omnivoice-int8hq"`, `OMNIVOICE_VOICE_ID = "default"`
  - copy constants `OMNIVOICE_ENGINE_MISSING_COPY`, `OMNIVOICE_MODEL_MISSING_COPY`, `OMNIVOICE_READY_COPY`, `OMNIVOICE_CHECKING_COPY`, `OMNIVOICE_GENERATING_COPY`, `OMNIVOICE_SAMPLE_FAILED_COPY`, `OMNIVOICE_DEFAULT_WITHOUT_MODEL_COPY` (verbatim from Global Constraints; `OMNIVOICE_READY_COPY` per Task 1)
  - `choose_omnivoice_seed(existing: object, *, randbelow: Callable[[int], int] = secrets.randbelow) -> int`
  - `build_omnivoice_save_event(*, speed: float, seed: int, request_id: int | None = None, reply_to: object | None = None) -> STTSSettingsSaveEvent`
  - `async def run_omnivoice_sample(text: str, *, speed: float, seed: int, service: object | None = None, max_response_bytes: int = 8 * 1024 * 1024) -> VoiceSampleResult`
- Resume schema: `STEP_VOICE` gains `"preset": str`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Wizards/test_first_run_voice_step_state.py`)

```python
import io
import wave

from tldw_chatbook.UI.Wizards import first_run_voice_step_state as vs


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"\x00\x00" * 2400)
    return buffer.getvalue()


def test_choose_seed_keeps_existing_valid_seed() -> None:
    assert vs.choose_omnivoice_seed(987654, randbelow=lambda n: 1) == 987654


@pytest.mark.parametrize("existing", [None, "", -1, 2**31, True, "12", 3.5])
def test_choose_seed_generates_when_missing_or_invalid(existing) -> None:
    assert vs.choose_omnivoice_seed(existing, randbelow=lambda n: n - 1) == 2**31 - 1


def test_omnivoice_save_event_shape() -> None:
    event = vs.build_omnivoice_save_event(speed=1.0, seed=42, request_id=3)
    assert dict(event.settings) == {"OMNIVOICE_SEED": 42}
    prefs = event.preferences
    assert (prefs.provider_id, prefs.model_id, prefs.voice_id, prefs.response_format) == (
        "omnivoice", "omnivoice-int8hq", "default", "wav"
    )
    assert event.commit_defaults_after_handoff is True
    assert event.request_id == 3


class _FakeService:
    def __init__(self, chunks):
        self.chunks, self.requests = chunks, []

    async def generate_audio_stream(self, request, internal_model_id, progress_sink=None):
        self.requests.append((request, internal_model_id))
        for chunk in self.chunks:
            yield chunk


async def test_sample_uses_shared_service_with_the_seed() -> None:
    body = _wav_bytes()
    service = _FakeService([body[:100], body[100:]])
    result = await vs.run_omnivoice_sample("Hello.", speed=1.0, seed=42, service=service)
    request, internal = service.requests[0]
    assert internal == "local_omnivoice_default"
    assert request.extra_params == {"seed": 42}
    assert (request.model, request.voice, request.response_format) == (
        "omnivoice-int8hq", "default", "wav"
    )
    assert result.body == body and result.response_format == "wav" and result.playable


async def test_sample_rejects_non_wav_and_oversize() -> None:
    with pytest.raises(ValueError):
        await vs.run_omnivoice_sample("Hi", speed=1.0, seed=1, service=_FakeService([b"ID3mp3"]))
    with pytest.raises(ValueError):
        await vs.run_omnivoice_sample(
            "Hi", speed=1.0, seed=1, service=_FakeService([_wav_bytes()]), max_response_bytes=10
        )
    with pytest.raises(ValueError):
        await vs.run_omnivoice_sample("   ", speed=1.0, seed=1, service=_FakeService([_wav_bytes()]))


def test_resume_schema_accepts_preset() -> None:
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        STEP_VOICE, _SETUP_DRAFT_FIELD_TYPES,
    )
    assert _SETUP_DRAFT_FIELD_TYPES[STEP_VOICE]["preset"] is str
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/Wizards/test_first_run_voice_step_state.py`
Expected: FAIL with `AttributeError: ... 'choose_omnivoice_seed'`.

- [ ] **Step 3: Implement** (add to `first_run_voice_step_state.py`; add `import secrets` and `from collections.abc import Callable` to the imports)

```python
VOICE_PRESET_OMNIVOICE = "omnivoice"
OMNIVOICE_MODEL_ID = "omnivoice-int8hq"
OMNIVOICE_VOICE_ID = "default"
_OMNIVOICE_SEED_LIMIT = 2**31

OMNIVOICE_ENGINE_MISSING_COPY = (
    'OmniVoice needs its local engine: pip install "tldw_chatbook[omnivoice_tts]", '
    "then run setup again from Settings ▸ Diagnostics ▸ Run Setup Wizard."
)
OMNIVOICE_MODEL_MISSING_COPY = "Downloads the OmniVoice model (1.1 GB, one time)."
# Task 1 decides: stable seed -> first form; otherwise append the profile hint.
OMNIVOICE_READY_COPY = "OmniVoice is installed — runs offline on this computer."
OMNIVOICE_CHECKING_COPY = "Checking the OmniVoice model…"
OMNIVOICE_GENERATING_COPY = "Generating locally (first run loads the model)…"
OMNIVOICE_SAMPLE_FAILED_COPY = (
    "Couldn't play a test sample — you can still save and test later in Speech Lab."
)
OMNIVOICE_DEFAULT_WITHOUT_MODEL_COPY = (
    "Install the OmniVoice model first, or uncheck Use as default."
)


def choose_omnivoice_seed(
    existing: object, *, randbelow: Callable[[int], int] = secrets.randbelow
) -> int:
    """Return the seed that fixes OmniVoice's default voice.

    A valid configured seed is kept so a re-run never changes the voice.

    Args:
        existing: ``[OmniVoiceSettings] seed`` as configured (any type).
        randbelow: Random source (injected by tests).

    Returns:
        A non-negative 31-bit integer seed.
    """
    if type(existing) is int and 0 <= existing < _OMNIVOICE_SEED_LIMIT:
        return existing
    return randbelow(_OMNIVOICE_SEED_LIMIT)


def build_omnivoice_save_event(
    *,
    speed: float,
    seed: int,
    request_id: int | None = None,
    reply_to: object | None = None,
) -> STTSSettingsSaveEvent:
    """Build the save event that makes OmniVoice the default voice.

    Args:
        speed: Default speaking rate.
        seed: The voice seed (from ``choose_omnivoice_seed``).
        request_id: Correlates the save result.
        reply_to: Widget receiving the save result.

    Returns:
        A settings event carrying only ``OMNIVOICE_SEED`` plus defaults.
    """
    return STTSSettingsSaveEvent(
        {"OMNIVOICE_SEED": seed},
        preferences=TTSPreferencesSnapshot(
            provider_id=VOICE_PRESET_OMNIVOICE,
            model_mode="exact",
            model_id=OMNIVOICE_MODEL_ID,
            voice_mode="exact",
            voice_id=OMNIVOICE_VOICE_ID,
            response_format="wav",
            speed=speed,
        ),
        request_id=request_id,
        reply_to=reply_to,
        commit_defaults_after_handoff=True,
        notify_outcome=False,
    )


async def run_omnivoice_sample(
    text: str,
    *,
    speed: float,
    seed: int,
    service: object | None = None,
    max_response_bytes: int = 8 * 1024 * 1024,
) -> VoiceSampleResult:
    """Synthesize one sample through the app's shared TTS service.

    Uses the same legacy request path as briefing audio, so the cached
    backend is reused (no second engine) and the seed matches the save.

    Args:
        text: Sample text (1–500 characters after trimming).
        speed: Speaking rate.
        seed: The voice seed the save will persist.
        service: TTS service (defaults to the app's bound service).
        max_response_bytes: Upper bound on the returned audio.

    Returns:
        A playable WAV sample.

    Raises:
        ValueError: Invalid text, oversize audio, or non-WAV audio.
    """
    from tldw_chatbook.TTS.legacy_request_builder import build_legacy_speech_request

    request, internal_model_id = build_legacy_speech_request(
        provider_id=VOICE_PRESET_OMNIVOICE,
        model_id=OMNIVOICE_MODEL_ID,
        voice=OMNIVOICE_VOICE_ID,
        text=validate_voice_sample_text(text),
        response_format="wav",
        speed=speed,
    )
    request.extra_params = {"seed": seed}
    if service is None:
        from tldw_chatbook.TTS.TTS_Generation import get_tts_service

        service = await get_tts_service()
    chunks: list[bytes] = []
    total = 0
    async for chunk in service.generate_audio_stream(request, internal_model_id):
        total += len(chunk)
        if total > max_response_bytes:
            raise ValueError("The TTS sample exceeded the response limit.")
        chunks.append(bytes(chunk))
    body = b"".join(chunks)
    if not (body.startswith(b"RIFF") and body[8:12] == b"WAVE"):
        raise ValueError("OmniVoice returned audio that could not be played.")
    return VoiceSampleResult(
        body=body, content_type="audio/wav", response_format="wav", playable=True
    )
```

In `first_run_setup_state.py`, `_SETUP_DRAFT_FIELD_TYPES[STEP_VOICE]` gains `"preset": str,` as its last key.

Set `OMNIVOICE_READY_COPY` per Task 1: if `SEED_STABLE` is False, use the second ready string from Global Constraints.

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/Wizards/test_first_run_voice_step_state.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/first_run_voice_step_state.py tldw_chatbook/UI/Wizards/first_run_setup_state.py Tests/Wizards/test_first_run_voice_step_state.py
git commit -m "feat(wizard): pure OmniVoice voice-step helpers (seed, save event, sample)"
```

---

### Task 5: OmniVoice panel in the Voice step

**Files:**
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` (`VoiceSetupStep`; resume restore in `SetupWizardContainer`, near the `voice_values = draft.values.get(wizard_state.STEP_VOICE, {})` block)
- Test: `Tests/Wizards/test_first_run_voice_omnivoice.py` (new)

**Interfaces:**
- Consumes: `omnivoice_setup_state`, `run_omnivoice_preflight`, `run_omnivoice_provision` (Task 3); everything listed as produced by Task 4.
- Produces: widget ids `setup-voice-preset-omnivoice`, `setup-voice-omnivoice-panel`, `setup-voice-omnivoice-status`, `setup-voice-omnivoice-install`, `setup-voice-omnivoice-progress`; step data key `preset`.

- [ ] **Step 1: Write the failing tests** (`Tests/Wizards/test_first_run_voice_omnivoice.py`)

```python
"""VoiceSetupStep: OmniVoice as a fourth service (spec 2026-09-25)."""

from __future__ import annotations

import asyncio
import io
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual import on
from textual.widgets import Button, Checkbox, Collapsible, RadioSet, Static

import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module
from Tests.Wizards.test_first_run_setup_wizard import _StepHost
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    STTSSettingsSaveResult,
)
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as vs
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_setup_state import STEP_VOICE
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import VoiceSetupStep


def _step(app_config: dict | None = None) -> VoiceSetupStep:
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config or {}), wizard_data={}
    )
    return VoiceSetupStep(
        wizard=wizard, config=WizardStepConfig(id=STEP_VOICE, title="Voice", step_number=4)
    )


class _Host(_StepHost):
    saved: STTSSettingsSaveEvent | None = None

    @on(STTSSettingsSaveEvent)
    def capture(self, event: STTSSettingsSaveEvent) -> None:
        self.saved = event


async def _select_omnivoice(step: VoiceSetupStep, pilot) -> None:
    step._select_preset_button("setup-voice-preset-omnivoice")
    await pilot.pause(0.2)


def _state(monkeypatch: pytest.MonkeyPatch, value: str) -> list:
    calls: list = []
    monkeypatch.setattr(
        wizard_module, "omnivoice_setup_state",
        lambda model_root, **_: calls.append(model_root) or value,
    )
    return calls


async def test_selecting_omnivoice_swaps_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert step.query_one("#setup-voice-omnivoice-panel").display is False
        await _select_omnivoice(step, pilot)
        assert step.query_one("#setup-voice-omnivoice-panel").display is True
        assert step.query_one("#setup-voice-advanced", Collapsible).display is False
        step._select_preset_button("setup-voice-preset-pocket")
        await pilot.pause(0.2)
        assert step.query_one("#setup-voice-omnivoice-panel").display is False
        assert step.query_one("#setup-voice-advanced", Collapsible).display is True


@pytest.mark.parametrize(
    ("state", "copy", "install_enabled", "test_enabled"),
    [
        ("engine_missing", vs.OMNIVOICE_ENGINE_MISSING_COPY, False, False),
        ("model_missing", vs.OMNIVOICE_MODEL_MISSING_COPY, True, False),
        ("ready", vs.OMNIVOICE_READY_COPY, False, True),
    ],
)
async def test_each_state_renders_copy_and_buttons(
    monkeypatch, state, copy, install_enabled, test_enabled
) -> None:
    _state(monkeypatch, state)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        assert str(step.query_one("#setup-voice-omnivoice-status", Static).renderable) == copy
        install = step.query_one("#setup-voice-omnivoice-install", Button)
        assert (install.display and not install.disabled) is install_enabled
        assert step.query_one("#setup-voice-test", Button).disabled is (not test_enabled)


async def test_install_runs_consent_then_provision_then_rereads(monkeypatch) -> None:
    states = iter(["model_missing", "ready"])
    monkeypatch.setattr(wizard_module, "omnivoice_setup_state", lambda *_a, **_k: next(states))
    order: list = []

    async def preflight(**_):
        order.append("preflight")
        return "REPORT"

    async def provision(report, *, progress=None, **_):
        order.append(("provision", report))
        return None

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", preflight)
    monkeypatch.setattr(wizard_module, "run_omnivoice_provision", provision)
    step = _step()
    host = _Host(step)
    monkeypatch.setattr(host, "push_screen", lambda screen, callback: callback(True))
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        for _ in range(40):
            await pilot.pause(0.05)
        assert order == ["preflight", ("provision", "REPORT")]
        assert str(step.query_one("#setup-voice-omnivoice-status", Static).renderable) == vs.OMNIVOICE_READY_COPY


async def test_install_double_press_runs_one_preflight(monkeypatch) -> None:
    _state(monkeypatch, "model_missing")
    started = asyncio.Event()
    count = {"preflight": 0}

    async def slow_preflight(**_):
        count["preflight"] += 1
        await asyncio.sleep(0.3)
        raise RuntimeError("offline")

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", slow_preflight)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        button = step.query_one("#setup-voice-omnivoice-install", Button)
        button.press()
        button.press()
        for _ in range(20):
            await pilot.pause(0.05)
        assert count["preflight"] == 1


async def test_preflight_failure_shows_message_and_allows_retry(monkeypatch) -> None:
    _state(monkeypatch, "model_missing")

    async def failing(**_):
        raise RuntimeError("network down")

    monkeypatch.setattr(wizard_module, "run_omnivoice_preflight", failing)
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-omnivoice-install", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
        status = str(step.query_one("#setup-voice-omnivoice-status", Static).renderable)
        assert status and status != vs.OMNIVOICE_MODEL_MISSING_COPY
        assert step.query_one("#setup-voice-omnivoice-install", Button).disabled is False


async def test_reshow_rereads_state_without_cancelling_install(monkeypatch) -> None:
    calls = _state(monkeypatch, "model_missing")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        before = len(calls)
        step._omnivoice_installing = True
        step.on_hide()
        step.on_show()
        await pilot.pause(0.2)
        assert len(calls) == before + 1
        assert step._omnivoice_installing is True


async def test_commit_without_default_saves_nothing(monkeypatch) -> None:
    _state(monkeypatch, "ready")
    step = _step()
    host = _Host(step)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        assert await step.commit() == (True, "")
        assert host.saved is None


async def test_commit_default_without_model_is_refused(monkeypatch) -> None:
    _state(monkeypatch, "model_missing")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-default", Checkbox).value = True
        assert await step.commit() == (False, vs.OMNIVOICE_DEFAULT_WITHOUT_MODEL_COPY)


async def test_commit_default_saves_omnivoice_with_the_sampled_seed(monkeypatch) -> None:
    _state(monkeypatch, "ready")
    seeds: list = []

    async def sample(text, *, speed, seed, **_):
        seeds.append(seed)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(b"\x00\x00" * 240)
        return vs.VoiceSampleResult(buffer.getvalue(), "audio/wav", "wav", True)

    monkeypatch.setattr(vs, "run_omnivoice_sample", sample)
    step = _step({"COMPREHENSIVE_CONFIG_RAW": {"OmniVoiceSettings": {"seed": 777}}})
    host = _Host(step)
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        step.query_one("#setup-voice-test", Button).press()
        for _ in range(20):
            await pilot.pause(0.05)
        step.query_one("#setup-voice-default", Checkbox).value = True
        commit = asyncio.create_task(step.commit())
        await pilot.pause(0.1)
        assert host.saved is not None
        assert dict(host.saved.settings) == {"OMNIVOICE_SEED": 777}
        assert seeds == [777]
        step.receive_stts_settings_save_result(
            STTSSettingsSaveResult(
                request_id=host.saved.request_id,
                persisted=True,
                provider_statuses={"omnivoice": "applied"},
                provider_configuration_revisions={"omnivoice": 1},
                provider_runtime_revisions={"omnivoice": 1},
                defaults_activated=True,
                defaults_activation_status="committed",
            )
        )
        assert await commit == (True, "")


async def test_step_data_records_the_preset(monkeypatch) -> None:
    _state(monkeypatch, "ready")
    step = _step()
    async with _Host(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _select_omnivoice(step, pilot)
        assert step.get_step_data()["preset"] == "omnivoice"


async def test_service_row_fits_at_80_columns(monkeypatch) -> None:
    import html
    import re

    _state(monkeypatch, "ready")
    step = _step()
    host = _Host(step)
    async with host.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        painted = html.unescape(
            "".join(re.findall(r">([^<>]*)</text>", host.export_screenshot()))
        ).replace("\xa0", " ")
        for label in ("PocketTTS", "OmniVoice"):
            assert label in painted
```

Also add a resume test to `Tests/Wizards/test_first_run_setup_wizard.py`, next to `test_voice_resume_restores_all_non_secret_controls` (it uses that file's `_make_wizard`/`_HostApp`):

```python
@pytest.mark.asyncio
async def test_voice_resume_restores_the_omnivoice_preset(monkeypatch):
    import tldw_chatbook.UI.Wizards.FirstRunSetupWizard as wizard_module

    monkeypatch.setattr(wizard_module, "omnivoice_setup_state", lambda *_a, **_k: "ready")
    resume = SetupDraft(
        version=SETUP_DRAFT_VERSION,
        track=TRACK_QUICK,
        active_step_id=STEP_VOICE,
        values={
            STEP_WELCOME: {"track": TRACK_QUICK},
            STEP_VOICE: {"preset": "omnivoice", "use_as_default": True},
        },
    )
    wizard = _make_wizard(resume_draft=resume)
    app = _HostApp(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container = wizard.query_one(SetupWizardContainer)
        step = container.steps[container._step_index_for_id(STEP_VOICE)]
        assert step._preset == "omnivoice"
        assert step.query_one("#setup-voice-omnivoice-panel").display is True
        assert step.query_one("#setup-voice-default", Checkbox).value is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/UI/test_app_quit_guard.py Tests/Wizards/test_first_run_voice_omnivoice.py`
Expected: FAIL (`NoMatches` for `#setup-voice-omnivoice-panel`, `AttributeError: _select_preset_button`).

- [ ] **Step 3: Implement in `VoiceSetupStep`**

Imports at the top of `FirstRunSetupWizard.py`:

```python
from tldw_chatbook.TTS.omnivoice_artifact_catalog import (
    omnivoice_setup_state,
    run_omnivoice_preflight,
    run_omnivoice_provision,
)
```

`__init__` additions:

```python
        self._omnivoice_state: str | None = None
        self._omnivoice_installing = False
        self._omnivoice_report: Any = None
        self._omnivoice_seed: int | None = None
        self._save_provider = "openai"
        self._save_use_as_default = False
```

`compose_step`: replace the subtitle string with the Global Constraints subtitle; add the fourth radio inside `#setup-voice-preset` after the custom button:

```python
                yield SetupRadioButton("OmniVoice", id="setup-voice-preset-omnivoice")
```

and, directly after the `SetupRadioSet` block, the panel:

```python
            with Vertical(id="setup-voice-omnivoice-panel") as panel:
                panel.display = False
                yield Static(vs.OMNIVOICE_CHECKING_COPY, id="setup-voice-omnivoice-status",
                             classes="setup-subtitle")
                install = Button("Install voice model", id="setup-voice-omnivoice-install")
                install.display = False
                yield install
                progress = ModelInstallProgress(None, id="setup-voice-omnivoice-progress")
                progress.display = False
                yield progress
```

(`vs` here means `voice_state`; the module is already imported as `voice_state` in this file — use that name.)

Helpers and handlers (add to `VoiceSetupStep`):

```python
    def _select_preset_button(self, button_id: str) -> None:
        """Press one service radio exactly as a user would (tests + resume)."""
        self.query_one(f"#{button_id}", RadioButton).value = True

    def _omnivoice_settings(self) -> Mapping[str, object]:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        raw = app_config.get("COMPREHENSIVE_CONFIG_RAW") if isinstance(app_config, Mapping) else None
        source = raw if isinstance(raw, Mapping) else app_config
        section = source.get("OmniVoiceSettings") if isinstance(source, Mapping) else None
        return section if isinstance(section, Mapping) else {}

    def _omnivoice_seed_value(self) -> int:
        if self._omnivoice_seed is None:
            self._omnivoice_seed = voice_state.choose_omnivoice_seed(
                self._omnivoice_settings().get("seed")
            )
        return self._omnivoice_seed

    def _set_omnivoice_mode(self, enabled: bool) -> None:
        self.query_one("#setup-voice-omnivoice-panel").display = enabled
        self.query_one("#setup-voice-advanced", Collapsible).display = not enabled
        if enabled:
            self.query_one("#setup-voice-add-key", Button).display = False
            self._load_omnivoice_state()
        self._invalidate_sample_evidence()

    @work(thread=True, group="setup-voice-omnivoice-state", exclusive=True, exit_on_error=False)
    def _load_omnivoice_state(self) -> None:
        model_root = self._omnivoice_settings().get("model_root")
        try:
            state = omnivoice_setup_state(model_root if isinstance(model_root, str) else None)
        except Exception:
            logger.opt(exception=True).warning("OmniVoice setup state read failed")
            state = "model_missing"
        self.app.call_from_thread(self._apply_omnivoice_state, state)

    def _apply_omnivoice_state(self, state: str, message: str | None = None) -> None:
        self._omnivoice_state = state
        copy = {
            "engine_missing": voice_state.OMNIVOICE_ENGINE_MISSING_COPY,
            "model_missing": voice_state.OMNIVOICE_MODEL_MISSING_COPY,
            "ready": voice_state.OMNIVOICE_READY_COPY,
        }[state]
        try:
            self.query_one("#setup-voice-omnivoice-status", Static).update(message or copy)
            install = self.query_one("#setup-voice-omnivoice-install", Button)
            install.display = state in {"engine_missing", "model_missing"}
            install.disabled = state != "model_missing" or self._omnivoice_installing
        except NoMatches:
            return
        self._refresh_sample_state()

    @on(Button.Pressed, "#setup-voice-omnivoice-install")
    def _on_omnivoice_install(self, event: Button.Pressed) -> None:
        event.stop()
        if self._omnivoice_installing or self._omnivoice_state != "model_missing":
            return
        self._omnivoice_installing = True
        self.query_one("#setup-voice-omnivoice-install", Button).disabled = True
        self._refresh_sample_state()
        self._omnivoice_preflight()

    @work(thread=True, group="setup-voice-omnivoice-install", exclusive=True, exit_on_error=False)
    def _omnivoice_preflight(self) -> None:
        try:
            report = asyncio.run(run_omnivoice_preflight())  # policy-exception: worker-thread loop
        except Exception as exc:
            logger.opt(exception=True).error("OmniVoice model preflight failed")
            self.app.call_from_thread(
                self._finish_omnivoice_install,
                install_failure_message(exc, model_label="OmniVoice voice model"),
            )
            return
        self.app.call_from_thread(self._show_omnivoice_consent, report)

    def _show_omnivoice_consent(self, report: Any) -> None:
        self._omnivoice_report = report
        self.app.push_screen(
            ModelInstallModal(
                report,
                model_label="OmniVoice voice model",
                container_id="setup-voice-omnivoice-install-modal",
                confirm_id="setup-voice-omnivoice-install-confirm",
                cancel_id="setup-voice-omnivoice-install-cancel",
            ),
            self._confirm_omnivoice_install,
        )

    def _confirm_omnivoice_install(self, confirmed: bool) -> None:
        if not confirmed:
            self._finish_omnivoice_install(None)
            return
        self._omnivoice_provision()

    @work(thread=True, group="setup-voice-omnivoice-install", exclusive=True, exit_on_error=False)
    def _omnivoice_provision(self) -> None:
        report = self._omnivoice_report
        try:
            asyncio.run(  # policy-exception: worker-thread loop
                run_omnivoice_provision(report, progress=make_progress_callback(self.post_message))
            )
        except Exception as exc:
            logger.opt(exception=True).error("OmniVoice model installation failed")
            self.app.call_from_thread(
                self._finish_omnivoice_install,
                install_failure_message(exc, model_label="OmniVoice voice model"),
            )
            return
        self.app.call_from_thread(self._finish_omnivoice_install, None)

    def _finish_omnivoice_install(self, error: str | None) -> None:
        self._omnivoice_installing = False
        self._omnivoice_report = None
        try:
            self.query_one("#setup-voice-omnivoice-progress", ModelInstallProgress).display = False
        except NoMatches:
            pass
        if error is not None:
            self._apply_omnivoice_state("model_missing", message=error)
            return
        self._load_omnivoice_state()

    @on(InstallProgressed)
    def _omnivoice_install_progressed(self, event: InstallProgressed) -> None:
        event.stop()
        try:
            progress = self.query_one("#setup-voice-omnivoice-progress", ModelInstallProgress)
        except NoMatches:
            return
        progress.display = True
        progress.update_progress(event.progress)
```

`_on_preset`: add `"setup-voice-preset-omnivoice": voice_state.VOICE_PRESET_OMNIVOICE,` to the mapping, and replace the tail after the early `return`s with:

```python
        if preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._preset = preset
            self._set_omnivoice_mode(True)
            return
        leaving_omnivoice = self._preset == voice_state.VOICE_PRESET_OMNIVOICE
        try:
            current = self._draft_from_controls()
        except (TypeError, ValueError):
            self.show_step_error("Enter a valid speed before changing the service preset.")
            return
        if self._preset == voice_state.VOICE_PRESET_CUSTOM:
            self._custom_draft = current
        self._preset = preset
        if leaving_omnivoice:
            self._set_omnivoice_mode(False)
        base = (
            self._custom_draft
            if preset == voice_state.VOICE_PRESET_CUSTOM and self._custom_draft is not None
            else current
        )
        self._apply_draft_to_controls(voice_state.apply_voice_preset(base, preset))
```

`_refresh_sample_state`: at the top of its `try:` block, after updating `#setup-voice-sample-count`, insert:

```python
            if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
                self.query_one("#setup-voice-add-key", Button).display = False
                self.query_one("#setup-voice-test", Button).disabled = (
                    self._test_in_progress_generation is not None
                    or self._omnivoice_installing
                    or self._omnivoice_state != "ready"
                    or not 1 <= trimmed_count <= 500
                )
                return
```

`on_show`: after `super().on_show()`:

```python
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._load_omnivoice_state()
```

`_on_test_and_hear`: at the start:

```python
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            self._start_omnivoice_sample()
            return
```

with:

```python
    def _speed_or_default(self) -> float:
        try:
            speed = float(self.query_one("#setup-voice-speed", Input).value)
        except ValueError:
            return 1.0
        return speed if math.isfinite(speed) and 0.25 <= speed <= 4.0 else 1.0

    def _start_omnivoice_sample(self) -> None:
        if self._omnivoice_state != "ready" or self._omnivoice_installing:
            return
        text = self.query_one("#setup-voice-sample", Input).value
        self._test_generation += 1
        generation = self._test_generation
        self._test_in_progress_generation = generation
        self.query_one("#setup-voice-status", Static).update(voice_state.OMNIVOICE_GENERATING_COPY)
        self._refresh_sample_state()
        self.run_worker(
            self._run_omnivoice_sample(generation, text, self._speed_or_default(),
                                       self._omnivoice_seed_value()),
            exclusive=True, group="setup-voice-sample", exit_on_error=False,
        )

    async def _run_omnivoice_sample(self, generation: int, text: str, speed: float, seed: int) -> None:
        try:
            result = await voice_state.run_omnivoice_sample(text, speed=speed, seed=seed)
        except asyncio.CancelledError:
            raise
        except Exception:
            if generation == self._test_generation:
                self._test_in_progress_generation = None
                self.query_one("#setup-voice-status", Static).update(
                    voice_state.OMNIVOICE_SAMPLE_FAILED_COPY
                )
                self._refresh_sample_state()
            return
        if generation != self._test_generation:
            return
        self._test_in_progress_generation = None
        played = await self._play_sample(result)
        self.query_one("#setup-voice-status", Static).update(
            "Verified. The sample is ready to hear."
            if played else "Verified, playback failed. Retry playback/test."
        )
        self._refresh_sample_state()
```

`commit()`: at the very top:

```python
        if self._preset == voice_state.VOICE_PRESET_OMNIVOICE:
            return await self._commit_omnivoice()
```

with:

```python
    async def _commit_omnivoice(self) -> tuple[bool, str]:
        if not self.query_one("#setup-voice-default", Checkbox).value:
            return True, ""
        if self._omnivoice_state != "ready":
            return False, voice_state.OMNIVOICE_DEFAULT_WITHOUT_MODEL_COPY
        request_id = self._next_save_request_id
        self._next_save_request_id += 1
        self._save_request_id = request_id
        self._save_provider = "omnivoice"
        self._save_use_as_default = True
        self._save_future = asyncio.get_running_loop().create_future()
        self.app.post_message(
            voice_state.build_omnivoice_save_event(
                speed=self._speed_or_default(),
                seed=self._omnivoice_seed_value(),
                request_id=request_id,
                reply_to=self,
            )
        )
        try:
            return await asyncio.wait_for(
                asyncio.shield(self._save_future), timeout=self._SAVE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return False, "Voice settings are still applying. Retry to continue."
        finally:
            self._save_request_id = None
            self._save_future = None
            self._save_provider = "openai"
```

In the existing HTTP `commit()` path, next to `self._save_draft = draft`, add `self._save_use_as_default = draft.use_as_default`.

`_receive_save_result`: replace every literal `"openai"` with `provider = self._save_provider` (read once at the top), and replace the `draft = self._save_draft ... if not draft.use_as_default` block with:

```python
        if not self._save_use_as_default:
            future.set_result((True, ""))
            return
```

`get_step_data`: add `"preset": self._preset,` to `values`.

Resume restore in `SetupWizardContainer` (the `voice_values` block): after the existing `_restore_radio_selection(... "setup-voice-preset-custom")` call, add:

```python
                if voice_values.get("preset") == voice_state.VOICE_PRESET_OMNIVOICE:
                    voice_step._preset = voice_state.VOICE_PRESET_OMNIVOICE
                    self._restore_radio_selection(
                        voice_step.query_one("#setup-voice-preset", RadioSet),
                        lambda button: button.id == "setup-voice-preset-omnivoice",
                    )
                    voice_step._set_omnivoice_mode(True)
```

and relax the guard so a preset-only value set still restores: the restored draft already falls back to `initial` for every missing key, so no other change is needed.

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=$PWD <venv-python> -m pytest -q -p no:cacheprovider Tests/UI/test_app_quit_guard.py Tests/Wizards/test_first_run_voice_omnivoice.py Tests/Wizards/test_first_run_setup_wizard.py Tests/Wizards/test_first_run_voice_step_state.py`
Expected: PASS (compare any failure against a pristine `origin/dev` worktree before accepting it as baseline).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/Wizards/test_first_run_voice_omnivoice.py Tests/Wizards/test_first_run_setup_wizard.py
git commit -m "feat(wizard): OmniVoice as a fourth Voice service (install, test, save)"
```

---

### Task 6: Docs, preflight, live UAT, backlog task, PR

**Files:**
- Modify: `Docs/User_Guide/First_Run_Setup.md` (Voice row + paragraph + "Verified against" stamp)
- Create: `backlog/tasks/task-<id> - Add-OmniVoice-to-the-first-run-Voice-step.md`
- Possibly modify: `Docs/security/production-diagnostic-inventory.json` (only via the checker, after reviewing its diff)

- [ ] **Step 1: Update the user guide**

In `Docs/User_Guide/First_Run_Setup.md`: the Voice table row becomes `| Voice | Spoken replies — PocketTTS, OmniVoice (local, installs its model here), Official OpenAI or a compatible endpoint; sample + "Test and Hear" | Settings ▸ Speech & TTS |`; append to the Voice paragraph: "Choosing **OmniVoice** shows a local panel instead of the endpoint fields: it tells you if the `omnivoice_tts` engine is missing, installs the 1.1 GB model through the same consent dialog as the model browser, and tests a sample on this computer. Saving it as the default also fixes its voice seed so replies keep the same voice."; update the top `Verified against` line to name this change and today's date.

- [ ] **Step 2: Run preflight**

Run: `PYTHON=<venv-python> ./scripts/preflight.sh; echo rc=$?`
Expected: `rc=0`. If the diagnostic inventory reports drift, read the rows with `--statements <file> --since <base>`, confirm no raw paths are logged, then `scripts/check_persistent_diagnostic_inventory.py --write`, and re-run.

- [ ] **Step 3: Live UAT (tmux, isolated HOME profile)**

Launch the app on a fresh scratch profile (`HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH` all under the scratchpad; minimal config `[general] users_name = "wizard_uat"`) at 80×24 and at 200×60. In the wizard: Voice → OmniVoice → confirm model_missing copy → Install → consent shows the Terms line → download completes → ready copy → Test and Hear (copy the played temp WAV, transcribe with faster-whisper `base`; must match the sample text) → check Use as default → Next → finish. Then verify `[OmniVoiceSettings] seed` is written in the scratch config and Speech Lab opens on OmniVoice. Also start a Parakeet install in the Speech step while OmniVoice downloads once. At 80×24 confirm the service row paints "OmniVoice" and "Custom compatible" legibly; if the latter clips, rename its label to "Custom" (and update any test that pins that label).

- [ ] **Step 4: Backlog task**

Sweep for the next free id across all refs and worktrees:
`git for-each-ref --format='%(refname)' refs/remotes/origin refs/heads | while read r; do git ls-tree -r --name-only "$r" -- backlog/tasks backlog/drafts backlog/archive; done | sed -n 's/.*task-\([0-9][0-9]*\)[ .].*/\1/p' | sort -n | tail -1` (plus `ls` of every worktree's `backlog/tasks`). Create the task file (status Done) with Description (why), 5 ACs (select OmniVoice; install in-wizard with consent; local test sample; save as default with stable seed; Next never blocked), and Implementation Notes (approach, Task 1 measurement result, files).

- [ ] **Step 5: Commit, push, open PR**

```bash
git add Docs/User_Guide/First_Run_Setup.md "backlog/tasks/task-<id> - Add-OmniVoice-to-the-first-run-Voice-step.md" Docs/security/production-diagnostic-inventory.json
git commit -m "docs: first-run Voice step OmniVoice option (user guide, task)"
git push -u origin feat/wizard-omnivoice-tts
gh pr create --base dev --title "feat(wizard): OmniVoice as a first-run Voice option" --body-file $SCRATCH/pr_body.md
```

`$SCRATCH/pr_body.md` contains: a Summary (the four user-visible behaviors), links to the spec and this plan, the Task 1 seed measurement (both F0 lists, spreads, decision), the test evidence (new test files and pass counts, baseline failures confirmed on `origin/dev`), the live UAT results (Whisper transcript of the sample, the written seed, Speech Lab default), and the `🤖 Generated with [Claude Code](https://claude.com/claude-code)` footer.

Delete `$SCRATCH/omnivoice-onnx` and the scratch profile after UAT.
