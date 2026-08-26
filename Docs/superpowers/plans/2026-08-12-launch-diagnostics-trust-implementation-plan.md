# Launch Diagnostics and Trust Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make startup rendering, notifications, metrics, and TTS resource diagnostics accurate, bounded, private by default, and equivalent in terminal and browser modes.

**Architecture:** Adapt legacy splash effects at the registry boundary into typed frames, centralize frame rendering and lifecycle fencing in the splash widget, and introduce typed startup outcomes for optional subsystems. Keep one explicit Prometheus exporter path, disabled by default, and package the required OpenAI TTS mapping as an importlib resource while treating model/voice downloads as lazy optional assets.

**Tech Stack:** Python 3.11+, Textual, Rich `Text`, prometheus-client, importlib.resources, setuptools wheel builds, pytest, browser/terminal Pilot tests.

## Global Constraints

- Existing splash effects may keep returning strings in this slice; the registry adapter must declare their format.
- Rendering never guesses frame format from content.
- Plain frames render with markup disabled; ANSI frames pass through one sanitizer before `Text.from_ansi`; Rich renderables pass directly.
- External metrics are disabled unless configuration explicitly enables them or `METRICS_PORT` is set.
- Metrics bind to loopback unless a remote host is explicitly configured.
- Exactly one metrics exporter server may start per process.
- Optional TTS models and voice packs are never downloaded or checked during unrelated startup.
- Use `Tests/...` as the canonical test path spelling.

---

## File Structure

- Modify `tldw_chatbook/Utils/Splash_Screens/base_effect.py`: typed frames, frame kinds, registry metadata, and legacy adapter.
- Modify `tldw_chatbook/Utils/Splash_Screens/__init__.py`: export typed frame APIs.
- Modify `tldw_chatbook/Utils/Splash_Screens/card_definitions.py`: explicit static fallback and effect frame kind for every card.
- Modify `tldw_chatbook/Widgets/splash_screen.py`: typed rendering, generation fences, first-paint timing, and deterministic fallback.
- Create `tldw_chatbook/Utils/startup_outcomes.py`: shared typed subsystem startup outcome.
- Create `tldw_chatbook/UI/startup_notifications.py`: bounded generation-scoped notification queue.
- Modify `tldw_chatbook/app.py`: consume typed outcomes and notification generations.
- Modify `tldw_chatbook/Metrics/metrics.py`: idempotent loopback exporter startup with typed outcome.
- Modify `tldw_chatbook/Metrics/Otel_Metrics.py`: no implicit exporter startup and typed availability outcome.
- Create `tldw_chatbook/Config_Files/openai_tts_mappings.json`: required packaged mappings already expected by `config.load_openai_mappings`.
- Modify `pyproject.toml`: include the mapping in package data.
- Add focused splash, metrics, notification, and wheel tests.

### Task 1: Define typed splash frames and legacy effect metadata

**Files:**
- Modify: `tldw_chatbook/Utils/Splash_Screens/base_effect.py`
- Modify: `tldw_chatbook/Utils/Splash_Screens/__init__.py`
- Modify: `tldw_chatbook/Utils/Splash_Screens/card_definitions.py`
- Create: `Tests/Widgets/test_splash_frames.py`

**Interfaces:**
- Produces `SplashFrameKind = Literal["plain", "ansi", "rich"]`.
- Produces `SplashFrame(kind, content, duration)` where Rich content is a Rich renderable and other content is text.
- Extends `register_effect(name, *, frame_kind="rich")`.
- Produces `next_splash_frame(effect) -> SplashFrame | None` as the sole legacy adapter.

- [ ] **Step 1: Write failing frame and registry tests**

```python
def test_legacy_string_effect_is_wrapped_with_declared_kind():
    @register_effect("test_plain", frame_kind="plain")
    class PlainEffect(BaseEffect):
        def update(self):
            return "[not markup]"
    frame = next_splash_frame(PlainEffect(None))
    assert frame == SplashFrame(kind="plain", content="[not markup]", duration=0.1)


def test_every_registered_card_has_static_fallback_and_frame_kind():
    load_all_effects()
    for name, card in get_all_card_definitions().items():
        assert isinstance(card.get("content"), str) and card["content"].strip(), name
        if card.get("effect"):
            assert card.get("frame_kind") in {"plain", "ansi", "rich"}, name
```

Add validation tests rejecting mismatched explicit frame content and unknown frame kinds.

- [ ] **Step 2: Run tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/Widgets/test_splash_frames.py -v`

Expected: FAIL because effects return untyped strings and card definitions do not declare frame kinds consistently.

- [ ] **Step 3: Implement frame records and registry metadata**

```python
SplashFrameKind: TypeAlias = Literal["plain", "ansi", "rich"]

@dataclass(frozen=True, slots=True)
class SplashFrame:
    kind: SplashFrameKind
    content: str | Text
    duration: float = 0.1

@dataclass(frozen=True, slots=True)
class EffectRegistration:
    effect_class: type[BaseEffect]
    frame_kind: SplashFrameKind
```

`next_splash_frame` returns an existing `SplashFrame` unchanged or wraps a legacy string with the registration's declared kind. Register existing Rich-markup effects as `rich`; classify any literal escape-sequence producers as `ansi`; static literal art is `plain`.

- [ ] **Step 4: Add deterministic fallback content to every card**

Use each card's existing `content` where present. Where absent, use one exported ASCII-only `MINIMAL_SPLASH_FALLBACK` containing `tldw chatbook`; do not synthesize fallback from an animation frame.

Run: `.venv/bin/python -m pytest Tests/Widgets/test_splash_frames.py -v`

Expected: PASS for every registered card/effect.

- [ ] **Step 5: Commit typed splash frames**

```bash
git add tldw_chatbook/Utils/Splash_Screens/base_effect.py tldw_chatbook/Utils/Splash_Screens/__init__.py tldw_chatbook/Utils/Splash_Screens/card_definitions.py Tests/Widgets/test_splash_frames.py
git commit -m "feat: type splash animation frames"
```

### Task 2: Render frames safely and fence splash lifecycle races

**Files:**
- Modify: `tldw_chatbook/Widgets/splash_screen.py:70-505`
- Modify: `Tests/Widgets/test_splash_frames.py`
- Modify: `Tests/Widgets/test_splash_screen_config_read.py`
- Modify: `Tests/UI/test_settings_splash_screen_viewer.py`
- Create: `Tests/Widgets/test_splash_lifecycle.py`

**Interfaces:**
- Produces `render_splash_frame(frame) -> Text | RenderableType`.
- Splash widget owns monotonically increasing `_frame_generation` and `_closed` state.
- Auto-close timer begins only after `_paint_first_frame` succeeds.

- [ ] **Step 1: Write failing renderer and lifecycle tests**

```python
def test_plain_frame_keeps_brackets_literal():
    rendered = render_splash_frame(SplashFrame("plain", "[bold]literal[/bold]"))
    assert rendered.plain == "[bold]literal[/bold]"


async def test_late_frame_after_close_is_ignored():
    splash = SplashScreen(card_name="test-delayed", duration=10)
    async with splash.run_test() as pilot:
        generation = splash._frame_generation
        splash.close()
        splash._apply_frame(generation, SplashFrame("plain", "late"))
        await pilot.pause()
        assert "late" not in splash.query_one("#splash-display", Static).render().plain
```

Add ANSI sanitizer tests that remove OSC/control sequences while preserving SGR color, first-paint timer tests, resize-generation tests, compact fallback tests, and reduced-motion static tests.

- [ ] **Step 2: Run tests and observe raw markup rendering**

Run: `.venv/bin/python -m pytest Tests/Widgets/test_splash_frames.py Tests/Widgets/test_splash_lifecycle.py Tests/Widgets/test_splash_screen_config_read.py -v`

Expected: FAIL because `Static.update` receives untyped strings, timing starts on mount, and there is no generation fence.

- [ ] **Step 3: Implement the single renderer**

```python
def render_splash_frame(frame: SplashFrame):
    if frame.kind == "plain":
        return Text(str(frame.content), no_wrap=False)
    if frame.kind == "ansi":
        return Text.from_ansi(sanitize_splash_ansi(str(frame.content)))
    return frame.content
```

The sanitizer permits printable text, newline/tab, and SGR sequences only; it removes OSC, cursor movement, title changes, and other control sequences. Every widget update calls this renderer.

- [ ] **Step 4: Add first-paint and generation fencing**

Wait until display width/height are positive, increment the generation before constructing an effect, paint the first frame, then start interval and duration timers. Resize increments generation and restarts frame creation. Close increments generation, marks closed, and stops all timers. `_apply_frame` returns without mutation when closed or generation differs.

- [ ] **Step 5: Run terminal/browser presentation tests**

Run: `.venv/bin/python -m pytest Tests/Widgets/test_splash_frames.py Tests/Widgets/test_splash_lifecycle.py Tests/Widgets/test_splash_screen_config_read.py Tests/UI/test_settings_splash_screen_viewer.py -v`

Expected: PASS with literal bracket text, bounded frame dimensions, deterministic compact fallback, and no late updates.

- [ ] **Step 6: Commit splash rendering and lifecycle**

```bash
git add tldw_chatbook/Widgets/splash_screen.py Tests/Widgets/test_splash_frames.py Tests/Widgets/test_splash_lifecycle.py Tests/Widgets/test_splash_screen_config_read.py Tests/UI/test_settings_splash_screen_viewer.py
git commit -m "fix: render splash frames safely across clients"
```

### Task 3: Fence and deduplicate startup notifications

**Files:**
- Create: `tldw_chatbook/UI/startup_notifications.py`
- Modify: `tldw_chatbook/app.py:8740-9105,9410-9520,10420-10480`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py`
- Create: `Tests/UI/test_startup_notifications.py`
- Modify: `Tests/UI/test_first_run_wizard_live_contract.py`

**Interfaces:**
- Produces `StartupNotification(generation, code, message, severity)`.
- Produces `StartupNotificationQueue.begin_generation`, `.stage`, `.drain`, and `.discard`.
- Deduplicates by `(generation, code)` and enforces a fixed maximum queue size.

- [ ] **Step 1: Write failing generation and deduplication tests**

```python
def test_old_generation_notifications_are_discarded():
    queue = StartupNotificationQueue(max_pending=8)
    first = queue.begin_generation()
    queue.stage(first, "theme", "Theme loaded", "information")
    second = queue.begin_generation()
    assert queue.drain(second) == ()


def test_duplicate_code_is_emitted_once():
    queue = StartupNotificationQueue(max_pending=8)
    generation = queue.begin_generation()
    queue.stage(generation, "setup", "Setup available", "information")
    queue.stage(generation, "setup", "Setup available", "information")
    assert len(queue.drain(generation)) == 1
```

Add first-run tests proving setup errors stay inline and welcome/setup toasts are withheld while the wizard is open.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `.venv/bin/python -m pytest Tests/UI/test_startup_notifications.py Tests/UI/test_first_run_wizard_live_contract.py -k "notification or toast or inline" -v`

Expected: FAIL because deferred notifications are not owned by a startup generation.

- [ ] **Step 3: Implement bounded notification ownership**

```python
@dataclass(frozen=True, slots=True)
class StartupNotification:
    generation: int
    code: str
    message: str
    severity: Literal["information", "warning", "error"]
```

Stage non-critical startup messages until the splash/wizard closes, drain only the current generation after the destination screen paints, and drop the oldest non-error item at capacity. Critical setup validation remains inline and is not duplicated into this queue.

- [ ] **Step 4: Run startup notification regressions**

Run: `.venv/bin/python -m pytest Tests/UI/test_startup_notifications.py Tests/UI/test_first_run_wizard_live_contract.py Tests/UI/test_product_maturity_phase1_first_run.py -v`

Expected: PASS.

- [ ] **Step 5: Commit notification fencing**

```bash
git add tldw_chatbook/UI/startup_notifications.py tldw_chatbook/app.py tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py Tests/UI/test_startup_notifications.py Tests/UI/test_first_run_wizard_live_contract.py Tests/UI/test_product_maturity_phase1_first_run.py
git commit -m "fix: fence startup notifications by generation"
```

### Task 4: Make metrics explicit, loopback-only by default, and truthful

**Files:**
- Create: `tldw_chatbook/Utils/startup_outcomes.py`
- Modify: `tldw_chatbook/Metrics/metrics.py:245-260`
- Modify: `tldw_chatbook/Metrics/Otel_Metrics.py:63-110`
- Modify: `tldw_chatbook/app.py:10920-10948`
- Create: `Tests/Metrics/test_metrics_startup.py`
- Modify: `Tests/Utils/test_metrics_logger.py`

**Interfaces:**
- Produces `StartupStatus = Started | Already running | Disabled | Unavailable | Degraded | Failed` as a `StrEnum`.
- Produces `StartupOutcome(subsystem, status, detail_code, host, port)` with bounded fields.
- Produces `resolve_external_metrics_config(app_config, environ) -> ExternalMetricsConfig`.
- `init_metrics_server(host, port) -> StartupOutcome` is process-idempotent.
- `init_otel_metrics(exporter_enabled=False) -> StartupOutcome` never starts another HTTP exporter.

- [ ] **Step 1: Write failing default-off and idempotence tests**

```python
def test_metrics_are_disabled_without_config_or_environment():
    config = resolve_external_metrics_config({}, {})
    assert not config.enabled
    assert config.host == "127.0.0.1"


def test_metrics_port_environment_opts_in_on_loopback(monkeypatch):
    config = resolve_external_metrics_config({}, {"METRICS_PORT": "9010"})
    assert config == ExternalMetricsConfig(True, "127.0.0.1", 9010)


def test_metrics_initialization_is_idempotent(fake_start_http_server):
    first = init_metrics_server("127.0.0.1", 9010)
    second = init_metrics_server("127.0.0.1", 9010)
    assert first.status is StartupStatus.STARTED
    assert second.status is StartupStatus.ALREADY_RUNNING
    assert fake_start_http_server.call_count == 1
```

Add invalid-port, explicit remote-bind, unavailable dependency, bind failure, and one-outcome-log tests.

- [ ] **Step 2: Run metrics tests and confirm default port startup fails**

Run: `.venv/bin/python -m pytest Tests/Metrics/test_metrics_startup.py Tests/Utils/test_metrics_logger.py -v`

Expected: FAIL because app startup currently binds port 8000 by default, then starts a second OTel Prometheus exporter and logs unconditional success.

- [ ] **Step 3: Implement typed outcomes and explicit configuration**

```python
class StartupStatus(StrEnum):
    STARTED = "started"
    ALREADY_RUNNING = "already_running"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class StartupOutcome:
    subsystem: str
    status: StartupStatus
    detail_code: str | None = None
    host: str | None = None
    port: int | None = None
```

Read `[metrics.external].enabled/host/port` and `METRICS_PORT`; environment port alone opts in. Reject invalid ports. A non-loopback host requires explicit config. Protect exporter state with a lock and return `ALREADY_RUNNING` for identical repeat calls or `FAILED/config_conflict` for a different address.

- [ ] **Step 4: Remove the second exporter and unconditional success logs**

Keep the existing prometheus-client exporter as the sole HTTP path. OTel initialization may install internal instruments only when available; it must not construct `PrometheusMetricReader`. The main block logs exactly the returned outcome once and says Disabled/Unavailable/Failed when appropriate.

- [ ] **Step 5: Run metrics and startup regressions**

Run: `.venv/bin/python -m pytest Tests/Metrics/test_metrics_startup.py Tests/Utils/test_metrics_logger.py Tests/Widgets/test_performance_metrics_nonblocking.py -v`

Expected: PASS and no network listener is created by the default configuration test.

- [ ] **Step 6: Commit private metrics startup**

```bash
git add tldw_chatbook/Utils/startup_outcomes.py tldw_chatbook/Metrics/metrics.py tldw_chatbook/Metrics/Otel_Metrics.py tldw_chatbook/app.py Tests/Metrics/test_metrics_startup.py Tests/Utils/test_metrics_logger.py Tests/Widgets/test_performance_metrics_nonblocking.py
git commit -m "fix: make external metrics explicit and idempotent"
```

### Task 5: Package required TTS mappings and keep optional assets lazy

**Files:**
- Create: `tldw_chatbook/Config_Files/openai_tts_mappings.json`
- Modify: `pyproject.toml:478-485`
- Modify: `tldw_chatbook/config.py:419-465`
- Modify: `tldw_chatbook/TTS/backends/kokoro.py:250-330`
- Create: `Tests/Packaging/conftest.py`
- Create: `Tests/Packaging/test_built_wheel_resources.py`
- Create: `Tests/TTS/test_optional_tts_assets.py`

**Interfaces:**
- Required resource: `tldw_chatbook.Config_Files/openai_tts_mappings.json`.
- Produces `OptionalTTSAssetStatus = Literal["not_checked", "not_installed", "installed"]`.
- Kokoro asset checks run only when its Settings/voice surface or explicit setup action requests them.

- [ ] **Step 1: Write failing resource and lazy-check tests**

```python
def test_installed_wheel_contains_openai_tts_mappings(installed_wheel_python):
    result = installed_wheel_python(
        "from importlib.resources import files; "
        "import json; "
        "p=files('tldw_chatbook.Config_Files').joinpath('openai_tts_mappings.json'); "
        "d=json.loads(p.read_text(encoding='utf-8')); "
        "assert d['models']['tts-1']=='openai_official_tts-1'; "
        "assert d['voices']['alloy']=='alloy'"
    )
    assert result.returncode == 0, result.stderr


def test_unrelated_startup_does_not_check_or_download_kokoro_assets(monkeypatch):
    checked = []
    monkeypatch.setattr(Path, "exists", lambda path: checked.append(path) or False)
    initialize_application_optional_services(enable_tts=False)
    assert checked == []
```

The wheel fixture builds with `python -m build --wheel`, installs into a temporary target, and runs with that target first on `PYTHONPATH`; it must not import from the source tree.

Implement the fixture in `Tests/Packaging/conftest.py` with `subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", wheel_dir], check=True)`, then install the resulting wheel using `[sys.executable, "-m", "pip", "install", "--no-deps", "--target", install_dir, wheel_path]`. Its returned runner sets `cwd` outside the repository and invokes `python -I -c` with a wrapper that inserts `install_dir` at `sys.path[0]` before executing the supplied script; isolated mode must not import the source checkout.

- [ ] **Step 2: Run tests and confirm the missing resource**

Run: `.venv/bin/python -m pytest Tests/Packaging/test_built_wheel_resources.py Tests/TTS/test_optional_tts_assets.py -v`

Expected: FAIL because the loader expects `openai_tts_mappings.json`, the file is absent, and package data cannot include it.

- [ ] **Step 3: Add the canonical mapping resource**

Create valid JSON with the same complete `models` and `voices` entries currently declared in `config.openai_tts_mappings`, including `tts-1`, `tts-1-hd`, ElevenLabs, Kokoro, and the existing voice aliases. Change package data to:

```toml
"tldw_chatbook.Config_Files" = ["*.json", "*.md", "rag_pipelines.toml"]
```

Keep built-in minimal defaults for a damaged installation, but return a typed degraded startup diagnostic instead of logging an expected warning for a correctly built wheel.

- [ ] **Step 4: Make optional model/voice checks user-triggered**

Initialize Kokoro status as `not_checked`. Its relevant UI requests one check and displays **Not installed** with an explicit Download/Choose files action. Download helpers run only from that action and never from module import, app startup, or provider setup.

- [ ] **Step 5: Build and test the wheel**

Run: `.venv/bin/python -m build --wheel --no-isolation`

Expected: one wheel under `dist/`.

Run: `.venv/bin/python -m pytest Tests/Packaging/test_built_wheel_resources.py Tests/TTS/test_optional_tts_assets.py Tests/UI/test_product_maturity_phase6_packaging_data_safety.py -v`

Expected: PASS using the installed wheel resource, with no source-tree fallback.

- [ ] **Step 6: Commit TTS distribution trust fixes**

```bash
git add tldw_chatbook/Config_Files/openai_tts_mappings.json pyproject.toml tldw_chatbook/config.py tldw_chatbook/TTS/backends/kokoro.py Tests/Packaging/conftest.py Tests/Packaging/test_built_wheel_resources.py Tests/TTS/test_optional_tts_assets.py Tests/UI/test_product_maturity_phase6_packaging_data_safety.py
git commit -m "fix: package required TTS mappings"
```

### Task 6: Run the launch and diagnostics slice gate

**Files:**
- Verify only; modify failures only when caused by this plan.

- [ ] **Step 1: Run focused lint**

Run: `.venv/bin/python -m ruff check tldw_chatbook/Utils/Splash_Screens tldw_chatbook/Widgets/splash_screen.py tldw_chatbook/UI/startup_notifications.py tldw_chatbook/Utils/startup_outcomes.py tldw_chatbook/Metrics tldw_chatbook/config.py tldw_chatbook/TTS/backends/kokoro.py`

Expected: PASS.

- [ ] **Step 2: Run splash and notification regressions**

Run: `.venv/bin/python -m pytest Tests/Widgets/test_splash_frames.py Tests/Widgets/test_splash_lifecycle.py Tests/Widgets/test_splash_screen_config_read.py Tests/UI/test_settings_splash_screen_viewer.py Tests/UI/test_startup_notifications.py Tests/UI/test_first_run_wizard_live_contract.py -v`

Expected: PASS.

- [ ] **Step 3: Run diagnostics and packaging regressions**

Run: `.venv/bin/python -m pytest Tests/Metrics/test_metrics_startup.py Tests/Utils/test_metrics_logger.py Tests/Widgets/test_performance_metrics_nonblocking.py Tests/Packaging/test_built_wheel_resources.py Tests/TTS/test_optional_tts_assets.py Tests/UI/test_product_maturity_phase6_packaging_data_safety.py -v`

Expected: PASS.

- [ ] **Step 4: Commit gate-only corrections when needed**

```bash
git add tldw_chatbook pyproject.toml Tests
git commit -m "test: close launch diagnostic regressions"
```

Skip this commit when the gate requires no corrections.
