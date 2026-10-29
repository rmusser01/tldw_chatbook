"""TASK-34100.6 slice F: one honest status line and a visible 'Check key'.

Review findings provider-06 (no visible way to check a cloud key),
provider-03 (a rejected key advanced with ✓; Provider stayed ✓ through a
failed check), cross-cutting-07 (errors before the user acted; 'test again'
with no control on screen) and provider-21 (a hard-coded model clause).

The catalog scope service is replaced by an in-process stand-in returning a
``ModelDiscoveryResult`` -- the same object the real ``discover_models``
returns -- so these run offline; the live checks are in Implementation Notes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input

from Tests.Wizards.provider_step_harness import (
    SIZE,
    WizardHost,
    current_step,
    key_input,
    pick_provider,
    pinned_error,
    resumed_on_provider,
    static_text,
    tracker_states,
    wait_for_step,
    wait_until,
)
from tldw_chatbook.Chat import local_server_discovery
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    DiscoveredModel,
    ModelDiscoveryError,
    ModelDiscoveryResult,
)
from tldw_chatbook.UI.Wizards.first_run_provider_check import key_check_line
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
    STEP_VOICE,
)

pytestmark = pytest.mark.bootstrap_profile

_STATUS = "#setup-provider-probe-status"
_PAID = "can't be checked without a paid request"


def _listed(provider: str, *model_ids: str) -> ModelDiscoveryResult:
    return ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider,
        endpoint_fingerprint=None,
        status="success",
        models=tuple(
            DiscoveredModel(
                provider=provider,
                provider_list_key=provider,
                model_id=model_id,
                display_name=model_id,
                source="runtime_discovery",
                endpoint_fingerprint="",
                discovered_at="2026-10-04T00:00:00Z",
            )
            for model_id in model_ids
        ),
    )


def _failed(provider: str, kind: str, category: str) -> ModelDiscoveryResult:
    return ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider,
        endpoint_fingerprint=None,
        status="error",
        error=ModelDiscoveryError(
            kind=kind, message="x", recovery_hint="y", category=category
        ),
    )


def _scope(result_for):
    calls: list[dict] = []

    async def discover_models(**kwargs):
        calls.append(kwargs)
        return result_for(kwargs)

    return SimpleNamespace(discover_models=discover_models), calls


def _wizard(monkeypatch, result_for, *env_cleared: str):
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", *env_cleared):
        monkeypatch.delenv(name, raising=False)

    async def no_scan(_config, **_kwargs):
        return ()

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", no_scan)
    wizard = resumed_on_provider()
    scope, calls = _scope(result_for)
    wizard.app_instance.llm_provider_catalog_scope_service = scope
    return wizard, calls


async def _type_key(pilot, step, value: str) -> None:
    field = key_input(step)
    field.focus()
    field.value = value
    await pilot.pause()


@pytest.mark.asyncio
async def test_check_key_sits_beside_the_key_for_keyed_providers_only(monkeypatch):
    wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-x"))
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        button = step.query_one("#setup-provider-check-key", Button)
        assert button.display and str(button.label) == "Check key"
        assert button.parent is key_input(step).parent, "beside the key field"
        # Cloud rows lose the dead endpoint panel and its two buttons.
        assert not step.query_one("#setup-provider-connection").display
        assert not step.query_one("#setup-provider-detect", Button).display

        await pick_provider(pilot, step, "ollama")
        assert not button.display


@pytest.mark.asyncio
async def test_nothing_is_an_error_before_the_user_acts(monkeypatch):
    """cross-cutting-07: a keyed provider without a key reads neutral."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await pilot.pause(0.2)
        assert static_text(step, _STATUS) == ""
        help_line = static_text(step, "#setup-provider-key-status")
        assert help_line.startswith("Needs an OpenAI API key — paste it above")
        assert help_line.index("paste it above") < help_line.index("OPENAI_API_KEY")
        assert pinned_error(wizard) == ""


@pytest.mark.asyncio
async def test_a_working_key_reads_key_works_with_its_model_count(monkeypatch):
    wizard, calls = _wizard(
        monkeypatch, lambda kw: _listed("openai", "gpt-a", "gpt-b", "gpt-c")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✓"))
        assert static_text(step, _STATUS) == "✓ Key works — OpenAI returned 3 models."
        assert step.query_one(_STATUS).has_class("-ok")
        sent = calls[-1]["staged_settings"]["api_settings"]["openai"]
        assert sent.get("api_key"), "the check lists models with the draft key"

        # cross-cutting-07: an edit clears the verdict; no 'test again' here.
        await _type_key(pilot, step, "sk-proj-" + "b" * 40)
        assert static_text(step, _STATUS) == ""


@pytest.mark.asyncio
async def test_enter_in_a_filled_key_field_runs_the_same_check(monkeypatch):
    wizard, calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-a"))
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pilot.press("enter")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✓"))
        assert static_text(step, _STATUS) == "✓ Key works — OpenAI returned 1 model."


@pytest.mark.asyncio
async def test_a_rejected_key_is_named_under_the_field_and_refuses_next(monkeypatch):
    """provider-03: a 401 is definitive -- no ✓, no 'continue' wording."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        assert static_text(step, _STATUS) == "✗ OpenAI rejected this key."
        note = step.query_one("#setup-provider-key-note")
        assert note.has_class("-error")
        assert "rejected this key" in str(note.render())

        ok, refusal = await step.commit()
        assert not ok
        assert "rejected this key" in refusal and "continue" not in refusal.lower()
        # Live finding: Next cleared the echo AND the field error; the error
        # stays, and only an action that works is named (no hidden Skip).
        assert note.has_class("-error") and "rejected this key" in str(note.render())
        assert "Ctrl+U" in refusal and "Skip" not in refusal
        assert static_text(step, "#setup-provider-key-status") == ""


@pytest.mark.asyncio
async def test_a_forbidden_check_is_explained_and_never_blocks_next(monkeypatch):
    """provider-03 / AC#13 as amended: the owner ruled on 2026-10-02 (Qodo
    #2958) that a 403 means this key may not list models, not that it cannot
    chat -- it is explained, never a refusal."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "forbidden")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("!"))
        assert "(HTTP 403)" in static_text(step, _STATUS)
        assert "rejected" not in static_text(step, _STATUS)
        assert not step.query_one("#setup-provider-key-note").has_class("-error")

        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert pinned_error(wizard) == ""


@pytest.mark.asyncio
async def test_after_a_forbidden_check_the_tracker_keeps_provider_at_attention(
    monkeypatch,
):
    """Review v3r1, G6-BV3-03 (live: a real Baseten 403): Provider said '!
    … won't list models for this key (HTTP 403)', Next advanced as ruled --
    and the tracker then showed ✓ on Provider over the step's own '!'.
    Provider wears '!' after Next and on every later step, while the 403
    still never blocks: Model asks no 'Continue anyway?' and Next goes on."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "forbidden")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: "(HTTP 403)" in static_text(step, _STATUS))

        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        container, model_step = current_step(wizard)
        await wait_until(
            pilot, lambda: bool(model_step.query("#setup-model-listing-unavailable"))
        )
        await wait_until(pilot, lambda: tracker_states(wizard)[STEP_PROVIDER] == "attention")
        assert container.provider_probe_failure() == ""  # still no failure: no gate
        assert model_step.confirm_before_advance() is None

        model_step.query_one("#setup-model-custom", Input).value = "gpt-4.1-mini"
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)
        assert pinned_error(wizard) == ""
        states = tracker_states(wizard)
        assert states[STEP_PROVIDER] == "attention"
        assert states[STEP_MODEL] == "complete"


@pytest.mark.asyncio
async def test_after_a_rejected_key_ctrl_u_works_where_the_refusal_says(monkeypatch):
    """Live 2026-10-04 (AC#1 path by keyboard): after Back from a rejected key,
    focus sat on the list, so the 'clear it (Ctrl+U)' the copy names did
    nothing and Next refused again. Back lands on the key field."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await wait_until(pilot, lambda: tracker_states(wizard)[STEP_PROVIDER] == "attention")

        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        await pilot.pause()
        assert app.focused is key_input(step)
        await pilot.press("ctrl+u")
        assert key_input(step).value == ""
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)
        states = tracker_states(wizard)
        assert states[STEP_PROVIDER] == "skipped" and states[STEP_MODEL] == "skipped"


@pytest.mark.asyncio
async def test_a_refused_next_puts_focus_on_the_key_it_names(monkeypatch):
    """Next pressed from the Next button: the refusal says Ctrl+U, so the
    key field takes focus and Ctrl+U clears it."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        wizard.query_one("#wizard-next", Button).focus()
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: "Ctrl+U" in pinned_error(wizard))
        await pilot.pause()
        assert app.focused is key_input(step)
        await pilot.press("ctrl+u")
        assert key_input(step).value == ""


@pytest.mark.asyncio
async def test_a_failed_check_flips_provider_to_attention_without_waiting_for_model(
    monkeypatch,
):
    """provider-03: '!' as soon as the failure is known, before 'Continue anyway?'."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("ollama", "request_failed", "connection_refused")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "ollama")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await wait_until(pilot, lambda: tracker_states(wizard)[STEP_PROVIDER] == "attention")


@pytest.mark.asyncio
async def test_a_key_rejected_after_model_shows_still_flips_provider(monkeypatch):
    """Live 2026-10-04: Next pressed at once on a typed Anthropic key. The
    check settled while Model showed '(loading models…)', so the tracker read
    Model's still-empty verdict and kept ✓ on Provider for good, beside Model's
    own 'Authentication failed'."""
    wizard, _calls = _wizard(monkeypatch, lambda kw: None, "ANTHROPIC_API_KEY")

    async def slow_rejection(**_kwargs):
        await asyncio.sleep(0.4)  # a real listing round trip
        return _failed("anthropic", "missing_credentials", "unauthorized")

    wizard.app_instance.llm_provider_catalog_scope_service = SimpleNamespace(
        discover_models=slow_rejection
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "anthropic")
        await _type_key(pilot, step, "sk-ant-api03-" + "a" * 40)
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _c, model = current_step(wizard)
        await wait_until(
            pilot, lambda: bool(model.query("#setup-model-connection-failed")), tries=250
        )
        await pilot.pause(0.2)
        assert tracker_states(wizard)[STEP_PROVIDER] == "attention"


def test_a_public_listing_never_confirms_a_key():
    """provider-06: OpenRouter's list answers any key; only key_verified can."""
    listed = _listed("openrouter", "a/b", "c/d")
    line, rejected = key_check_line("OpenRouter", listed, public_listing=True)
    assert line.startswith("! OpenRouter's model list answers without a key")
    assert not rejected
    # TASK-34100.2's authenticated check sets key_verified on the result.
    verified = SimpleNamespace(
        status="success", models=listed.models, error=None, key_verified=True
    )
    assert key_check_line("OpenRouter", verified, public_listing=True)[0] == (
        "✓ Key works — OpenRouter returned 2 models."
    )


@pytest.mark.parametrize(
    "result",
    [
        # Azure's real shape: refused before any request (no listing route
        # the wizard can use), status 'unsupported' with no category.
        ModelDiscoveryResult(
            provider="azure",
            provider_list_key="azure",
            endpoint_fingerprint=None,
            status="unsupported",
            error=ModelDiscoveryError(
                kind="unsupported_endpoint",
                message="This endpoint is not a valid OpenAI-compatible models endpoint.",
                recovery_hint="y",
            ),
        ),
        _failed("azure", "missing_endpoint", ""),
        None,
    ],
    ids=["unsupported", "missing-endpoint", "no-result"],
)
def test_a_check_that_sent_nothing_never_claims_a_paid_request(result):
    """Review round 2, G6-VR2-07: Azure's Check key sent no request (live: 0
    TCP connections, even for a resource URL that doesn't resolve) yet said
    it "can't be checked without a paid request" -- a claim about the
    provider nobody asked. A check that sent nothing says only that the key
    isn't checked here."""
    line, rejected = key_check_line("Azure OpenAI", result, public_listing=False)
    assert line == (
        "! Azure OpenAI's key isn't checked here — it's checked when you send "
        "your first message."
    )
    assert not rejected and _PAID not in line


@pytest.mark.asyncio
async def test_leaving_a_typed_key_says_it_is_not_saved_unless_chosen(monkeypatch):
    """coverage-08: the key stays in memory, and the step says it isn't saved."""
    wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("x", "m"))
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        await pick_provider(pilot, step, "ollama")
        await pilot.pause(0.2)
        note = "#setup-provider-unsaved-note"
        assert static_text(step, note) == (
            "! Your OpenAI key won't be saved unless you choose OpenAI."
        )
        assert step.query_one(note).display
        await pick_provider(pilot, step, "openai")
        assert key_input(step).value == "sk-proj-" + "a" * 40
        assert not step.query_one(note).display


@pytest.mark.asyncio
async def test_a_key_setup_already_saved_is_never_called_unsaved(monkeypatch):
    """Live 2026-10-27 (g6bv5-key/11): a Baseten key was saved when Model's
    Next committed it; Back to Provider and a move to another provider then
    said '! Your Baseten key won't be saved unless you choose Baseten.' --
    the config already held it, and choosing another provider removes
    nothing. A typed key equal to the one saved for its provider is no
    unsaved key (coverage-08 names only keys that would be lost)."""
    typed = "sk-proj-" + "a" * 40
    wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-4.1-mini"))
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, typed)
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _c, model_step = current_step(wizard)
        model_step.query_one("#setup-model-custom", Input).value = "gpt-4.1-mini"
        await pilot.pause()
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_VOICE)
        saved = wizard.app_instance.app_config["api_settings"]["openai"]
        assert saved.get("api_key") == typed  # the commit saved it

        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        await pilot.press("ctrl+b")
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "ollama")
        await pilot.pause(0.2)
        note = "#setup-provider-unsaved-note"
        assert "won't be saved" not in static_text(step, note)
        assert not step.query_one(note).display

        # A different key typed over the saved one would be lost: still said.
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "b" * 40)
        await pick_provider(pilot, step, "ollama")
        await pilot.pause(0.2)
        assert static_text(step, note) == (
            "! Your OpenAI key won't be saved unless you choose OpenAI."
        )


@pytest.mark.asyncio
async def test_an_http_error_names_its_status_and_an_extra_path_gets_a_hint():
    """provider-15: 'HTTP 500 — …', and a 404 under /api suggests removing it."""
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    outcomes = [
        SettingsEndpointProbeOutcome(
            state="unreachable", category="http_status", summary="unreachable: HTTP 500"
        ),
        SettingsEndpointProbeOutcome(
            state="model_listing_unavailable",
            category="http_status",
            summary="Model listing unavailable; chat endpoint not tested",
        ),
    ]
    probe = AsyncMock(side_effect=outcomes)
    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        step._probe = probe
        await pick_provider(pilot, step, "custom")
        endpoint = step.query_one("#setup-provider-endpoint", Input)
        endpoint.value = "http://127.0.0.1:9099"
        await pilot.pause()
        step._launch_probe()
        await wait_until(pilot, lambda: "HTTP" in static_text(step, _STATUS))
        assert static_text(step, _STATUS) == (
            "✗ HTTP 500 — the server hit an error; check its logs, then Test again."
        )
        endpoint.value = "http://127.0.0.1:9099/api"
        await pilot.pause()
        step._launch_probe()
        await wait_until(pilot, lambda: "remove" in static_text(step, _STATUS))
        assert static_text(step, _STATUS) == (
            "! The server answered, but there's no model list under /api — "
            "remove /api from the endpoint?"
        )
        assert step.query_one(_STATUS).has_class("-warn"), "a hint, never a block"


def test_the_model_count_is_the_listing_not_the_evidence_bound():
    """Live finding: OpenAI's key check read 'returned 100 models' -- the
    evidence keeps at most 100 ids; the line counts the whole listing."""
    listed = _listed("openai", *(f"m-{index}" for index in range(137)))
    assert key_check_line("OpenAI", listed, public_listing=False)[0] == (
        "✓ Key works — OpenAI returned 137 models."
    )



def test_the_expert_route_is_a_dim_second_line():
    """cross-cutting-07: the on-screen fix first; env/TOML on a dim second line."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Wizards.first_run_provider_key_entry import key_help_line

    line = key_help_line(
        "openai", SimpleNamespace(reason="Missing API key", env_var="OPENAI_API_KEY")
    )
    first, second = line.plain.split("\n")
    assert first.startswith("Needs an OpenAI API key — paste it above")
    assert "OPENAI_API_KEY" not in first
    assert second == "Or export OPENAI_API_KEY, or add api_key to config.toml."
    # Set apart in italics, never Rich 'dim': dim over the muted line measured
    # 3.72:1 (review round 1, live F8); the muted colour itself is AA-pinned
    # in test_theme_contrast.py.
    styles = [str(span.style) for span in line.spans]
    assert any("italic" in style for style in styles)
    assert not any("dim" in style for style in styles)


_EXPORTED_OPENAI_KEY = "sk-proj-" + "e" * 40


@pytest.mark.asyncio
async def test_a_rejected_exported_key_names_fixes_that_work_and_skip_advances(
    monkeypatch,
):
    """Review round 1, F1 (both reviewers): a revoked OPENAI_API_KEY blocked
    Provider. The refusal named Ctrl+U for a hidden field, the Skip button
    was hidden because the env var made the provider 'ready', and the skip
    itself returned early for the same reason."""
    from tldw_chatbook.UI.Wizards.first_run_provider_skip import SKIP_LABEL

    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        # The pick's own listing is the check: a 401 on the exported key.
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        assert static_text(step, _STATUS) == "✗ OpenAI rejected the key in OPENAI_API_KEY."
        assert not key_input(step).display, "the environment is the default source"
        note = static_text(step, "#setup-provider-key-note")
        assert "OPENAI_API_KEY" in note and "paste a new one" not in note
        skip = step.query_one("#setup-provider-skip", Button)
        assert skip.display, "a rejected key keeps the skip in reach"

        ok, refusal = await step.commit()
        assert not ok
        assert "Ctrl+U" not in refusal
        assert "Store a different key for this app" in refusal and SKIP_LABEL in refusal

        await pilot.click("#setup-provider-skip")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        assert step.get_step_data()["skipped"] is True
        assert wizard.query_one("SetupWizardContainer").staged_provider_draft is None


@pytest.mark.asyncio
async def test_the_key_line_never_repeats_a_check_already_run(monkeypatch):
    """Review round 1, live F6: 'Press Check key to try it.' stayed beside the
    settled verdict, and 'nothing to store' stayed after choosing to store."""
    wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-a"))
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        assert "Press Check key" in static_text(step, "#setup-provider-key-status")
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✓"))
        key_line = static_text(step, "#setup-provider-key-status")
        assert key_line == "Key staged."

    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-a"))
    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✓"))
        key_line = static_text(step, "#setup-provider-key-status")
        assert "OPENAI_API_KEY" in key_line and "Press Check key" not in key_line

        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause()
        key_line = static_text(step, "#setup-provider-key-status")
        assert "nothing is stored" not in key_line
        assert key_line.startswith("Paste the key to store above")


@pytest.mark.asyncio
async def test_a_saved_key_that_outranks_the_environment_offers_one_control(
    monkeypatch,
):
    """Review round 1, F5: with a saved key and OPENAI_API_KEY both set, the
    radio claimed the environment while the saved key was the one used."""
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    async def no_scan(_config, **_kwargs):
        return ()

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", no_scan)
    wizard = resumed_on_provider(
        lambda config: config.setdefault("api_settings", {})
        .setdefault("openai", {})
        .update(api_key="sk-proj-" + "s" * 40)
    )
    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        assert step._credential_at_request_boundary()[0] == "stored"
        assert not step.query_one("#setup-provider-key-source").display
        assert step.query_one("#setup-provider-key-actions").display
        key_line = static_text(step, "#setup-provider-key-status")
        # Pre-merge review (2026-10-29), minor 6: scoped to Console, where it
        # holds for every provider (see the engine-preset test below).
        assert "Console uses it instead of OPENAI_API_KEY." in key_line


@pytest.mark.asyncio
async def test_a_saved_engine_preset_key_names_only_where_it_outranks_the_variable(
    monkeypatch,
):
    """Pre-merge review (2026-10-29), minor 6: 'It's used instead of
    TOGETHER_API_KEY' was not true of an engine preset everywhere: the hosted
    engine reads the variable before the stored key unless its caller passes
    a key (Docs/Architecture/llm-providers.md, "Credential precedence").
    Console passes readiness's key, which is the stored one, so the line
    scopes the claim to Console."""
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    async def no_scan(_config, **_kwargs):
        return ()

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", no_scan)
    wizard = resumed_on_provider(
        lambda config: config.setdefault("api_settings", {})
        .setdefault("together", {})
        .update(api_key="tgp-saved-" + "s" * 40)
    )
    monkeypatch.setenv("TOGETHER_API_KEY", "tgp-exported-" + "e" * 40)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "together")
        assert step._credential_at_request_boundary()[0] == "stored"
        key_line = static_text(step, "#setup-provider-key-status")
        assert "Console uses it instead of TOGETHER_API_KEY." in key_line
        assert "It's used instead of" not in key_line


async def _custom_at(pilot, step, endpoint: str) -> None:
    await pick_provider(pilot, step, "custom")
    field = step.query_one("#setup-provider-endpoint", Input)
    field.value = endpoint
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "wanted", "unwanted"),
    [
        (
            "tls_untrusted",
            ("certificate isn't trusted", "Settings ▸ Network"),
            ("verification off", "verify=False", "couldn't reach"),
        ),
        ("tls_plain_http", ("use http:// instead of https://",), ("couldn't reach",)),
    ],
)
async def test_provider_names_a_certificate_or_plain_http_failure(
    monkeypatch, category, wanted, unwanted
):
    """AC#7 / AC#19 on the Provider step's own line (review round 1, TE-01:
    only the classifier and Model's copy were tested). No inline toggle turns
    verification off."""
    async def no_scan(_config, **_kwargs):
        return ()

    monkeypatch.setattr(local_server_discovery, "discover_local_servers", no_scan)
    # Configured, so the pick's own check is the one that reports (an edit
    # clears the line and waits for Test connection, Check again or Next).
    wizard = resumed_on_provider(
        lambda config: config.setdefault("api_settings", {})
        .setdefault("custom", {})
        .update(api_url="https://127.0.0.1:8443/v1/chat/completions")
    )
    scope, _calls = _scope(lambda kw: _failed("custom", "request_failed", category))
    wizard.app_instance.llm_provider_catalog_scope_service = scope
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "custom")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("!"))
        line = static_text(step, _STATUS)
        for text in wanted:
            assert text in line
        for text in unwanted:
            assert text.lower() not in line.lower()
        if "Check again" in line:  # review round 2, R2-04: only a visible control
            assert step.query_one("#setup-provider-no-server").display


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (
            "tls_untrusted",
            "! OpenAI answered, but its certificate isn't trusted. Add your CA "
            "bundle in Settings ▸ Network, then press Check key.",
        ),
        ("tls_plain_http", "! This server speaks plain HTTP — use http:// instead of https://."),
    ],
)
def test_a_key_check_names_a_certificate_or_plain_http_cause(category, expected):
    """Review round 2, R2-03: Check key (and Enter in a filled key field)
    read every transport failure as "didn't answer; it's checked when you
    send your first message" -- a key behind a TLS-inspecting proxy
    (coverage-19's primary case) got no Settings ▸ Network pointer and a
    promise the first message would fail the same way."""
    line, rejected = key_check_line(
        "OpenAI", _failed("openai", "request_failed", category), public_listing=False
    )
    assert (line, rejected) == (expected, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", [True, False], ids=["check-key", "auto-check"])
async def test_a_cloud_certificate_failure_names_check_key_whichever_check_ran(
    monkeypatch, requested
):
    """Review round 2, R2-03 + R2-04: the Check-key path said 'didn't
    answer' while the auto check named the certificate -- and the auto line
    said 'then Check again', a control a cloud row never shows."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "request_failed", "tls_untrusted")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "a" * 40)
        if requested:
            await pilot.click("#setup-provider-check-key")
        else:
            step._begin_selected_provider_discovery(step._effective_provider_draft())
        await wait_until(pilot, lambda: "certificate" in static_text(step, _STATUS))
        line = static_text(step, _STATUS)
        assert "Settings ▸ Network" in line and line.endswith("then press Check key.")
        assert "didn't answer" not in line and "Check again" not in line
        assert not step.query_one("#setup-provider-no-server").display  # Check again's block
        assert step.query_one("#setup-provider-check-key").display


@pytest.mark.asyncio
async def test_test_connection_then_next_keeps_the_certificate_cause_on_model(
    monkeypatch,
):
    """Review round 1, live F1: after Test connection, Model read the probe
    evidence's 'tls_untrusted' as 'tls untrusted' -- never the 'untrusted
    certificate' its copy keys on -- and showed the generic 'Couldn't reach
    the server (tls untrusted)' with no Settings ▸ Network pointer."""
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("custom", "request_failed", "tls_untrusted")
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        step._probe = AsyncMock(
            return_value=SettingsEndpointProbeOutcome(
                state="unreachable",
                category="tls_untrusted",
                summary="unreachable: certificate not trusted",
            )
        )
        await _custom_at(pilot, step, "https://127.0.0.1:8443")
        step._launch_probe()
        await wait_until(pilot, lambda: "certificate" in static_text(step, _STATUS))
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _c, model = current_step(wizard)
        await wait_until(pilot, lambda: bool(model.query("#setup-model-connection-failed")))
        # TASK-33008.5: the line is a Static, not a radio option.
        row = static_text(model, "#setup-model-connection-failed")
        assert "certificate isn't trusted" in row and "Settings ▸ Network" in row


@pytest.mark.asyncio
async def test_a_failure_model_finds_itself_flips_provider_at_once(monkeypatch):
    """Review round 1, live F4: when Model's own wait settled the failure
    (Provider's check timed out with no outcome), Provider kept ✓ beside
    Model's 'isn't reachable' -- the '!' flip covered only failures Provider
    handed off."""
    from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery

    monkeypatch.setattr(model_discovery, "MODEL_DISCOVERY_TIMEOUT_SECONDS", 0.5)
    never = asyncio.Event()

    async def hang(**_kwargs):
        await never.wait()

    wizard, _calls = _wizard(monkeypatch, lambda kw: None)
    wizard.app_instance.llm_provider_catalog_scope_service = SimpleNamespace(
        discover_models=hang
    )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "ollama")
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _c, model = current_step(wizard)
        await wait_until(
            pilot, lambda: bool(model.query("#setup-model-connection-failed")), tries=250
        )
        await pilot.pause(0.2)
        assert tracker_states(wizard)[STEP_PROVIDER] == "attention"


def _http_failed(provider: str, code: int) -> ModelDiscoveryResult:
    return ModelDiscoveryResult(
        provider=provider,
        provider_list_key=provider,
        endpoint_fingerprint=None,
        status="error",
        error=ModelDiscoveryError(
            kind="request_failed",
            message=f"The models endpoint answered HTTP {code}.",
            recovery_hint="y",
            category="http_status",
        ),
    )



@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            _failed("openai", "missing_credentials", "forbidden"),
            "! OpenAI answered, but won't list models for this key (HTTP 403) — it's "
            "checked when you send your first message.",
        ),
        (
            _http_failed("openai", 503),
            "! OpenAI answered with an error (HTTP 503), so the key couldn't be "
            "checked — it's checked when you send your first message.",
        ),
        (
            _listed("openai"),
            "! OpenAI answered with an empty model list, so the key couldn't be "
            "checked — it's checked when you send your first message.",
        ),
    ],
    ids=["403", "503", "empty"],
)
def test_a_provider_that_answered_is_never_called_uncheckable(result, expected):
    """Review round 1, F6: a 403, a 5xx and an empty listing all fell through
    to "can't be checked without a paid request" -- false: the provider
    answered."""
    line, rejected = key_check_line("OpenAI", result, public_listing=False)
    assert line == expected and not rejected
    assert _PAID not in line


@pytest.mark.parametrize("code", [404, 405])
def test_a_listing_route_that_is_not_there_is_the_paid_request_line(code):
    """Live F7: Cloudflare Workers AI answers GET .../ai/v1/models with 405 --
    its key can't be checked without a paid request, and the line says so."""
    result = (
        _failed("cloudflare", "unsupported_endpoint", "http_status")
        if code == 404
        else _http_failed("cloudflare", code)
    )
    line, _rejected = key_check_line("Cloudflare Workers AI", result, public_listing=False)
    assert line.startswith(f"! Cloudflare Workers AI {_PAID}")


@pytest.mark.parametrize("code", [405])
def test_an_answered_http_error_is_listing_unavailable_on_model_not_unreachable(code):
    """Live F7: a 405 from a cloud API flipped Provider to '!' and Back showed
    "The last connection check couldn't reach the server. Check it's running"
    -- for a server that answered. A listing route that isn't there (404,
    405) is a listing that isn't available. Review round 2 (R2-02) narrowed
    this from every HTTP answer: a 5xx or 429 is a failure (next test)."""
    from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
    from tldw_chatbook.UI.Wizards import first_run_setup_state as state

    _models, ui_state, category = model_discovery._model_discovery_ui_outcome(
        _http_failed("cloudflare", code)
    )
    assert ui_state == "listing_unavailable"
    assert state.classify_discovery_failure(ui_state, category) == ""


@pytest.mark.parametrize("code", [429, 500, 503])
def test_a_server_error_or_rate_limit_is_a_failure_that_names_its_status(code):
    """Review round 2, R2-02: round 1 mapped every HTTP answer to 'listing
    unavailable', so a 5xx or 429 kept ✓ on Provider, hid Model's Retry and
    skipped its one 'Continue anyway?' gate -- while Provider's own line
    said the check failed. llama-server answers 503 on every route while its
    model loads. The server answered, so no copy says it couldn't be
    reached; every copy names the status."""
    from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
    from tldw_chatbook.UI.Wizards import first_run_setup_state as state

    _models, ui_state, category = model_discovery._model_discovery_ui_outcome(
        _http_failed("llama_cpp", code)
    )
    assert (ui_state, category) == ("connection_failed", f"HTTP {code}")
    failure = state.classify_discovery_failure(ui_state, category)
    assert failure == state.PROVIDER_PROBE_SERVER_ERROR
    row = model_discovery.connection_failed_text(
        "llama_cpp", category, "http://127.0.0.1:9099/v1"
    )
    assert row.startswith(f"http://127.0.0.1:9099 answered HTTP {code}")
    assert row.endswith("Retry, or enter a model ID below.")
    for copy in (
        row,
        state.probe_failure_summary_detail(failure),
        state.probe_failure_confirm_text(failure),
        state.probe_failure_return_notice(failure),
    ):
        assert copy and "reach" not in copy and "running" not in copy, copy
    loading = model_discovery.connection_failed_text("llama_cpp", "HTTP 503", "")
    assert "it may still be loading" in loading


@pytest.mark.parametrize(
    ("transport", "named"),
    [("tls_plain_http", "http://"), ("tls_untrusted", "Settings ▸ Network")],
)
def test_a_server_that_answered_over_the_wrong_transport_is_never_unreachable(
    transport, named
):
    """Found live (g6-r2fix/41): llama.cpp at https:// against its plain-HTTP
    port -- Model said it speaks plain HTTP and Provider's own line did too,
    but Back pinned "The last connection check couldn't reach the server.
    Check it's running" above them. A server that answered with plain HTTP
    or an untrusted certificate keeps '!' and Model's one gate, and every
    copy names the fix."""
    from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
    from tldw_chatbook.UI.Wizards import first_run_setup_state as state

    _models, ui_state, category = model_discovery._model_discovery_ui_outcome(
        _failed("llama_cpp", "request_failed", transport)
    )
    failure = state.classify_discovery_failure(ui_state, category)
    assert failure, "a failure: '!' on Provider and Model's one gate"
    for copy in (
        state.probe_failure_summary_detail(failure),
        state.probe_failure_confirm_text(failure),
        state.probe_failure_return_notice(failure),
    ):
        assert copy and "reach" not in copy and "running" not in copy, copy
    assert named in state.probe_failure_return_notice(failure)


def test_a_tested_http_error_names_its_status_on_model():
    """Review round 2, R2-02: Test connection then Next -- Model read the
    evidence's 'http status' as a connection failure ('Couldn't reach the
    server (http status)')."""
    from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery

    evidence = SimpleNamespace(category="http_status")
    owner = SimpleNamespace(_last_probe_status_code=503)
    assert model_discovery.evidence_failure_category(evidence, owner) == "HTTP 503"
    assert model_discovery.evidence_failure_category(evidence) == "HTTP error"
    row = model_discovery.connection_failed_text("custom", "HTTP error", "")
    assert row == "The server answered with an HTTP error. Retry, or enter a model ID below."


@pytest.mark.asyncio
async def test_a_loading_server_flips_provider_and_keeps_models_retry_and_gate(monkeypatch):
    """Review round 2, R2-02, end to end: llama.cpp (here Ollama's row)
    answering 503 while its model loads. Provider names the status and the
    control that retries it; Model keeps Retry and its one gate; the tracker
    shows '!' on Provider."""
    wizard, _calls = _wizard(monkeypatch, lambda kw: _http_failed("ollama", 503))
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "ollama")
        await wait_until(pilot, lambda: "HTTP 503" in static_text(step, _STATUS))
        line = static_text(step, _STATUS)
        assert line == (
            "! Ollama answered HTTP 503 — it may still be loading or busy; "
            "wait, then press Check again."
        )
        assert step.query_one("#setup-provider-no-server").display  # Check again's block
        await pilot.press("ctrl+n")
        await wait_for_step(pilot, wizard, STEP_MODEL)
        _c, model = current_step(wizard)
        await wait_until(
            pilot, lambda: bool(model.query("#setup-model-connection-failed")), tries=250
        )
        await pilot.pause(0.2)
        # TASK-33008.5: the line is a Static, not a radio option.
        row = static_text(model, "#setup-model-connection-failed")
        assert "answered HTTP 503" in row and "reach" not in row
        assert not model.query_one("#setup-model-retry").has_class("hidden")
        assert tracker_states(wizard)[STEP_PROVIDER] == "attention"
        assert "answered with an error" in (model.confirm_before_advance() or "")


@pytest.mark.asyncio
@pytest.mark.local_server_probe
async def test_discovery_carries_the_status_code_it_was_answered_with():
    import httpx

    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        discover_openai_compatible_models,
    )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(405))
    ) as client:
        result = await discover_openai_compatible_models(
            provider="openai",
            provider_list_key="OpenAI",
            endpoint="https://api.example.test/v1",
            api_key="sk-proj-" + "a" * 40,
            client=client,
        )
    assert result.error is not None and result.error.category == "http_status"
    assert "HTTP 405" in result.error.message


@pytest.mark.asyncio
async def test_a_401_with_no_key_sent_says_the_server_needs_one():
    """Review round 1, live F13: a local llama-server started with --api-key
    answered 401 to a keyless Test connection, and the step said '✗
    Unauthorized - the server rejected this API key' -- no key was sent."""
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    wizard = resumed_on_provider()
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        step._probe = AsyncMock(
            return_value=SettingsEndpointProbeOutcome(
                state="unreachable", category="unauthorized", summary="unreachable: unauthorized"
            )
        )
        await _custom_at(pilot, step, "http://127.0.0.1:9311")
        step._launch_probe()
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        # Told as dev's TASK-33008 tells a keyless 401 (merged 2026-10-19):
        # nothing was rejected, and the section it names is no longer optional.
        line = static_text(step, _STATUS)
        assert "needs an API key and none was set. Add one under Authentication" in line
        assert "rejected" not in line
        assert str(step.query_one("#setup-provider-auth-toggle").title) == "Authentication"


_HOSTED_CASES = {
    "minimax": "",
    "together": "",
    "zai": "",
    "moonshot": "",
    "cloudflare": "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1",
    "databricks": "https://dbc-0123.cloud.databricks.com/serving-endpoints",
}


@pytest.mark.parametrize("provider", sorted(_HOSTED_CASES))
def test_a_hosted_presets_check_reaches_the_provider_with_its_base_url(provider):
    """Review round 1, live F7's real cause (instrumented live, g6-r1fix/19):
    for a hosted preset the wizard staged the chat URL as ``api_base_url``,
    and the hosted resolver refused it ('… base URL must not include a
    terminal /chat/completions endpoint path') before any request: Check key
    said "can't be checked without a paid request" for a real MiniMax key,
    and Model read "Couldn't reach the server (request failed). Check it's
    running" though nothing was asked. The staged base is what Next saves."""
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.UI.Wizards import first_run_setup_state as state
    from tldw_chatbook.UI.Wizards.first_run_model_discovery import (
        _first_run_discovery_staged_settings,
    )

    config = load_settings()
    draft = state.resolve_first_run_provider_draft(
        state.FirstRunProviderDraft(
            provider=provider,
            endpoint=_HOSTED_CASES[provider],
            credential=state.ProviderCredentialDraft("draft", "test-key-" + "a" * 32),
        ),
        config,
    )
    staged = _first_run_discovery_staged_settings(
        draft, state.build_first_run_model_discovery_key(draft)
    )
    service = LocalLLMProviderCatalogService()
    base_url, _key = service._resolve_hosted_provider(
        provider_key=provider, saved_settings=config, staged_settings=staged
    )
    assert not base_url.rstrip("/").endswith("/chat/completions"), base_url


@pytest.mark.asyncio
@pytest.mark.parametrize("backing", ["environment", "stored"])
async def test_a_rejected_typed_key_backed_by_another_names_fixes_that_work(
    monkeypatch, backing
):
    """Review round 2, R2-05: a typed key rejected while an exported (after
    'Store a different key') or saved (after Replace) key backs the provider
    was refused with '…clear it (Ctrl+U) and press Next to skip' -- but the
    empty field falls back to the other key, so Next advanced with it."""
    def only_the_typed_key_fails(kw):
        settings = kw["staged_settings"]["api_settings"]["openai"]
        if settings.get("credential_source") == "draft":
            return _failed("openai", "missing_credentials", "unauthorized")
        return _listed("openai", "gpt-a")

    if backing == "environment":
        wizard, _calls = _wizard(monkeypatch, only_the_typed_key_fails)
        monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    else:
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(name, raising=False)

        async def no_scan(_config, **_kwargs):
            return ()

        monkeypatch.setattr(local_server_discovery, "discover_local_servers", no_scan)
        wizard = resumed_on_provider(
            lambda config: config.setdefault("api_settings", {})
            .setdefault("openai", {})
            .update(api_key="sk-proj-" + "s" * 40)
        )
        scope, _calls = _scope(only_the_typed_key_fails)
        wizard.app_instance.llm_provider_catalog_scope_service = scope
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        if backing == "environment":
            await pilot.click("#setup-provider-key-source-store")
        else:
            await pilot.click("#setup-provider-key-replace")
        await pilot.pause()
        await _type_key(pilot, step, "sk-proj-" + "w" * 40)
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: bool(pinned_error(wizard)))
        refusal = pinned_error(wizard)
        assert "Ctrl+U" not in refusal and "to skip" not in refusal, refusal
        fix = (
            "Use OPENAI_API_KEY from your environment"
            if backing == "environment"
            else "Keep current"
        )
        assert fix in refusal and "Skip — connect later" in refusal


@pytest.mark.asyncio
@pytest.mark.parametrize("backing", ["environment", "stored"])
async def test_a_typed_key_that_cannot_be_sent_is_refused_never_swapped(
    monkeypatch, backing
):
    """Review round 2, R2-06: a typed key with a smart dash under 'Store a
    different key' (or Replace) was silently unusable, so the credential fell
    back to the exported (or saved) key and Next advanced with it -- the
    field note warned, but nothing refused, and note_key_entered() ran."""
    if backing == "environment":
        wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-a"))
        monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    else:
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(name, raising=False)

        async def no_scan(_config, **_kwargs):
            return ()

        monkeypatch.setattr(local_server_discovery, "discover_local_servers", no_scan)
        wizard = resumed_on_provider(
            lambda config: config.setdefault("api_settings", {})
            .setdefault("openai", {})
            .update(api_key="sk-proj-" + "s" * 40)
        )
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        if backing == "environment":
            await pilot.click("#setup-provider-key-source-store")
        else:
            await pilot.click("#setup-provider-key-replace")
        await pilot.pause()
        await _type_key(pilot, step, "sk-proj-abc–" + "d" * 40)
        await pilot.press("ctrl+n")
        await pilot.pause(0.3)
        _c, current = current_step(wizard)
        assert current is step, "Next advanced with the other key"
        assert "API keys never use" in pinned_error(wizard)
        assert wizard.query_one("SetupWizardContainer").staged_provider_draft is None



@pytest.mark.asyncio
async def test_a_test_connection_result_is_brought_on_screen_at_120x40(monkeypatch):
    """Review round 2, G6-VR2-06: for a local row the Test result rendered
    below the fold at 120x40 (behind three blank rows), so a click on Test
    connection looked dead until the user scrolled."""
    from unittest.mock import AsyncMock

    from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
        SettingsEndpointProbeOutcome,
    )

    wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("custom", "m"))
    app = WizardHost(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        step._probe = AsyncMock(
            return_value=SettingsEndpointProbeOutcome(
                state="reachable", summary="reachable", model_ids=("m1", "m2")
            )
        )
        await _custom_at(pilot, step, "http://127.0.0.1:9099")
        test = step.query_one("#setup-provider-test", Button)
        actions = test.parent
        assert actions.region.height == 1, "the Test row holds two blank rows"
        test.scroll_visible(animate=False)
        await pilot.pause(0.2)
        await pilot.click("#setup-provider-test")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✓ Connected"))
        await pilot.pause(0.3)
        status = step.query_one(_STATUS).region
        assert status.height and step.region.y <= status.y < step.region.bottom, (
            f"the result {status} is outside the step viewport {step.region}"
        )


@pytest.mark.asyncio
async def test_a_per_account_url_field_has_no_dead_rows_below_it(monkeypatch):
    """Review round 2, G6-VR2-11 (3): 8-10 blank rows sat between Azure's
    resource URL field and Authentication -- an empty chat-URL line, an
    empty status line, and a 3-row button row whose buttons were hidden."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    wizard, _calls = _wizard(monkeypatch, lambda kw: None)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "azure")
        await pilot.pause(0.2)
        url = step.query_one("#setup-provider-endpoint", Input).region
        auth = step.query_one("#setup-provider-auth-toggle").region
        assert auth.y - url.bottom <= 3, (url, auth)


@pytest.mark.asyncio
async def test_a_check_key_result_is_brought_on_screen_at_80x24(monkeypatch):
    """Review round 2, G6-VR2-06: Check key's answer is written to the step's
    status line, above the Authentication section; with the key field
    scrolled into view at 80x24 the answer sat off screen."""
    wizard, calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-a", "gpt-b"))
    app = WizardHost(wizard)
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await _type_key(pilot, step, "sk-proj-" + "k" * 40)
        check = step.query_one("#setup-provider-check-key", Button)
        check.press()
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✓ Key works"))
        await pilot.pause(0.3)
        step.scroll_end(animate=False)  # the user scrolls back down to the key
        await pilot.pause(0.2)
        status = step.query_one(_STATUS)
        assert not (step.region.y <= status.region.y < step.region.bottom), (
            "precondition: the status line starts off screen"
        )
        checks = len(calls)
        check.press()  # and checks again
        await wait_until(pilot, lambda: len(calls) > checks)
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✓ Key works"))
        await pilot.pause(0.3)
        region = status.region
        assert region.height and step.region.y <= region.y < step.region.bottom, (
            f"the answer {region} is outside the step viewport {step.region}"
        )


@pytest.mark.asyncio
async def test_the_environment_source_hides_the_fields_own_label_and_show(monkeypatch):
    """Review round 2, G6-VR2-11 (1): with 'Use OPENAI_API_KEY' chosen the
    key field is hidden, but its 'API key' label and the Show toggle stayed
    (Show only flipped its own label)."""
    wizard, _calls = _wizard(monkeypatch, lambda kw: _listed("openai", "gpt-a"))
    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        label = step.query_one("#setup-provider-key-label")
        reveal = step.query_one("#setup-provider-key-reveal", Button)
        assert not key_input(step).display
        assert not label.display and not reveal.display
        # Live (g6-r2fix/52): the 3-row key row kept two blank rows under
        # Check key and Skip once its field was hidden.
        row = key_input(step).parent
        assert row.region.height == 1, row.region
        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause()
        assert key_input(step).display and label.display and reveal.display
        assert row.region.height == 3, row.region


@pytest.mark.asyncio
async def test_choosing_store_takes_down_the_refusal_that_named_it(monkeypatch):
    """Review round 2, G6-VR2-11 (2): after 'Store a different key for this
    app' was chosen, the pinned strip still told the user to choose it."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: "Store a different key" in pinned_error(wizard))
        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause()
        assert key_input(step).display
        assert pinned_error(wizard) == ""


@pytest.mark.asyncio
async def test_the_key_placeholder_fits_its_field_at_80x24(monkeypatch):
    """Review round 2, G6-VR2-11 (3): at 80x24 the placeholder read
    'Paste your AP' -- the field beside Show, Check key and Skip has 13
    columns."""
    wizard, _calls = _wizard(monkeypatch, lambda kw: None)
    app = WizardHost(wizard)
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await pilot.pause(0.2)
        field = key_input(step)
        assert len(field.placeholder) <= field.content_region.width, (
            field.placeholder,
            field.content_region,
        )


@pytest.mark.asyncio
async def test_store_with_an_empty_field_never_advances_on_the_rejected_key(monkeypatch):
    """Found live (g6-r2fix/55-56): after a 401 on OPENAI_API_KEY, choosing
    'Store a different key for this app' and pressing Next with the field
    still empty advanced -- on the exported key OpenAI had just rejected --
    to Model's 'Authentication failed'. Store bumps the credential revision,
    so the rejection (kept per exact discovery key) no longer matched."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause()
        assert key_input(step).display and key_input(step).value == ""
        assert step.query_one("#setup-provider-skip", Button).display
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: bool(pinned_error(wizard)))
        _c, current = current_step(wizard)
        assert current is step, "Next advanced on the rejected exported key"
        refusal = pinned_error(wizard)
        assert "paste a working key above" in refusal
        assert "choose 'Store a different key" not in refusal, "it is chosen"


@pytest.mark.asyncio
async def test_a_refusal_never_offers_a_key_already_rejected(monkeypatch):
    """Found live (g6-r2fix/61): OPENAI_API_KEY rejected, 'Store a different
    key', a typed key rejected too -- and the refusal offered 'Use
    OPENAI_API_KEY from your environment', the key OpenAI had just turned
    down (review cross-cutting-07: only actions that can succeed are named)."""
    wizard, _calls = _wizard(
        monkeypatch, lambda kw: _failed("openai", "missing_credentials", "unauthorized")
    )
    monkeypatch.setenv("OPENAI_API_KEY", _EXPORTED_OPENAI_KEY)
    app = WizardHost(wizard)
    async with app.run_test(size=SIZE) as pilot:
        await wait_for_step(pilot, wizard, STEP_PROVIDER)
        _c, step = current_step(wizard)
        await pick_provider(pilot, step, "openai")
        await wait_until(pilot, lambda: static_text(step, _STATUS).startswith("✗"))
        await pilot.click("#setup-provider-key-source-store")
        await pilot.pause()
        await _type_key(pilot, step, "sk-proj-" + "w" * 40)
        await pilot.click("#setup-provider-check-key")
        await wait_until(pilot, lambda: static_text(step, _STATUS) == "✗ OpenAI rejected this key.")
        await pilot.press("ctrl+n")
        await wait_until(pilot, lambda: bool(pinned_error(wizard)))
        refusal = pinned_error(wizard)
        assert "OPENAI_API_KEY" not in refusal, refusal
        assert "paste a working key above" in refusal and "Skip" in refusal
