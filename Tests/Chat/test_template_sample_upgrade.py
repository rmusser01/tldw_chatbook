"""TASK-34100.2 review round 1 (F4): an upgrade never turns a user's table into a sample.

The template's ``template_sample = true`` marker was deep-merged into every
existing profile at load time, so a TabbyAPI, Custom or vLLM section the user
configured BEFORE the marker existed (hand-edited, or saved by an older
Settings or setup build) read as a shipped placeholder and was hidden under
the switcher's NOT RUNNING group. The marker is decided from the user's own
file instead: a table that exists there without the key predates the marker,
so it is the user's (``false``); a table the file never had keeps the
template's ``true``. Nothing is written.

Pass 3 (2026-10-06): that rule also took the marker off every sample table an
OLD profile carries -- the app wrote the whole template into config.toml on
first run, so a profile created before the marker holds TabbyAPI's
``tabby-model`` and vLLM's placeholder endpoint without the key, and they read
'Ready · not tested' again. A pre-marker table that still holds the shipped
sample endpoint and model (identical in every template revision since those
sections were added) stays a sample; one whose endpoint or model differs is the
user's. Only pre-marker tables are compared; a table with the key is decided by
the key alone.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_session_settings import provider_is_template_sample

#: The template's placeholder local sections (each marked ``template_sample``).
SAMPLE_SECTIONS = (
    "vllm",
    "aphrodite",
    "tabbyapi",
    "custom",
    "custom_2",
    "local_llamacpp",
    "local_vllm",
    "local_ollama",
    "local_mlx_lm",
)


def _pre_marker_template() -> str:
    """The shipped template as a build from before the marker wrote it: no
    ``template_sample`` lines, and Custom / Custom 2 still streaming off."""
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    body = "\n".join(
        line
        for line in CONFIG_TOML_CONTENT.splitlines()
        if not line.strip().startswith("template_sample")
    )
    return body.replace(
        "streaming = true # TASK-34100.2: a local OpenAI-compatible server streams",
        "streaming = false",
    ) + "\n"

PRE_MARKER_PROFILE = """
[api_settings.tabbyapi]
model = "my-real-tabby-model"

[api_settings.custom]
model = "lmstudio-model"
api_url = "http://127.0.0.1:1234/v1/chat/completions"
"""


def _load_settings_from(monkeypatch, body: str) -> dict:
    from tldw_chatbook import config as config_module

    config_path = Path(config_module.get_cli_config_path())
    config_path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    return config_module.load_settings(force_reload=True)


@private_profile_test
def test_a_table_saved_before_the_marker_existed_is_the_users(
    request: pytest.FixtureRequest, monkeypatch
) -> None:
    settings = _load_settings_from(monkeypatch, PRE_MARKER_PROFILE)

    assert settings["api_settings"]["tabbyapi"]["model"] == "my-real-tabby-model"
    assert provider_is_template_sample(settings, "tabbyapi") is False
    assert provider_is_template_sample(settings, "custom") is False
    # A placeholder section the user's file never had is still the template's.
    assert provider_is_template_sample(settings, "vllm") is True


@private_profile_test
def test_a_fresh_profile_keeps_the_template_markers(
    request: pytest.FixtureRequest, monkeypatch
) -> None:
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    settings = _load_settings_from(monkeypatch, CONFIG_TOML_CONTENT)

    assert provider_is_template_sample(settings, "tabbyapi") is True
    assert provider_is_template_sample(settings, "custom") is True


@private_profile_test
def test_an_explicit_marker_in_the_users_file_is_kept(
    request: pytest.FixtureRequest, monkeypatch
) -> None:
    settings = _load_settings_from(
        monkeypatch,
        "[api_settings.tabbyapi]\ntemplate_sample = true\nmodel = \"tabby-model\"\n\n"
        "[api_settings.vllm]\ntemplate_sample = false\nmodel = \"served\"\n",
    )

    assert provider_is_template_sample(settings, "tabbyapi") is True
    assert provider_is_template_sample(settings, "vllm") is False


@private_profile_test
def test_an_old_profile_written_from_the_pre_marker_template_keeps_its_samples(
    request: pytest.FixtureRequest, monkeypatch
) -> None:
    from tldw_chatbook import config as config_module

    body = _pre_marker_template()
    assert "template_sample" not in body
    settings = _load_settings_from(monkeypatch, body)

    for section in SAMPLE_SECTIONS:
        assert settings["api_settings"][section]["template_sample"] is True, section
    assert provider_is_template_sample(settings, "tabbyapi") is True
    # Read-side only: the user's own values stay, and the file gains no marker.
    assert settings["api_settings"]["custom"]["streaming"] is False
    on_disk = Path(config_module.get_cli_config_path()).read_text(encoding="utf-8")
    assert "template_sample" not in on_disk


@pytest.mark.parametrize(
    "table",
    (
        {"api_url": "http://127.0.0.1:5000/v1/chat/completions", "model": "tabby-model"},
        {"api_url": "http://localhost:8080/v1/chat/completions", "model": "exl2-served"},
    ),
    ids=("endpoint-changed", "model-changed"),
)
def test_a_pre_marker_table_whose_endpoint_or_model_changed_is_the_users(
    table: dict,
) -> None:
    from tldw_chatbook import config as config_module

    merged = config_module._atomic_config_values_from_raw(
        {"api_settings": {"tabbyapi": table}}
    )

    assert merged["api_settings"]["tabbyapi"]["template_sample"] is False


def test_a_pre_marker_sample_table_whose_other_keys_differ_is_still_a_sample() -> None:
    """Only the sample's endpoint and model identify it: a drifted
    ``streaming`` (Custom shipped it off) or a tuned temperature does not make
    the placeholder endpoint the user's."""
    from tldw_chatbook import config as config_module
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    shipped = DEFAULT_CONFIG_FROM_TOML["api_settings"]["custom"]
    merged = config_module._atomic_config_values_from_raw(
        {
            "api_settings": {
                "custom": {
                    "api_url": shipped["api_url"],
                    "model": shipped["model"],
                    "streaming": False,
                    "temperature": 0.2,
                }
            }
        }
    )

    assert merged["api_settings"]["custom"]["template_sample"] is True
    assert merged["api_settings"]["custom"]["streaming"] is False


def test_the_write_path_snapshot_reads_the_same_rule() -> None:
    """The atomic snapshot (what a save validates against) merges the raw
    file the same way the loader does."""
    from tldw_chatbook import config as config_module

    merged = config_module._atomic_config_values_from_raw(
        {"api_settings": {"tabbyapi": {"model": "my-real-tabby-model"}}}
    )

    assert merged["api_settings"]["tabbyapi"]["template_sample"] is False
    assert merged["api_settings"]["vllm"]["template_sample"] is True


@pytest.mark.parametrize(
    ("section", "table"),
    (
        ("vllm", {"api_base_url": "http://gpu-box.lan:8000/v1"}),
        ("custom", {"base_url": "http://10.0.0.5:1234/v1"}),
        ("tabbyapi", {"api_base": "http://127.0.0.1:5000/v1"}),
    ),
    ids=("vllm-api_base_url", "custom-base_url", "tabbyapi-api_base"),
)
def test_a_pre_marker_table_sent_to_its_own_base_url_is_the_users(
    section: str, table: dict
) -> None:
    """Review round 1 (R1-2): the Console sends to ``api_base_url``, then
    ``api_base`` and ``base_url``, before ``api_url``. A table that points at
    the user's own server through one of those keys, with the template's
    ``api_url`` left behind, is the user's -- including vLLM, whose sample
    model ``""`` is also what a real vLLM user writes."""
    from tldw_chatbook import config as config_module

    merged = config_module._atomic_config_values_from_raw(
        {"api_settings": {section: table}}
    )

    assert merged["api_settings"][section]["template_sample"] is False


def test_a_pre_marker_table_whose_base_url_is_blank_still_reads_its_api_url() -> None:
    """A blank alias is no endpoint (the Console skips it too), so the table
    still points at the sample's ``api_url``."""
    from tldw_chatbook import config as config_module

    merged = config_module._atomic_config_values_from_raw(
        {"api_settings": {"tabbyapi": {"api_base_url": "  ", "model": "tabby-model"}}}
    )

    assert merged["api_settings"]["tabbyapi"]["template_sample"] is True


#: The endpoint and model every template revision shipped for each sample
#: section before the marker existed (all 484 revisions of config.py on dev,
#: re-scanned in review round 1). A literal copy, so an edit to the frozen
#: constant -- for example "syncing" it with a template change -- fails here.
HISTORICAL_SAMPLES = {
    "vllm": ("http://localhost:8000/v1/chat/completions", ""),
    "aphrodite": ("http://localhost:2242/v1/chat/completions", "aphrodite-engine"),
    "tabbyapi": ("http://localhost:8080/v1/chat/completions", "tabby-model"),
    "custom": ("http://localhost:1234/v1/chat/completions", "custom-model-alpha"),
    "custom_2": ("http://localhost:5678/v1/chat/completions", "custom-model-gamma"),
    "local_llamacpp": ("http://localhost:8001/v1/chat/completions", "custom-model-gamma"),
    "local_vllm": ("http://localhost:8008/v1/chat/completions", "custom-model-gamma"),
    "local_ollama": ("http://localhost:5678/v1/chat/completions", "custom-model-gamma"),
    "local_mlx_lm": ("http://localhost:5678/v1/chat/completions", "custom-model-gamma"),
}


def test_the_pre_marker_identities_are_the_historical_template_values() -> None:
    """Review round 1 (R1-3): old profiles are recognised by the values the
    old templates shipped, frozen in code, not by today's template."""
    from tldw_chatbook.config import PRE_MARKER_TEMPLATE_SAMPLES

    assert {
        section: (table["api_url"], table["model"])
        for section, table in PRE_MARKER_TEMPLATE_SAMPLES.items()
    } == HISTORICAL_SAMPLES
    assert set(PRE_MARKER_TEMPLATE_SAMPLES) == set(SAMPLE_SECTIONS)


def test_a_template_edit_never_turns_an_old_profiles_sample_into_the_users(
    monkeypatch,
) -> None:
    """Review round 1 (R1-3): comparing old profiles with the LIVE template
    meant a future template edit (a new port, a renamed placeholder model)
    flipped every pre-marker sample to 'the user's', back to 'Ready · not
    tested'. Simulate that edit: a profile written before the marker still
    holds the historical values and stays a sample."""
    from tldw_chatbook import config as config_module

    edited = deepcopy(config_module.DEFAULT_CONFIG_FROM_TOML)
    edited["api_settings"]["tabbyapi"]["api_url"] = (
        "http://localhost:5000/v1/chat/completions"
    )
    edited["api_settings"]["tabbyapi"]["model"] = "tabby-model-renamed"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_FROM_TOML", edited)
    url, model = HISTORICAL_SAMPLES["tabbyapi"]

    settled = config_module.settle_pre_marker_template_samples(
        {"api_settings": {"tabbyapi": {"api_url": url, "model": model}}}
    )

    assert "template_sample" not in settled["api_settings"]["tabbyapi"]
    assert (
        config_module.deep_merge_dicts(edited, dict(settled))["api_settings"][
            "tabbyapi"
        ]["template_sample"]
        is True
    )


def test_the_sample_endpoint_keys_are_the_ones_a_send_reads() -> None:
    """config.py cannot import the Chat package, so it spells the endpoint
    precedence itself; it must stay the Console's (review round 1, R1-2)."""
    from tldw_chatbook.Chat import console_provider_endpoints
    from tldw_chatbook.config import TEMPLATE_SAMPLE_ENDPOINT_KEYS

    assert (
        TEMPLATE_SAMPLE_ENDPOINT_KEYS
        == console_provider_endpoints._ENDPOINT_SETTING_KEYS
    )


# --- Review v5r2 G2C-R3-3: MLX-LM is pointed at a server by host and port ---

#: Edits to the shipped MLX-LM table, and whether they make it the user's.
#: ``chat_with_mlx_lm`` builds its URL from ``host``/``port`` (defaults
#: 127.0.0.1 and 8080) and the Console forwards it no base URL, so ``api_url``
#: never routes a send: editing it alone leaves the endpoint a send uses (and
#: so the sample) unchanged.
MLX_EDITS = (
    ({"port": 5679}, True),
    ({"host": "10.0.0.5"}, True),
    ({"api_url": "http://127.0.0.1:5679/v1/chat/completions"}, False),
    ({"port": 8080}, False),
)
MLX_EDIT_IDS = ("port", "host", "api_url-only", "port-at-the-handler-default")


@pytest.mark.parametrize(("edit", "users"), MLX_EDITS, ids=MLX_EDIT_IDS)
def test_mlx_lm_is_the_users_once_the_host_or_port_its_send_uses_differs(
    edit: dict, users: bool
) -> None:
    """Values decide by the endpoint a send would use. For MLX-LM that is
    the host/port pair: a user who points it at their own server with
    ``port`` and keeps the shipped model was hidden under NOT RUNNING as
    'MLX-LM (Apple silicon) (not checked)', while an ``api_url`` edit, which
    the send ignores, read as theirs. Marked (the switcher) and pre-marker
    (the loader) tables read the same rule."""
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.console_session_settings import (
        provider_holds_template_sample_values,
    )
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    marked = {**deepcopy(DEFAULT_CONFIG_FROM_TOML["api_settings"]["local_mlx_lm"]), **edit}
    assert marked["template_sample"] is True  # a hand edit, never a Save
    pre_marker = {key: value for key, value in marked.items() if key != "template_sample"}

    holds = provider_holds_template_sample_values(
        {"api_settings": {"local_mlx_lm": marked}}, "local_mlx_lm"
    )
    settled = config_module._atomic_config_values_from_raw(
        {"api_settings": {"local_mlx_lm": pre_marker}}
    )

    assert holds is (not users)
    assert settled["api_settings"]["local_mlx_lm"]["template_sample"] is (not users)


@pytest.mark.parametrize(("edit", "_users"), MLX_EDITS, ids=MLX_EDIT_IDS)
def test_mlx_lm_sample_identity_is_the_url_its_send_posts_to(
    monkeypatch, edit: dict, _users: bool
) -> None:
    """The identity mirrors ``chat_with_mlx_lm``'s own URL, so the two
    cannot drift apart: the handler is called with the same table (no
    network; its downstream request is captured) and the base URL it would
    post to is the endpoint the sample identity compares."""
    from types import SimpleNamespace

    from tldw_chatbook import config as config_module
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls

    table = {**deepcopy(config_module.DEFAULT_CONFIG_FROM_TOML["api_settings"]["local_mlx_lm"]), **edit}
    sent: dict = {}
    monkeypatch.setattr(
        local_calls,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={"api_settings": {"local_mlx_lm": table}}),
    )
    monkeypatch.setattr(
        local_calls,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: sent.update(kwargs) or {"choices": []},
    )

    local_calls.chat_with_mlx_lm(
        [{"role": "user", "content": "hi"}], provider_name="local_mlx_lm"
    )

    endpoint, model = config_module._template_sample_identity(
        table, section="local_mlx_lm"
    )
    assert endpoint == sent["api_base_url"]
    assert model == sent["model_name"] == table["model"]
