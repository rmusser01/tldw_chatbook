"""TASK-18803: Moonshot and Z.ai per-model request-capability predicates.

The chat request builders used to gate ``reasoning_effort`` on exact model-id
literals (``kimi-k3``, ``glm-5.2``) and Moonshot sampling params on a frozen
``moonshot-v1-`` prefix -- the same staleness mechanism TASK-18414/18802
removed for Anthropic and OpenAI. These tests pin the replacement predicates'
family boundaries.

Moonshot rows are probe-verified against api.moonshot.ai (2026-08-20, real
key -- chatcmpl ids in the predicate docstrings/comments in
``model_capabilities.py``). No Z.ai key exists in this repo, so the GLM rows
pin the conservative version-floor liberalisation of the old exact-id pin,
not wire behavior.
"""

import pytest

from tldw_chatbook.model_capabilities import (
    ModelCapabilities,
    moonshot_model_rejects_sampling_params,
    moonshot_model_requires_min_temperature_for_multiple_choices,
    moonshot_model_returns_reasoning_content,
    moonshot_model_supports_reasoning_effort,
    zai_model_supports_reasoning_effort,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "model",
    [
        "kimi-k3",
        "kimi-k3-turbo",  # release-day suffix: the old literal pin rejected it
        "kimi-k4",
        "kimi-k2.5",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "KIMI-K3",
        "moonshot/kimi-k3",
    ],
)
def test_moonshot_versioned_kimi_rejects_sampling_and_supports_effort(model):
    assert moonshot_model_rejects_sampling_params(model) is True
    assert moonshot_model_supports_reasoning_effort(model) is True


@pytest.mark.parametrize("model", ["kimi-latest", "kimi", "kimi-thinking-preview"])
def test_moonshot_unversioned_kimi_supports_effort_but_accepts_sampling(model):
    # kimi-latest + the full sampling set answered 200 on the wire
    # (chatcmpl-6a872b9816ceb0c0ae780b1e); it must not be treated as a
    # sampling-rejecting reasoning id.
    assert moonshot_model_supports_reasoning_effort(model) is True
    assert moonshot_model_rejects_sampling_params(model) is False


@pytest.mark.parametrize(
    "model",
    [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
        "moonshot-v1-auto",
        "moonshot-v1-8k-vision-preview",
        "kimiko-7b",  # boundary: 'kimi' must not match inside a longer word
        "kimik3",
        "",
        None,
        42,
    ],
)
def test_moonshot_legacy_and_lookalikes_never_match_kimi_series(model):
    assert moonshot_model_supports_reasoning_effort(model) is False
    assert moonshot_model_rejects_sampling_params(model) is False


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("moonshot-v1-8k", True),
        ("moonshot-v1-32k", True),
        ("moonshot-v1-auto", True),
        ("moonshot-v1-8k-vision-preview", True),
        ("moonshot/moonshot-v1-8k", True),
        ("kimi-k3", False),
        ("kimi-latest", False),
        ("moonshot-v2-8k", False),
        ("moonshot-v10-8k", False),  # v1 must sit at a token boundary
        (None, False),
    ],
)
def test_moonshot_min_temperature_interplay_is_v1_family_only(model, expected):
    assert (
        moonshot_model_requires_min_temperature_for_multiple_choices(model)
        is expected
    )


@pytest.mark.parametrize(
    "model",
    [
        "glm-5.2",
        "glm-5.2-air",  # release-day suffix: the old literal pin rejected it
        "glm-5.3",
        "glm-5.10",
        "glm-6",
        "glm-6.1",
        "GLM-5.2",
        "zai/glm-5.2",
    ],
)
def test_zai_glm_at_or_above_floor_supports_reasoning_effort(model):
    assert zai_model_supports_reasoning_effort(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "glm-5",  # below the 5.2 floor: keeps the historical rejection
        "glm-5.1",
        "glm-4.6",
        "glm-4.5-air",
        "glm-5x",  # version must sit at a token boundary
        "glmnet-5.2",
        "",
        None,
        42,
    ],
)
def test_zai_below_floor_or_lookalikes_never_match(model):
    assert zai_model_supports_reasoning_effort(model) is False


@pytest.mark.parametrize(
    "model",
    [
        "kimi-k3",
        "kimi-k3-turbo",  # release-day suffix: the old literal pin missed it
        "kimi-k4",
        "kimi-k2.5",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "KIMI-K3",
        "moonshot/kimi-k3",
    ],
)
def test_moonshot_versioned_kimi_returns_reasoning_content(model):
    """TASK-19170 probes: every versioned kimi id answered with
    reasoning_content, with and without reasoning_effort (chatcmpl ids in the
    predicate docstring)."""
    assert moonshot_model_returns_reasoning_content(model) is True


@pytest.mark.parametrize(
    "model",
    [
        # kimi-latest accepts reasoning_effort on the wire (TASK-18803) but
        # returned NO reasoning_content (chatcmpl-6a8768a616ceb0c0ae780f2c) --
        # the response-side family is narrower than the request-side one.
        "kimi-latest",
        "kimi",
        "kimi-thinking-preview",
        "moonshot-v1-8k",
        "moonshot-v1-auto",
        "kimiko-7b",
        "kimik3",
        "",
        None,
        42,
    ],
)
def test_moonshot_unversioned_and_lookalikes_do_not_return_reasoning_content(model):
    assert moonshot_model_returns_reasoning_content(model) is False


def test_moonshot_response_side_family_is_narrower_than_request_side():
    """kimi-latest: reasoning_effort accepted (18803) but no reasoning_content
    returned (19170) -- the two predicates must be allowed to disagree."""
    assert moonshot_model_supports_reasoning_effort("kimi-latest") is True
    assert moonshot_model_returns_reasoning_content("kimi-latest") is False


def test_chat_predicates_survive_a_user_configured_capability_table():
    """Request-validity facts must not be reachable from the user-overridable
    capability tables (same design rule TASK-18414/18802 pinned)."""
    ModelCapabilities(
        config={
            "models": {
                "kimi-k3": {"supports_reasoning_effort": False},
                "glm-5.2": {"supports_reasoning_effort": False},
            },
            "patterns": {},
        }
    )
    assert moonshot_model_supports_reasoning_effort("kimi-k3") is True
    assert moonshot_model_rejects_sampling_params("kimi-k3") is True
    assert moonshot_model_returns_reasoning_content("kimi-k3") is True
    assert zai_model_supports_reasoning_effort("glm-5.2") is True
