"""TASK-33620.2 / TASK-37031: every shipped Claude model that reads images is
vision-capable, with a fresh config and with an old one.

The shipped default Console model is Anthropic ``claude-haiku-4-5``. Before
this fix no row in the code defaults (``DEFAULT_MODEL_PATTERNS``) or the
shipped config template (``[model_capabilities]``) matched it, nor
``claude-opus-5`` or ``claude-fable-5``: the Anthropic patterns cover only
``claude-3``, ``*opus-4*`` and ``*sonnet-4*``. So staging an image on the
default model blocked Send.

Live facts this pins (api.anthropic.com, 2026-10-27, recorded under the
TASK-33620.2 evidence): ``GET /v1/models/<id>`` reports
``capabilities.image_input.supported = true`` for claude-haiku-4-5
(-20251001), claude-opus-5, claude-fable-5 and claude-sonnet-5, and a real
``/v1/messages`` call with a 64x64 PNG on claude-haiku-4-5 answered "Green
top, purple bottom." The dated claude-3/claude-4 ids in the shipped list now
return not_found (retired); the claude-2 ids never took images.

A config.toml written by an older release carries the whole
``[model_capabilities]`` block, and its ``patterns.Anthropic`` list replaces
the shipped one outright (``deep_merge_dicts`` merges tables, not arrays).
That is why the family rule lives in code, beneath the user's own rows.
"""

from __future__ import annotations

import tomllib

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook import model_capabilities as capabilities_module
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.model_capabilities import ModelCapabilities

pytestmark = pytest.mark.unit

#: The shipped Anthropic ids that never accepted image input (Claude 2).
TEXT_ONLY_SHIPPED_ANTHROPIC = frozenset({"claude-2.1", "claude-2.0"})

#: The ``[model_capabilities]`` block a config.toml written before this fix
#: carries, verbatim from the template at origin/dev f9baac8706.
OLD_MODEL_CAPABILITIES_BLOCK = """
[model_capabilities]

[model_capabilities.models]
"gpt-4-vision-preview" = { vision = true, max_images = 1 }
"gpt-4-turbo" = { vision = true, max_images = 10 }
"gpt-4o" = { vision = true, max_images = 10 }
"claude-3-opus-20240229" = { vision = true, max_images = 5 }
"claude-3-sonnet-20240229" = { vision = true, max_images = 5 }
"claude-3-haiku-20240307" = { vision = true, max_images = 5 }
"claude-3-5-sonnet-20240620" = { vision = true, max_images = 5 }
"claude-3-5-sonnet-20241022" = { vision = true, max_images = 5 }

[model_capabilities.patterns]
Anthropic = [
    { pattern = "^claude-3", vision = true },
    { pattern = "^claude.*opus-4", vision = true },
    { pattern = "^claude.*sonnet-4", vision = true }
]

[model_capabilities.defaults]
unknown_models_vision = false
log_unknown_models = true
"""


def _shipped_anthropic_models() -> list[str]:
    models = config_module.DEFAULT_CONFIG_FROM_TOML["providers"]["Anthropic"]
    assert models, "the shipped config template lists no Anthropic models"
    return list(models)


def _fresh_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        config_module.DEFAULT_CONFIG_FROM_TOML["model_capabilities"]
    )


def _old_file_capabilities() -> ModelCapabilities:
    # The loader's own merge: defaults first, then the user's file on top.
    merged = config_module.deep_merge_dicts(
        config_module.DEFAULT_CONFIG_FROM_TOML,
        tomllib.loads(OLD_MODEL_CAPABILITIES_BLOCK),
    )
    old_patterns = merged["model_capabilities"]["patterns"]["Anthropic"]
    assert [row["pattern"] for row in old_patterns] == [
        "^claude-3",
        "^claude.*opus-4",
        "^claude.*sonnet-4",
    ], "the old file's Anthropic list must replace the shipped one"
    return ModelCapabilities(merged["model_capabilities"])


@pytest.mark.parametrize(
    "build", [_fresh_capabilities, _old_file_capabilities], ids=["fresh", "old-file"]
)
@pytest.mark.parametrize("provider", ["Anthropic", "anthropic"])
def test_every_shipped_anthropic_model_that_reads_images_is_vision_capable(
    build, provider
):
    capabilities = build()
    shipped = _shipped_anthropic_models()
    assert {"claude-haiku-4-5", "claude-opus-5", "claude-fable-5"} <= set(shipped)

    verdicts = {
        model: capabilities.is_vision_capable(provider, model) for model in shipped
    }

    assert {m for m, v in verdicts.items() if not v} == (
        TEXT_ONLY_SHIPPED_ANTHROPIC & set(shipped)
    ), verdicts


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-4-5-20251001",  # the dated id haiku-4-5 resolves to
        "claude-haiku-5-5",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-8",
        "claude-fable-5-1",
        "claude-mythos-5",
    ],
)
def test_current_claude_families_read_images_under_an_old_file(model):
    assert _old_file_capabilities().is_vision_capable("Anthropic", model) is True


def test_a_users_explicit_text_only_row_still_wins():
    capabilities = ModelCapabilities(
        {
            "models": {"claude-haiku-4-5": {"vision": False}},
            "patterns": {"Anthropic": []},
        }
    )

    assert capabilities.is_vision_capable("Anthropic", "claude-haiku-4-5") is False


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("Anthropic", "claude-2.1"),
        ("Anthropic", "claude-instant-1.2"),
        ("Anthropic", "claude-haiku-3"),  # tier-first names start at 4
        ("llama_cpp", "claude-haiku-4-5"),  # a local alias is not Anthropic's model
    ],
)
def test_the_family_rule_does_not_reach_past_anthropic_claude_4(provider, model):
    assert _old_file_capabilities().is_vision_capable(provider, model) is False


def test_the_console_vision_gate_lets_claude_haiku_4_5_take_an_image(monkeypatch):
    """The Console's own gate, through the module-level registry it reads.

    claude-haiku-4-5 is the shipped Anthropic default (``[character_defaults]``
    and the summarizer), and the model TASK-37031 found blocked live.
    """
    merged = config_module.deep_merge_dicts(
        config_module.DEFAULT_CONFIG_FROM_TOML,
        tomllib.loads(OLD_MODEL_CAPABILITIES_BLOCK),
    )
    monkeypatch.setattr(
        config_module, "load_cli_config_and_ensure_existence", lambda *a, **k: merged
    )
    monkeypatch.setattr(capabilities_module, "_global_capabilities", None)

    assert merged["character_defaults"]["model"] == "claude-haiku-4-5"
    assert attachment_core.vision_block_reason("anthropic", "claude-haiku-4-5") is None
    assert attachment_core.vision_block_reason("Anthropic", "claude-haiku-4-5") is None
    # Negative control: the same gate still refuses a text-only Claude.
    assert "can't accept images" in (
        attachment_core.vision_block_reason("anthropic", "claude-2.1") or ""
    )


def _models_dev_stub(rows):
    """A ``_models_dev_capabilities`` stand-in serving ``rows`` by model id."""

    def lookup(provider, model):
        row = rows.get(model)
        return dict(row) if row else {}

    return lookup


def test_the_family_rule_keeps_the_models_dev_gap_fill(monkeypatch):
    """The family rule adds ``vision``; it must not replace the models.dev row.

    A user who turned on ``[model_catalog] use_models_dev`` gets claude-opus-5's
    upstream context window. The first cut of the family rule returned a bare
    ``{"vision": True}`` ahead of the gap-fill, so that window was lost and
    Console request capacity fell back to an unverified 200K provider guess.
    """
    from tldw_chatbook.Utils.token_counter import resolve_context_window

    monkeypatch.setattr(
        capabilities_module,
        "_models_dev_capabilities",
        _models_dev_stub(
            {
                "claude-opus-5": {
                    "vision": True,
                    "context_window": 1_000_000,
                    "source": "models.dev",
                },
                # An upstream row that says "no images" for a family that
                # reads them: the verified family rule decides vision, the
                # row still supplies the window.
                "claude-fable-5": {
                    "vision": False,
                    "context_window": 500_000,
                    "source": "models.dev",
                },
            }
        ),
    )
    capabilities = _old_file_capabilities()

    assert capabilities.get_model_capabilities("Anthropic", "claude-opus-5") == {
        "vision": True,
        "context_window": 1_000_000,
        "source": "models.dev",
    }
    assert capabilities.get_model_capabilities("anthropic", "claude-fable-5") == {
        "vision": True,
        "context_window": 500_000,
        "source": "models.dev",
    }
    # Unknown upstream too: the family rule alone still answers vision.
    assert capabilities.get_model_capabilities("Anthropic", "claude-haiku-4-5") == {
        "vision": True
    }

    window = resolve_context_window(
        "anthropic", "claude-opus-5", capabilities=capabilities
    )
    assert (window.tokens, window.source, window.verified) == (
        1_000_000,
        "model catalog",
        True,
    )


def test_a_users_text_only_row_wins_over_models_dev_and_the_family_rule(
    monkeypatch,
):
    monkeypatch.setattr(
        capabilities_module,
        "_models_dev_capabilities",
        _models_dev_stub(
            {"claude-haiku-4-5": {"vision": True, "source": "models.dev"}}
        ),
    )
    capabilities = ModelCapabilities(
        {
            "models": {"claude-haiku-4-5": {"vision": False}},
            "patterns": {"Anthropic": []},
        }
    )

    assert capabilities.get_model_capabilities("Anthropic", "claude-haiku-4-5") == {
        "vision": False
    }
