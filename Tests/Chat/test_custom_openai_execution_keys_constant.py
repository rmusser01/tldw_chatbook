"""One shared custom-keys constant for the custom execution family.

ADR-179 Phase 2 Task 6 decision 1: every surface keyed on the custom
family's EXECUTION keys consumes ``CUSTOM_OPENAI_EXECUTION_KEYS`` from
``Chat/console_provider_support.py`` -- the engine swap routes
``openai_compatible`` custom-ep entries through ``custom-hosted``, and a
hand-maintained literal set would silently drop the swapped key from
base-URL forwarding, credential decisions, and thinking support.

The literal-grep test below scans ``tldw_chatbook/`` for the bare
``custom-openai-api`` / ``custom-openai-api-2`` strings and fails on any
file outside the audited allowlist (decision 2), so a NEW membership site
cannot slip past the constant. The allowlist is exactly the per-site audit
of Task 6 decision 1 -- identity/registration surfaces whose literals are
inherent (aliases, dispatch keys, record keys, the legacy handlers' own
wire keys) -- and every allowlisted file carries a PINNED expected
occurrence count, so even those files cannot quietly grow new literals.
"""
from __future__ import annotations

import re
from pathlib import Path

from tldw_chatbook.Chat.console_provider_gateway import (
    _CUSTOM_CREDENTIAL_DECISION_PROVIDERS,
)
from tldw_chatbook.Chat.console_provider_support import (
    CUSTOM_OPENAI_EXECUTION_KEYS,
    _CUSTOM_OPENAI_THINKING_KEYS,
    console_generation_control_support,
)

_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook"
)

# Per-site audit (Task 6 decision 1) -- files that may keep bare custom
# literals, with the pinned occurrence count of each spelling:
# (path relative to repo root): (bare "custom-openai-api", "custom-openai-api-2")
_AUDITED_LITERAL_ALLOWLIST: dict[str, tuple[int, int]] = {
    # The defining module (the constant's own members + readiness aliases).
    "tldw_chatbook/Chat/console_provider_support.py": (3, 3),
    # Registry record keys (registration data; stdlib-only module).
    "tldw_chatbook/provider_registry.py": (2, 2),
    # Dispatch registration + the legacy param-map keys + the copied-map
    # comment (identity/registration, not membership).
    "tldw_chatbook/Chat/Chat_Functions.py": (3, 2),
    # Audit-leave: identity handler-key universe (settings options).
    "tldw_chatbook/Chat/console_session_settings.py": (1, 1),
    # Audit-leave: identity aliases (execution spelling -> readiness key).
    "tldw_chatbook/Chat/provider_endpoint_contract.py": (1, 1),
    # Audit-leave: identity aliases (setup persistence).
    "tldw_chatbook/Chat/provider_setup_persistence.py": (1, 1),
    # Audit-leave: docstring reference to the legacy table's own keys.
    "tldw_chatbook/Chat/provider_readiness.py": (1, 1),
    # Audit-leave: identity/handler universe for model-list resolution.
    "tldw_chatbook/LLM_Provider_Catalog/model_discovery_provider_identity.py": (
        1,
        1,
    ),
    # Audit-leave: picker classification on identity-derived execution keys
    # (custom-hosted never appears there; custom-ep ids classify via the
    # custom group keys).
    "tldw_chatbook/Widgets/Console/console_provider_picker.py": (1, 0),
    # The legacy named handlers and their ADR-066 wire keys (untouched by
    # the swap; the kill switch keeps them the live path when off).
    "tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py": (3, 2),
    # Audit-leave: user-configured summarization endpoint dispatch;
    # "custom-hosted" is not a selectable summarization endpoint.
    "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": (1, 1),
    # Audit-include (registry-parity-forced membership): the module is
    # deliberately pure (no provider imports), so it keeps literals --
    # NATIVE_TOOLS_PROVIDERS == provider_registry.NATIVE_TOOLS_KEYS pins it.
    "tldw_chatbook/Agents/native_tools.py": (1, 1),
}

_BARE = re.compile(r"custom-openai-api(?!-2)")
_SUFFixed = re.compile(r"custom-openai-api-2")


def _counts(text: str) -> tuple[int, int]:
    return len(_BARE.findall(text)), len(_SUFFixed.findall(text))


def test_constant_shape() -> None:
    assert CUSTOM_OPENAI_EXECUTION_KEYS == frozenset(
        {"custom-openai-api", "custom-openai-api-2", "custom-hosted"}
    )


def test_membership_sites_consume_the_constant() -> None:
    # support: ADR-066 thinking keys (and through them
    # _LOCAL_REASONING_EXECUTION_KEYS) cover the swapped key.
    assert _CUSTOM_OPENAI_THINKING_KEYS is CUSTOM_OPENAI_EXECUTION_KEYS
    # gateway: the credential-decision providers are the constant.
    assert _CUSTOM_CREDENTIAL_DECISION_PROVIDERS is CUSTOM_OPENAI_EXECUTION_KEYS


def test_custom_hosted_keeps_the_local_reasoning_support_projection() -> None:
    # The include must preserve the legacy custom projections: effort is
    # offered with "unknown" support (arbitrary custom servers), budget is
    # unsupported (dropped on the custom wire).
    assert (
        console_generation_control_support(
            "custom-hosted", None, "reasoning_effort"
        )
        == "unknown"
    )
    assert (
        console_generation_control_support(
            "custom-hosted", None, "thinking_budget_tokens"
        )
        == "unsupported"
    )


def test_no_bare_custom_literals_outside_the_audited_allowlist() -> None:
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        bare, suffixed = _counts(text)
        if bare == 0 and suffixed == 0:
            continue
        relative = str(path.relative_to(_PACKAGE_ROOT.parent))
        allowed = _AUDITED_LITERAL_ALLOWLIST.get(relative)
        if allowed is None:
            offenders.append(f"{relative}: unexpected literals {bare}/{suffixed}")
        elif allowed != (bare, suffixed):
            offenders.append(
                f"{relative}: literal counts {bare}/{suffixed} != pinned "
                f"{allowed[0]}/{allowed[1]}"
            )
    assert not offenders, "\n".join(offenders)


def test_allowlist_entries_all_exist() -> None:
    # A stale allowlist entry (file moved/renamed) must fail, not rot.
    for relative in _AUDITED_LITERAL_ALLOWLIST:
        assert (_PACKAGE_ROOT.parent / relative).is_file(), relative
