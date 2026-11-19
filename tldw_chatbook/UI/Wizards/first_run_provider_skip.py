"""A real skip for the first-run Provider step (TASK-34100.6 slice A).

Review findings new-a11y-terminal-visual-02, new-provider-01 and provider-01
(2026-10-02). The old skip (Enter in the empty key field) cleared the step's
provider choice but never the wizard-level staged draft, so Model kept showing
the skipped provider and its template model and Next looped on a generic
'Saving the provider and model setup failed'. These helpers own the pieces the
pinned step and container modules call:

* ``release_staged_provider_setup`` drops the wizard's staged draft and every
  model handoff derived from it, without touching the step's own per-provider
  drafts, so Back still shows the user's choice and any key they typed.
* ``model_save_failure_message`` names why a Model save failed and an action
  that can work.
"""

from __future__ import annotations

from typing import Any

def release_staged_provider_setup(wizard: Any) -> None:
    """Release the staged provider connection for a skip.

    Unlike ``SetupWizardContainer.clear_provider_setup_sensitive_state`` this
    leaves the Provider step's own drafts alone: a skip keeps the user's
    choice and anything they typed (review four rules: a skip never silently
    discards a key). Only the wizard-level handoff that Model reads is dropped,
    so Model honestly renders 'Pick a provider first'.

    Args:
        wizard: The ``SetupWizardContainer`` (anything else is ignored).
    """
    if wizard is None or not hasattr(wizard, "_staged_provider_draft"):
        return
    wizard.invalidate_provider_write_expectation()
    wizard._provider_stage_generation += 1
    task = getattr(wizard, "_provider_commit_task", None)
    if (
        task is not None
        and not task.done()
        and not getattr(wizard, "_provider_commit_write_started", False)
    ):
        wizard._provider_commit_generation += 1
        task.cancel()
        wizard._provider_commit_task = None
        wizard._provider_commit_identity = None
    wizard._staged_provider_draft = None
    wizard._provider_setup_committed = False
    wizard._committed_provider_model = ""
    wizard._committed_provider_expected_state = None
    # Clears the discovered models, outcomes and preconditions, the recorded
    # Model step data, and Model's own selection (including a template model).
    wizard.invalidate_provider_model_handoff()


def model_save_failure_message(wizard: Any) -> str:
    """Why the Model step's provider+model save failed, and what works.

    Args:
        wizard: The ``SetupWizardContainer``.

    Returns:
        One line naming the real cause and an action that can succeed.
    """
    if getattr(wizard, "staged_provider_draft", None) is None:
        return (
            "No provider is connected, so there is no model to save. Go Back to "
            "connect one, or clear the model name and press Next to skip."
        )
    result = getattr(wizard, "_provider_last_config_result", None)
    phase = getattr(result, "failure_phase", None)
    if getattr(result, "conflict", False):
        return (
            "Your settings changed in another window while saving. Press Next to "
            "save again."
        )
    if phase == "before_replace":
        return (
            "Couldn't write config.toml, so nothing was saved. Check the disk "
            "and the file's permissions, then press Next to try again."
        )
    if phase == "cache_reload":
        return (
            "config.toml was written, but the app couldn't reload it. Press Next "
            "to try again."
        )
    return (
        "The provider settings couldn't be saved. Go Back to Provider to check "
        "them, or clear the model name and press Next to skip."
    )


#: The skip control's label, also named by the refusal and help copy.
SKIP_LABEL = "Skip — connect later"


def skip_provider_step(step: Any) -> tuple[bool, str]:
    """Commit the Provider step as a skip: nothing staged, the choice kept.

    Args:
        step: The ``ProviderStep``.

    Returns:
        ``(True, "")``: a skip always advances.
    """
    step._skipped = True
    release_staged_provider_setup(getattr(step, "wizard", None))
    return True, ""


def empty_key_skips(step: Any, readiness: Any, typed: str) -> bool:
    """Whether Next on a not-ready provider is the empty-key skip (AC#3).

    An empty key field means "no key yet", so Next skips exactly as Enter in
    the field does. Four not-ready states are not "no key yet" and keep their
    refusal: a Claude subscription that is being checked or has lapsed (the
    credential is the login, not the field); a stored key the user just
    chose to Clear (skipping would silently keep the key they asked to drop);
    a key that is there while the per-account URL is missing (readiness
    reports a missing URL only once the key is found); and a variable the
    user declared in ``api_key_env_var`` that isn't set (review round 1, F4).
    The explicit 'Skip — connect later' control still skips all four.

    Args:
        step: The ``ProviderStep``.
        readiness: Its current ``ProviderReadiness``.
        typed: The key field's value.

    Returns:
        True when Next should skip the step.
    """
    if typed.strip() or getattr(step, "_clear_requested", False):
        return False
    if missing_url_reason(readiness) or _declared_env_var(step, readiness):
        return False
    return getattr(readiness, "subscription_status", None) is None


def ask_resumed_key(step: Any, skip: bool = False) -> str:
    """Show a resumed provider's controls; the refusal while it needs a key.

    Review round 2, G6-VR2-05: a resumed run restores the provider without
    its controls (keys aren't kept), so the empty-key skip silently skipped a
    key field the user never saw. The first Next shows the section and asks.
    Restoring the full layout on resume itself is provider-04 (TASK-34100.10).

    Args:
        step: The ``ProviderStep``, with ``selected_provider_key`` restored.
        skip: An explicit skip was asked for; it is never refused.

    Returns:
        The refusal, or "" when the provider needs no key now.
    """
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    if skip or step.selected_provider_key in step._provider_drafts:
        return ""  # an explicit skip, or picked on screen: its controls show
    step.select_provider(step.selected_provider_key)
    readiness = step._current_provider_readiness()
    if readiness.ready or not readiness.requires_api_key:
        return ""
    step._focus_key_after_pick()
    display = provider_display_name(step.selected_provider_key)
    return (
        f"Paste your {display} API key above (keys aren't kept when setup is "
        f"interrupted), or choose {SKIP_LABEL}."
    )


def pick_configures(step: Any, provider_key: str) -> bool:
    """Whether a pick (Enter, Space or a click) must configure the provider.

    A re-pick of the shown provider is a no-op (provider-26). A resumed run
    restores the provider without its controls, so the first pick of it
    configures it and shows its key field (review round 2, G6-VR2-05).
    """
    return (
        provider_key != step.selected_provider_key
        or provider_key not in step._provider_drafts
    )


def missing_url_reason(readiness: Any) -> str:
    """'resource URL' when readiness reports a missing per-account URL, else ""."""
    reason = str(getattr(readiness, "reason", "") or "")
    if reason.startswith("Missing ") and reason.endswith("URL"):
        return reason.removeprefix("Missing ")
    return ""


def _declared_env_var(step: Any, readiness: Any) -> str:
    """A key variable the user named in config (not the provider's default)."""
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var

    env_var = str(getattr(readiness, "env_var", "") or "")
    key = getattr(step, "selected_provider_key", "")
    return env_var if env_var and env_var != default_api_key_env_var(key) else ""


def provider_refusal(step: Any, readiness: Any) -> str:
    """Refusal copy for a Provider Next that has something typed but isn't ready.

    Leads with the on-screen fix (review cross-cutting-07 rule 1) and names
    the real gap: a missing per-account URL is reported as a missing URL with
    the provider's display name (provider-05), never 'API key required'.

    Args:
        step: The ``ProviderStep``.
        readiness: Its current ``ProviderReadiness``.

    Returns:
        One line, with no "Retry with Next" suffix.
    """
    from tldw_chatbook.Chat.provider_catalog import provider_display_name

    display = provider_display_name(step.selected_provider_key)
    if getattr(readiness, "subscription_status", None) is not None:
        return str(readiness.user_message)
    typed = step.query_one("#setup-provider-api-key").value
    if not typed.strip().isascii():
        from tldw_chatbook.UI.Wizards.first_run_provider_key_entry import (
            non_ascii_message,
        )

        count = sum(1 for character in typed if not character.isascii())
        return non_ascii_message(count)
    if getattr(step, "_clear_requested", False):
        return (
            f"You chose Clear for the stored {display} API key. Paste a new key "
            f"above, press Keep current, or choose {SKIP_LABEL}."
        )
    if what := missing_url_reason(readiness):
        return f"Add your {display} {what} above to continue, or choose {SKIP_LABEL}."
    if (env_var := _declared_env_var(step, readiness)) and not typed.strip():
        return (
            f"Your config reads the {display} key from {env_var}, which isn't set in "
            f"the shell that started chatbook. Paste a key above, or choose {SKIP_LABEL}."
        )
    return (
        f"That doesn't look like a {display} API key. Paste it again above, or "
        f"choose {SKIP_LABEL}."
    )
