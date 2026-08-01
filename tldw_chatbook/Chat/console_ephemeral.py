"""Temporary (non-persisted) Console conversations: shared vocabulary.

A temporary session never acquires a ``persisted_conversation_id`` (see
``ConsoleChatStore.persist_session_if_needed``), so every durable write in
the store no-ops on its own. What this module owns is the OTHER half of the
guarantee: the UI actions that would write a derived artifact to disk even
though no conversation row exists.

The registry below is the single place that list lives. Adding a new
artifact-producing Console action means adding a row here.

Honest scope of the automated coverage (final-review F3): the three tests
in ``Tests/Chat/test_console_ephemeral.py`` that reference this registry
(``test_blocked_reason_only_applies_to_temporary_sessions``,
``test_blocked_reasons_name_the_artifact_not_the_feature``,
``test_user_facing_copy_never_overstates_the_guarantee``) all iterate the
registry's OWN keys -- they check that every entry is well-formed, never
that some artifact-producing action is missing FROM it. They cannot catch
a new sink that forgets to add a row here. The one test that genuinely can
is ``Tests/UI/test_console_native_chat_flow.py::
test_console_save_as_labels_are_all_registered_in_the_ephemeral_gate``: it
spies on ``blocked_reason`` while driving the real per-message Save-as call
path, which builds its action id as ``f"save-as-{label.lower()}"`` from a
label list independent of this registry -- the one dynamically-constructed
id family here, and the one most likely to drift silently (a missing key
just falls back to a generic "Not available" sentence instead of raising).
Every other row is reached through a static string literal at its own call
site (``grep -rn 'blocked_reason(' tldw_chatbook/`` to enumerate them
today); keeping those in sync with this dict is manual discipline this
docstring cannot enforce, and no test here claims otherwise. Agent
tool-call sinks are covered by a SECOND function in this module,
``tool_blocked_reason`` -- which does not depend on this dict being
complete, because its policy is source-based and its default is refuse
(see that function's own docstring).

The promise is LOCAL DURABILITY only: "not saved locally". Nothing here may
imply privacy or provider-side behavior.

Sink audit (task 1): searching ``tldw_chatbook/Chat/`` and
``tldw_chatbook/Widgets/Console/`` for write/export patterns, and tracing
every ``action_id`` reachable from the Console message-action row and
composer menu in ``tldw_chatbook/UI/Screens/chat_screen.py``, surfaced four
per-message "Save as..." destinations and a message-level "Save Image"
action that write to local storage independently of conversation
persistence, plus a context-snapshot exporter reachable from any Console
session. None of these were named in the design spec's known-sinks list.
A review follow-up also traced the message-action row's ``speak`` entry to
a real file write (TTS playback audio); it is deliberately NOT in this
registry because the file is a transient OS-temp playback buffer,
secure-deleted within seconds and never exposed to the user, not a durable
artifact -- see the spec's audit table for the full reasoning.
See ``Docs/superpowers/specs/2026-07-31-temporary-conversations-design.md``
(``## Sink audit (task 1)``) for the full table and reasoning.
"""

from __future__ import annotations

#: Composer-menu action id for promoting a temporary chat ("Save this chat").
ACTION_SAVE_CHAT = "save-chat"

#: Chip label shown in the Console status strip while a chat is temporary.
TEMPORARY_LABEL = "Temporary — not saved"

#: Chip tooltip. Says what survives and what does not, without implying more.
TEMPORARY_TOOLTIP = (
    "This chat is not saved locally. It is lost when the tab closes or the "
    "app restarts. Activate to save it."
)

#: Action id -> why it is unavailable while the chat is temporary. Keyed by
#: the ids the Console workbench, composer menu, and message-action row
#: already use, so a lookup needs no translation layer. Also keyed, since
#: final-review F4, by the three write-shaped AGENT TOOL names
#: (``create_note``/``update_note``/``write_file`` -- see
#: ``Agents/tool_catalog.py``'s ``BuiltinToolProvider.invoke``) that an
#: ordinary Console reply can compose and dispatch independently of any
#: action_id above; ``blocked_reason`` is a plain string->string lookup, so
#: reusing the one dict keeps UI actions and these three built-in tool
#: names in one place instead of splitting them across two lists that
#: could drift apart. This is NOT a single registry for every
#: artifact-producing sink; in particular it is only ever consulted for
#: AUDITED (built-in) tools. MCP tools (``Agents/mcp_tool_provider.py``'s
#: ``MCPToolProvider``) and skill-provided tools (``Agents/
#: tool_catalog.py``'s ``SkillToolProvider``) are arbitrary third-party
#: code whose write behavior cannot be determined by name, so they are
#: never looked up here at all -- ``tool_blocked_reason`` refuses them
#: wholesale in a temporary session, and that refusal is enforced at the
#: shared choke point ``ToolCatalogRegistry.invoke_by_name``.
EPHEMERAL_BLOCKED_ACTIONS: dict[str, str] = {
    "generate-image": (
        "Generating an image writes a file to disk — not available in a "
        "temporary chat."
    ),
    "save-chatbook": (
        "Saving a Chatbook exports a file to disk — not available in a "
        "temporary chat."
    ),
    "save-image": (
        "Saving the image writes a file to disk — not available in a "
        "temporary chat."
    ),
    "save-as-note": (
        "Saving as a Note writes it to the local Notes database — not "
        "available in a temporary chat."
    ),
    "save-as-media": (
        "Saving as Media writes it to the local Media library — not "
        "available in a temporary chat."
    ),
    "save-as-prompt": (
        "Saving as a Prompt writes it to the local Prompts library — not "
        "available in a temporary chat."
    ),
    "save-as-chatbook": (
        "Saving as a Chatbook artifact exports a file to disk — not "
        "available in a temporary chat."
    ),
    "save-context": (
        "Saving the context snapshot writes a JSON file to disk — not "
        "available in a temporary chat."
    ),
    "create_note": (
        "The create_note tool writes to the local Notes database — not "
        "available in a temporary chat."
    ),
    "update_note": (
        "The update_note tool writes to the local Notes database — not "
        "available in a temporary chat."
    ),
    "write_file": (
        "The write_file tool writes a file to disk — not available in a "
        "temporary chat."
    ),
}


def blocked_reason(action_id: str, *, ephemeral: bool) -> str | None:
    """Return why ``action_id`` is unavailable, or ``None`` when it is available.

    Args:
        action_id: Console action id (workbench action, composer menu entry,
            or message-action row entry).
        ephemeral: Whether the active session is temporary.

    Returns:
        The reason sentence to show on the disabled control, or ``None`` when
        the action is available (which is always the case outside a temporary
        chat).
    """
    if not ephemeral:
        return None
    return EPHEMERAL_BLOCKED_ACTIONS.get(action_id)


#: Catalog ``source`` values whose tools are audited in-tree well enough to
#: be judged one name at a time against ``EPHEMERAL_BLOCKED_ACTIONS``. Only
#: the built-ins qualify: they are this repo's own code, enumerated in
#: ``Agents/tool_catalog.py``'s ``_GATEABLE_BUILTINS`` plus the two always-on
#: tools, so "does this write locally?" is answerable by reading them.
#: Deliberately a whitelist, never a blacklist -- see ``tool_blocked_reason``.
EPHEMERAL_AUDITED_TOOL_SOURCES: frozenset[str] = frozenset({"builtin"})

#: Human-readable phrases for the unaudited sources we know about today, used
#: only to make the refusal sentence specific. An unrecognised source is not
#: an error -- it just gets the generic phrase and is still refused.
_UNAUDITED_SOURCE_PHRASES: dict[str, str] = {
    "skill": "a skill",
    "mcp": "an MCP server",
}

_UNKNOWN_SOURCE_PHRASE = "an unaudited provider"

#: Refusal sentence for a non-built-in tool. Same convention as the rows
#: above: name what would be written (here: "may write to this device",
#: since that is genuinely all we know), never imply privacy.
EPHEMERAL_UNAUDITED_TOOL_REASON = (
    "The {name} tool is provided by {source} and may write to this device — "
    "not available in a temporary chat."
)


def tool_blocked_reason(
    tool_name: str, *, source: str | None, ephemeral: bool
) -> str | None:
    """Return why an agent tool call is refused, or ``None`` to allow it.

    The policy for a temporary session, in one place:

    * A tool from an **audited** source (``EPHEMERAL_AUDITED_TOOL_SOURCES``,
      i.e. the in-tree built-ins) is judged per name against
      ``EPHEMERAL_BLOCKED_ACTIONS`` -- the write-shaped ones are refused and
      the read-only ones (``read_file``/``list_directory``/``glob_files``/
      ``grep_files``/``calculator``/``get_current_datetime``) keep working.
    * A tool from **any other** source -- skills, MCP servers, a provider
      added after this was written, or a caller that could not determine the
      source at all (``None``) -- is refused outright. That code is arbitrary
      and its write behavior cannot be established statically, so unknown
      capability must fail toward not-writing, exactly as the store's own
      persistence gate does.

    The default being *refuse* is the point: a provider added later is
    covered by this function on the day it is added, without anyone
    remembering to come back here.

    Args:
        tool_name: The LLM-facing tool name the model called.
        source: The tool's catalog ``source`` (``ToolCatalogEntry.source``),
            or ``None`` when the caller could not resolve one.
        ephemeral: Whether the running session is temporary.

    Returns:
        The refusal sentence to hand back to the model, or ``None`` when the
        call may proceed (which is always the case outside a temporary chat).
    """
    if not ephemeral:
        return None
    if source in EPHEMERAL_AUDITED_TOOL_SOURCES:
        return blocked_reason(tool_name, ephemeral=True)
    phrase = _UNAUDITED_SOURCE_PHRASES.get(source or "", _UNKNOWN_SOURCE_PHRASE)
    return EPHEMERAL_UNAUDITED_TOOL_REASON.format(name=tool_name, source=phrase)
