"""Native invocation projection behind the current Console controller methods."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any


def _native_provider_entries(maximum, messages, skill_bindings):
    """Keep exact checked native producers on the direct provider effect lane."""
    from tldw_chatbook.Agents.agent_models import PluginContextText, check_host_context

    check_host_context(messages, strip=False)
    entries = tuple(
        row for row in maximum.get("available_skills", ()) if row.get("plugin_owned")
    )
    selected = {
        index
        for index, row in enumerate(entries)
        if row.get("name") in skill_bindings or row.get("plugin_constraint_only")
    }
    for message in messages:
        content = message.get("content")
        parts = (
            [part.get("text") for part in content if isinstance(part, dict)]
            if isinstance(content, list)
            else [content]
        )
        for text in parts:
            if not isinstance(text, PluginContextText):
                continue
            for origin in text.checked_origins():
                matches = [
                    index
                    for index, row in enumerate(entries)
                    if (
                        row.get("plugin_installation_id"),
                        row.get("plugin_revision"),
                        row.get("plugin_component_id"),
                    )
                    == (origin.installation_id, origin.revision, origin.component_id)
                ]
                if len(matches) != 1:
                    raise PermissionError("plugin_context_producer_unavailable")
                selected.add(matches[0])
    pending = list(selected)
    while pending:
        row = entries[pending.pop()]
        for dependency in row.get("plugin_dependencies", ()):
            matches = [
                index
                for index, candidate in enumerate(entries)
                if candidate.get("plugin_installation_id")
                == row["plugin_installation_id"]
                and candidate.get("plugin_revision") == row["plugin_revision"]
                and candidate.get("plugin_component_id") == dependency
            ]
            if len(matches) != 1:
                raise PermissionError("plugin_context_dependency_unavailable")
            if matches[0] not in selected:
                selected.add(matches[0])
                pending.append(matches[0])
    return dict(
        maximum, available_skills=[entries[index] for index in sorted(selected)]
    )


def _native_selector(source, entries):
    from . import console_chat_controller as _owner

    word, rest = _owner._split_skill_command_word(source.lstrip())
    if not word.startswith(("/", "@")) or word.count(":") != 1:
        return None, rest, None
    alias, local_id = word[1:].split(":", 1)
    if not alias or not local_id:
        return None, rest, None
    kind = "command" if word[0] == "/" else "rule"
    rows = [
        row
        for row in entries
        if row.get("plugin_owned")
        and row.get("plugin_kind") == kind
        and row["name"] == f"{alias}:{kind}:{local_id}"
    ]
    if (
        not rows
        and kind == "rule"
        and any(
            row.get("plugin_owned")
            and row.get("plugin_kind") == "agent"
            and row["name"] == f"{alias}:agent:{local_id}"
            for row in entries
        )
    ):
        raise PermissionError("plugin_agent_child_only")
    if len(rows) != 1 or kind == "rule" and rows[0].get("plugin_rule_mode") != "manual":
        raise PermissionError("plugin_selector_unavailable")
    return rows[0], rest, kind


async def _legacy_native_substitution(self, messages, turn_context, invocation_source):
    from . import console_chat_controller as _owner

    maximum = turn_context.skill_context_maximum if turn_context is not None else {}
    final = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
        ),
        None,
    )
    if final is None or not isinstance(messages[final].get("content"), str):
        return None
    source = (
        messages[final]["content"] if invocation_source is None else invocation_source
    )
    row, rest, kind = self._native_selector(source, maximum.get("available_skills", ()))
    if row is None:
        return None
    service = getattr(
        getattr(self._skills_service, "local_service", None), "plugin_service", None
    )
    if service is None:
        raise PermissionError("plugin_owner_unavailable")
    admitted = await service.admit(maximum, maximum["plugin_run_id"])
    row, rest, kind = self._native_selector(source, admitted["available_skills"])
    arguments = rest
    if kind == "command" and len(row.get("arguments", ())) == 1:
        arguments = json.dumps({row["arguments"][0]: rest}, ensure_ascii=False)
    rendered = await service.execute_skill(
        row["name"],
        admission_token=row["plugin_admission"],
        args=arguments if kind == "command" else "",
    )
    blocks = [{"role": "user", "content": rendered["rendered_prompt"]}]
    if kind == "rule":
        _, expanded_rest = _owner._split_skill_command_word(
            messages[final]["content"].lstrip()
        )
        if expanded_rest:
            blocks.append({"role": "user", "content": expanded_rest})
    result = [*messages[:final], *blocks, *messages[final + 1 :]]
    from tldw_chatbook.Agents.agent_models import check_host_context

    check_host_context(result, strip=False)
    await service.check_entries(admitted["available_skills"])
    return result, None, (), (), ""


async def _native_skill_configuration(self, configuration, draft):
    from . import console_chat_controller as _owner

    maximum = configuration.skill_context_maximum
    entries = [
        row
        for row in maximum.get("available_skills", ())
        if row.get("plugin_owned") and row.get("plugin_kind") == "skill"
    ]
    if not entries:
        return configuration
    local = getattr(self._skills_service, "local_service", None)
    service = getattr(local, "plugin_service", None)
    if service is None:
        raise PermissionError("plugin_owner_unavailable")
    admitted = await service.admit(maximum, maximum["plugin_run_id"])
    entries = [
        row
        for row in admitted["available_skills"]
        if row.get("plugin_kind") == "skill" and row.get("user_invocable")
    ]
    word, rest = _owner._split_skill_command_word(draft.lstrip())
    resolution = (
        _owner.resolve_skill_command(
            word[len(_owner.MENTION_SIGIL) :],
            rest,
            self._skill_candidates_from_context(admitted)
            + self._skill_blocked_candidates_from_context(admitted),
        )
        if word.startswith(_owner.MENTION_SIGIL)
        else None
    )
    selected = [
        row
        for row in entries
        if resolution is not None
        and resolution.kind == "resolved"
        and row["name"] == resolution.name
    ]
    if not selected:
        names = frozenset(
            row["name"] for row in entries if row.get("context") == "inline"
        )
        mentioned = {
            mention.name for mention in _owner.find_embedded_mentions(draft, names)
        }
        selected = [row for row in entries if row["name"] in mentioned]
    if not selected:
        return configuration
    from tldw_chatbook.Chat.provider_readiness import provider_config_key
    from tldw_chatbook.Plugins.host_references import mapped_model

    keys = {
        (row["plugin_installation_id"], row["plugin_component_id"]) for row in selected
    }
    models = set()
    for snapshot in await service.component_snapshots(admitted):
        for component_id in snapshot.selection:
            if (snapshot.installation_id, component_id) not in keys:
                continue
            provider, model = mapped_model(
                snapshot.inspection.inventory[component_id],
                json.loads(snapshot.mappings_json),
            )
            if model:
                if provider_config_key(provider) != provider_config_key(
                    configuration.provider_selection.provider
                ):
                    raise PermissionError("plugin_model_parent_mismatch")
                models.add(model)
    if len(models) > 1:
        raise PermissionError("plugin_model_conflict")
    if models:
        from tldw_chatbook.Chat.console_configuration_capture import (
            max_history_images,
        )

        model = models.pop()
        provider = configuration.provider_selection.provider
        capabilities = dict(configuration.capabilities)
        capabilities["vision"] = bool(
            capabilities.get("vision")
        ) and _owner.is_vision_capable(provider, model)
        capabilities["max_history_images"] = min(
            int(capabilities.get("max_history_images", 0) or 0),
            max_history_images(provider, model),
        )
        return replace(
            configuration,
            provider_selection=replace(
                configuration.provider_selection, explicit_model=model
            ),
            capabilities=capabilities,
        )
    return configuration


async def _apply_skill_substitution(
    self,
    provider_messages,
    turn_context=None,
    *,
    invocation_source=None,
    reference_offsets=(),
):
    """Apply manual material, then whole active rules in the untrusted lane."""

    try:
        result = await self._legacy_native_substitution(
            provider_messages, turn_context, invocation_source
        )
    except (PermissionError, ValueError, KeyError):
        return (
            provider_messages,
            "Plugin context is unavailable; review selected components.",
            (),
            (),
            "",
        )
    legacy_result = result is not None
    if result is None:
        result = await self._substitute_skills(
            provider_messages,
            turn_context,
            invocation_source=invocation_source,
            reference_offsets=reference_offsets,
        )
    messages, refuse, notes, bindings, bundle = result
    if refuse or turn_context is None:
        return result
    maximum = turn_context.skill_context_maximum
    rules = [
        row
        for row in maximum.get("available_skills", ())
        if row.get("plugin_kind") == "rule" and row.get("plugin_rule_mode") == "always"
    ]
    if not rules:
        return result
    local = getattr(self._skills_service, "local_service", None)
    plugins = getattr(local, "plugin_service", None)
    try:
        if plugins is None:
            raise PermissionError("plugin_owner_unavailable")
        admitted = await plugins.admit(maximum, maximum["plugin_run_id"])
        blocks = await plugins.render_rules(admitted["available_skills"])
        # A legacy rule can replace one user row with body + remainder.
        # Always rules precede that whole contribution, not just its tail.
        source_rows = provider_messages if legacy_result else messages
        index = next(
            (
                i
                for i in range(len(source_rows) - 1, -1, -1)
                if source_rows[i].get("role") == "user"
            ),
            None,
        )
        if index is None:
            raise PermissionError("plugin_input_owner_unavailable")
        messages = [*messages[:index], *blocks, *messages[index:]]
        from tldw_chatbook.Agents.agent_models import check_host_context

        check_host_context(messages, strip=False)
        await plugins.check_entries(admitted["available_skills"])
    except (PermissionError, ValueError):
        return (
            provider_messages,
            "Plugin instructions are unavailable or too large; review selected components.",
            (),
            (),
            "",
        )
    return messages, None, notes, bindings, bundle


async def _prepare_native_references(
    self,
    draft,
    configuration,
    *,
    origin,
    machine_input,
    resumed_preparation,
    prepared_continuation,
):
    from . import console_chat_controller as _owner

    clean_draft = draft
    resumed_reference = (
        prepared_continuation
        if resumed_preparation is not None and prepared_continuation is not None
        else None
    )
    if (
        resumed_reference is not None
        and resumed_reference.invocation_source is not None
    ):
        clean_draft = resumed_reference.invocation_source
    _selected_native, _native_rest, native_kind = _native_selector(
        clean_draft, configuration.skill_context_maximum.get("available_skills", ())
    )
    first_word, _ = _owner._split_skill_command_word(clean_draft.lstrip())
    native_literal_arguments = native_kind == "command" or any(
        first_word == _owner.MENTION_SIGIL + row["name"] and row.get("plugin_owned")
        for row in configuration.skill_context_maximum.get("available_skills", ())
    )
    # Select finite metadata only. The original reference owner performs the
    # single expansion after ordinary admission and returns its exact offsets.
    executed_draft_text = (
        resumed_preparation.executed_draft
        if resumed_reference is not None
        else clean_draft
    )
    reference_records: tuple = ()
    reference_offsets = (
        resumed_reference.reference_offsets if resumed_reference is not None else ()
    )
    candidates = ()
    if (
        not origin.automatic
        and machine_input is None
        and resumed_preparation is None
        and not native_literal_arguments
    ):
        from .console_references import _native_reference_candidates

        candidates = tuple(_native_reference_candidates(clean_draft))
        if native_kind == "rule":
            start = len(clean_draft) - len(clean_draft.lstrip())
            candidates = tuple(
                candidate for candidate in candidates
                if not (candidate.start == start and candidate.raw == first_word)
            )
    return clean_draft, executed_draft_text, reference_records, reference_offsets, candidates


def _render_skill_bundle_block(results):
    """Render one combined "Bundled files" block for a turn's bound skills.

    Task 5 (skills-fork-reachability): `_apply_skill_substitution` builds
    this as pure string work from `execute_skill` results it already holds
    -- no re-execution, no extra service calls -- for every skill actually
    bound this turn (leading-resolved, or embedded mentions that spliced).
    `run_reply` (never here) is the only place that ever appends the
    returned string to a message, so plain sends and the stored transcript
    never see it.

    Row format matches `_BridgeSkillRunner.run`'s own bundle-pointer block
    byte-for-byte (Task 4) -- ``{path} ({size} bytes)`` / ``{path} ({size}
    bytes, binary)``, comma-joined under one combined header -- so a bound
    skill's `skill_file` reads look identical whether granted turn-side
    (this function) or fork-side (a spawned skill reading its own bundle).

    Args:
        results: `execute_skill` result mappings for the bound skills, in
            any order; a result missing `reference_files` (absent when a
            skill has no bundle beyond SKILL.md) or with it empty
            contributes no rows.

    Returns:
        The combined block, or ``""`` when no result carries any rows.
    """
    from . import console_chat_controller as _owner

    rows: list[str] = []
    for result in results:
        refs = (
            result.get("reference_files")
            if isinstance(result, _owner.Mapping)
            else None
        )
        if not refs:
            continue
        rows.extend(
            f"{ref['path']} ({ref['size']} bytes"
            f"{'' if ref.get('is_text', True) else ', binary'})"
            for ref in refs
        )
    if not rows:
        return ""
    return "Bundled files (readable via skill_file): " + ", ".join(rows)


def _skill_candidates_from_context(context):
    """Build the user-invocable, trusted skill candidate population.

    Mirrors ``ConsoleSkillController.
    _console_skill_trusted_candidates_from_context``'s filter -- kept as
    a small duplicate rather than a shared import because `Chat/`
    business logic must not depend on `UI/Screens/` (project layering),
    and `console_skill_resolver` deliberately stays unaware of trust/
    context shape (see its own module docstring).
    """
    from . import console_chat_controller as _owner

    available = (
        context.get("available_skills") if isinstance(context, _owner.Mapping) else None
    )
    return tuple(
        _owner.SkillCommandCandidate(
            name=str(item.get("name")),
            description=str(item.get("description") or ""),
        )
        for item in (available or [])
        if isinstance(item, _owner.Mapping)
        and item.get("name")
        and item.get("user_invocable", True)
        and not item.get("trust_blocked", False)
    )


def _skill_blocked_candidates_from_context(context):
    """Build the user-invocable, trust-BLOCKED (needs-review) skill
    candidate population.

    Companion to `_skill_candidates_from_context`: unioned with it in
    `_apply_skill_substitution` to widen the DETECTION population (never
    the executable one) so a `$blocked-name` mention resolves a name
    instead of silently staying literal, sigil-prefixed text with no
    refusal or note at all. `execute_skill` remains the sole authority
    on whether a resolved name may actually run -- candidates built here
    are never executed directly by this method's caller. A blocked
    skill flagged ``user_invocable: False`` is excluded, mirroring
    `_skill_candidates_from_context`'s own filter discipline.
    """
    from . import console_chat_controller as _owner

    blocked = (
        context.get("blocked_skills") if isinstance(context, _owner.Mapping) else None
    )
    return tuple(
        _owner.SkillCommandCandidate(
            name=str(item.get("name")),
            description=str(item.get("description") or ""),
        )
        for item in (blocked or [])
        if isinstance(item, _owner.Mapping)
        and item.get("name")
        and item.get("user_invocable", True)
    )


async def _substitute_skills(
    self,
    provider_messages,
    turn_context=None,
    *,
    invocation_source=None,
    reference_offsets=(),
):
    """Render-fresh the triggering turn's skill mention(s) at payload build time.

    Spec: "Invocation semantics" §5 (the substitution rule) -- one rule
    for fresh sends AND retry/regenerate/continue. Only the FINAL
    ``role == "user"`` message in ``provider_messages`` (the turn
    actually driving this send) is ever a substitution candidate; every
    earlier message -- including an earlier raw skill mention sitting
    in history -- is left untouched, so the persisted transcript always
    keeps the literal text the user typed (the raw mention is what gets
    submitted and stored; only the ephemeral provider payload for this
    turn is ever rendered). Re-resolves against a FRESH candidate
    snapshot and re-verifies trust through ``execute_skill`` on every
    call (never a cached snapshot), so a retry issued after a skill was
    edited (now untrusted) refuses/skips instead of silently re-running
    a stale render.

    Both forms are DETECTED against trusted candidates UNION
    user-invocable blocked (needs-review) skills -- a blocked skill must
    still be found (leading refuses, embedded degrades to literal +
    note) rather than silently staying plain, sigil-prefixed text with
    no signal at all. `execute_skill` remains the sole trust authority;
    detection here never grants execution.

    Two independent forms, tried in order:

    Leading form -- the message, with leading whitespace stripped
    (mirroring `_annotate_skill_commands`'s own preview `lstrip()`
    assumption -- a resolved leading mention replaces the whole message
    either way, so the leading whitespace simply disappears), starts
    with `MENTION_SIGIL` and the leading word resolves to a known
    skill: the REST of the (stripped) message is passed as that skill's
    args (`cap_skill_args`). A resolved leading mention is never also
    scanned for embedded mentions -- its args are opaque payload, not
    further mentions to expand.

    Embedded form -- tried whenever the leading form doesn't apply (no
    leading `MENTION_SIGIL`, or the leading word doesn't resolve):
    scans the ORIGINAL (unstripped) message. Every `$skill-name`
    mention anywhere in the message (case-sensitive, code-span-masked,
    document order -- `find_embedded_mentions`) is looked up ARGLESS
    (`execute_skill(name, mode="local", args="")`, once per unique
    name, right-to-left splice so earlier spans stay valid) and spliced
    in place at the mention's exact span, preserving all surrounding
    prose. Only an ``execution_mode == "inline"`` result splices;
    anything else (e.g. ``fork``, which has no "in place" meaning for
    an embedded mention) silently leaves that mention's literal `$name`
    text untouched. A trust-blocked mention (`SkillTrustBlockedError`)
    also leaves the literal text untouched but records a
    `SKILL_MENTION_SKIPPED_NOTE` for the caller to surface as a
    non-aborting system row.

    Args:
        provider_messages: The fully-built payload about to be sent to
            the provider (already includes any leading session-system
            message and any synthesized continuation instruction).

    Returns:
        A 5-tuple ``(provider_messages, refuse, notes, skill_bindings,
        skill_bundle_block)`` (Task 5, skills-fork-reachability).
        ``skill_bindings`` is the leading-RESOLVED skill's name (both
        ``inline`` and ``fork`` outcomes -- never on refuse) plus every
        embedded mention name that actually SPLICED (never a
        trust-blocked-literal or fork-literal mention).
        ``skill_bundle_block`` is the fully-rendered "Bundled files"
        block (`_render_skill_bundle_block`) for every bound skill
        whose `execute_skill` result carried non-empty
        `reference_files`, built as pure string work from the results
        already in hand this call (no re-execution, no extra service
        calls), or ``""`` when nothing bound has any. It is NEVER
        inserted into ``provider_messages`` here -- only ``run_reply``
        (bridge-side) ever appends it, so plain sends and the stored
        transcript never see it.

        ``(provider_messages, None, (), (), "")`` unchanged when there
        is no skills service configured, substitution is disabled,
        there is no final user message, that message's content isn't a
        string, or neither form applies. ``(new_messages, None, notes,
        skill_bindings, skill_bundle_block)`` when the leading form
        resolves and renders (``notes`` always empty for the leading
        form) or when the embedded pass splices one or more mentions
        (``notes`` carries one `SKILL_MENTION_SKIPPED_NOTE` per unique
        trust-blocked mention name, in document order); ``inline``
        replaces just the final message in place (history preserved);
        leading-form ``fork`` drops every message before it except a
        leading ``role == "system"`` message (clean context = session
        system prompt + rendered turn only).
        ``(provider_messages, refuse_copy, (), (), "")`` -- the
        ORIGINAL, unmodified messages, paired with
        `SKILL_UNTRUSTED_REFUSE` copy -- when a LEADING resolved skill
        is no longer trusted (`SkillTrustBlockedError` at
        execute-time); the caller must append `refuse_copy` as a
        system row and abort the turn without sending. An embedded
        mention never refuses/aborts -- it degrades to a
        literal-plus-note instead, and the send proceeds.
    """
    from . import console_chat_controller as _owner

    if self._skills_service is None or not self._skill_substitution_enabled:
        return provider_messages, None, (), (), ""

    final_index: int | None = None
    for index in range(len(provider_messages) - 1, -1, -1):
        if provider_messages[index].get("role") == _owner.ConsoleMessageRole.USER.value:
            final_index = index
            break
    if final_index is None:
        return provider_messages, None, (), (), ""

    content = provider_messages[final_index].get("content")
    if not isinstance(content, str):
        return provider_messages, None, (), (), ""
    source_content = content if invocation_source is None else invocation_source
    if _owner.MENTION_SIGIL not in source_content:
        # Fast path: no sigil anywhere means neither form can possibly
        # apply -- plain-text sends never touch the skills service.
        return provider_messages, None, (), (), ""

    frozen_context = (
        turn_context.skill_context_maximum if turn_context is not None else {}
    )

    plugin_entries = {
        item["name"]: item
        for item in frozen_context.get("available_skills", ())
        if item.get("plugin_owned")
    }
    local_service = getattr(self._skills_service, "local_service", None)
    plugin_service = getattr(local_service, "plugin_service", None)
    if plugin_entries:
        try:
            frozen_context = await plugin_service.admit(
                frozen_context, frozen_context["plugin_run_id"]
            )
            plugin_entries = {
                item["name"]: item
                for item in frozen_context["available_skills"]
                if item.get("plugin_owned")
            }
        except PermissionError:
            return (
                provider_messages,
                "Plugin authority changed; review plugins and retry.",
                (),
                (),
                "",
            )

    def plugin_options(name: str) -> dict[str, str]:
        entry = plugin_entries.get(name)
        return {"plugin_admission": entry["plugin_admission"]} if entry else {}

    def ensure_frozen_definition(name: str) -> None:
        if name in plugin_entries:
            return  # The plugin owner revalidates exact admission at execute.
        expected = next(
            (
                str(item.get("definition_digest"))
                for item in frozen_context.get("available_skills", ())
                if item.get("name") == name and item.get("definition_digest")
            ),
            None,
        )
        if expected is None:
            return
        local = getattr(self._skills_service, "local_service", None)
        if local is None and hasattr(self._skills_service, "trust_service"):
            local = self._skills_service
        trust = getattr(local, "trust_service", None)
        try:
            current = trust.current_fingerprint_digest(name)
        except Exception:  # noqa: BLE001 -- uncertainty revokes old authority
            current = None
        if current != expected:
            raise _owner.SkillTrustBlockedError(
                skill_name=name,
                reason_code="skill_definition_changed",
                trust_status="quarantined_modified",
            )

    context = (
        frozen_context
        if frozen_context.get("backend") == "local"
        else await self._skills_service.get_context(mode="local")
    )
    candidates = self._skill_candidates_from_context(context)
    # DETECTION population = trusted candidates UNION user-invocable
    # blocked (needs-review) skills. A blocked skill must still be
    # DETECTED -- leading refuses, embedded degrades to literal + note
    # -- rather than silently staying plain, sigil-prefixed text with no
    # signal at all. `execute_skill` (not this resolution step) remains
    # the sole authority on whether a resolved name may actually run:
    # a name that resolves here to a blocked skill hits
    # `SkillTrustBlockedError` at the `execute_skill` call below/in the
    # embedded loop, which already drives the refuse/skip-with-note
    # paths.
    detection_candidates = candidates + self._skill_blocked_candidates_from_context(
        context
    )

    # --- Leading form: message starts with a resolvable $skill-name.
    # Leading whitespace is tolerated (stripped before the sigil check
    # and the word/rest split) to match `_annotate_skill_commands`'s own
    # `lstrip()` assumption in the preview -- a resolved leading mention
    # replaces the ENTIRE message on both the inline-replace and fork
    # paths, so the leading whitespace simply disappears either way.
    stripped_content = source_content.lstrip()
    if stripped_content.startswith(_owner.MENTION_SIGIL):
        word, rest = _owner._split_skill_command_word(stripped_content)
        name = word[len(_owner.MENTION_SIGIL) :]
        if name:
            resolution = _owner.resolve_skill_command(name, rest, detection_candidates)
            if resolution.kind == "resolved":
                args = (
                    rest
                    if plugin_entries.get(resolution.name, {}).get("plugin_kind")
                    == "command"
                    else _owner.cap_skill_args(rest)
                )
                try:
                    ensure_frozen_definition(resolution.name)
                    result = await self._skills_service.execute_skill(
                        resolution.name,
                        mode="local",
                        args=args,
                        **plugin_options(resolution.name),
                    )
                except _owner.SkillTrustBlockedError as exc:
                    refuse = _owner.SKILL_UNTRUSTED_REFUSE.format(
                        name=resolution.name, reason=exc.reason_code
                    )
                    return provider_messages, refuse, (), (), ""

                rendered = (
                    result.get("rendered_prompt", "")
                    if isinstance(result, _owner.Mapping)
                    else ""
                )
                rendered_message = {
                    "role": _owner.ConsoleMessageRole.USER.value,
                    "content": rendered,
                }
                execution_mode = (
                    result.get("execution_mode")
                    if isinstance(result, _owner.Mapping)
                    else None
                )
                # Task 5: a resolved leading mention always binds its
                # name (fork AND inline outcomes -- never on refuse,
                # which already returned above) and its block is
                # rendered from this single execute_skill result.
                bindings = (resolution.name,)
                block = (
                    _owner._render_skill_bundle_block([result])
                    if isinstance(result, _owner.Mapping)
                    else ""
                )
                if execution_mode == "fork":
                    leading = (
                        [provider_messages[0]]
                        if provider_messages
                        and provider_messages[0].get("role")
                        == _owner.ConsoleMessageRole.SYSTEM.value
                        else []
                    )
                    return leading + [rendered_message], None, (), bindings, block

                new_messages = list(provider_messages)
                new_messages[final_index] = {
                    **provider_messages[final_index],
                    "content": rendered,
                }
                return new_messages, None, (), bindings, block

    # --- Embedded pass: no leading mention, or the leading word didn't
    # resolve to a known skill. Scans the ORIGINAL (unstripped) content
    # -- the leading-whitespace tolerance above only applies to the
    # leading form. `names` is the same detection population (trusted
    # UNION user-invocable blocked) so a blocked mention is found and
    # routed through the trust-blocked-note path below instead of
    # staying invisible.
    names = frozenset(candidate.name for candidate in detection_candidates)
    mentions = _owner.find_embedded_mentions(source_content, names)
    if not mentions:
        return provider_messages, None, (), (), ""

    rendered_by_name: dict[str, str | None] = {}
    # Task 5: results_by_name only keeps a name's execute_skill result
    # when that mention actually SPLICED (execution_mode == "inline")
    # -- a blocked-literal or fork-literal mention's result is
    # discarded here, so it can never leak into skill_bindings or the
    # rendered bundle block below.
    results_by_name: dict[str, _owner.Mapping[str, Any]] = {}
    notes: list[str] = []
    for mention in mentions:
        if mention.name in rendered_by_name:
            continue
        try:
            ensure_frozen_definition(mention.name)
            result = await self._skills_service.execute_skill(
                mention.name,
                mode="local",
                args="",
                **plugin_options(mention.name),
            )
        except _owner.SkillTrustBlockedError:
            rendered_by_name[mention.name] = None
            notes.append(_owner.SKILL_MENTION_SKIPPED_NOTE.format(name=mention.name))
            continue
        execution_mode = (
            result.get("execution_mode") if isinstance(result, _owner.Mapping) else None
        )
        rendered = (
            result.get("rendered_prompt", "")
            if isinstance(result, _owner.Mapping)
            else ""
        )
        # Fork (or anything non-inline) cannot splice in place: leave
        # the mention literal, no note (this is not a trust failure).
        rendered_by_name[mention.name] = (
            rendered if execution_mode == "inline" else None
        )
        if execution_mode == "inline" and isinstance(result, _owner.Mapping):
            results_by_name[mention.name] = result

    if plugin_entries:
        from tldw_chatbook.Plugins.context import check_context_budget

        try:
            check_context_budget(
                [
                    rendered_by_name[mention.name]
                    for mention in mentions
                    if mention.name in plugin_entries
                    and rendered_by_name.get(mention.name) is not None
                ]
            )
            await plugin_service.check_entries(plugin_entries.values())
        except PermissionError:
            return (
                provider_messages,
                "Plugin context is unavailable or too large; review selected skills.",
                (),
                (),
                "",
            )
    new_content = content
    for mention in reversed(mentions):
        body = rendered_by_name.get(mention.name)
        if body is None:
            continue
        start = mention.start + sum(
            size for end, size in reference_offsets if end <= mention.start
        )
        end = mention.end + sum(
            size for offset, size in reference_offsets if offset <= mention.end
        )
        new_content = new_content[:start] + body + new_content[end:]
    if new_content == content:
        return provider_messages, None, tuple(notes), (), ""
    if plugin_entries:
        from tldw_chatbook.Agents.agent_models import carry_plugin_context

        new_content = carry_plugin_context(
            new_content,
            *(
                rendered_by_name[mention.name]
                for mention in mentions
                if rendered_by_name.get(mention.name) is not None
            ),
        )

    # Task 5: bound names are every unique mention that actually
    # spliced, in first-occurrence document order (`dict.fromkeys` on
    # `mentions` dedups while preserving order) -- never a
    # blocked-literal or fork-literal mention, which never reached
    # `results_by_name`.
    spliced_names = tuple(
        name
        for name in dict.fromkeys(mention.name for mention in mentions)
        if rendered_by_name.get(name) is not None
    )
    block = _owner._render_skill_bundle_block(
        results_by_name[name] for name in spliced_names if name in results_by_name
    )
    new_messages = list(provider_messages)
    new_messages[final_index] = {
        **provider_messages[final_index],
        "content": new_content,
    }
    return new_messages, None, tuple(notes), spliced_names, block


from .console_ordinary_tree import _function_record as _native_record

_NATIVE_WORKFLOW_RECEIVERS = (
    ("_prepare_native_references", _native_record(_prepare_native_references)),
    ("_native_selector", _native_record(_native_selector)),
)

from . import console_references as _native_refs

_NATIVE_REFERENCE_PARSER = (
    _native_record(_native_refs._native_reference_candidates),
    _native_refs._REFERENCE_RE,
    _native_refs.ReferenceCandidate,
    _native_record(_native_refs.ReferenceCandidate.__init__),
)
