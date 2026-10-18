"""Native invocation projection behind the current Console controller methods."""

import asyncio
import json
from dataclasses import replace


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
    _native_selector, _native_rest, native_kind = self._native_selector(
        clean_draft, configuration.skill_context_maximum.get("available_skills", ())
    )
    first_word, _ = _owner._split_skill_command_word(clean_draft.lstrip())
    native_literal_arguments = native_kind == "command" or any(
        first_word == _owner.MENTION_SIGIL + row["name"] and row.get("plugin_owned")
        for row in configuration.skill_context_maximum.get("available_skills", ())
    )
    # TASK-27021: expand @-references (files/folders/diff) into the text
    # the PROVIDER sees, before the one preparation construction below.
    # The user echo keeps the RAW draft; a compact system row records what
    # expanded/refused (26020 AC#6). Expansion failures never block the
    # send -- the raw draft goes through with a note. AGENT_WAKE drafts
    # are machine-composed and never expanded.
    executed_draft_text = (
        resumed_preparation.executed_draft
        if resumed_reference is not None
        else clean_draft
    )
    reference_records: tuple = ()
    reference_offsets = (
        resumed_reference.reference_offsets if resumed_reference is not None else ()
    )
    # Qodo #6 (PR #2313): a RESUMED preparation re-enters this seam with
    # the already-expanded executed_draft; expansion is not idempotent
    # (each raw @token survives ahead of its inserted block), so re-running
    # it would inject every referenced file a second time. Expand only on
    # the first pass.
    if (
        not origin.automatic
        and machine_input is None
        and resumed_preparation is None
        and not native_literal_arguments
    ):
        try:
            from tldw_chatbook.Chat.console_references import (
                ReferenceExpansion,
                build_console_reference_resolver,
                expand_references,
                find_reference_candidates,
                run_git_reference,
            )

            candidates = find_reference_candidates(clean_draft)
            if native_kind == "rule":
                start = len(clean_draft) - len(clean_draft.lstrip())
                candidates = [
                    candidate
                    for candidate in candidates
                    if not (candidate.start == start and candidate.raw == first_word)
                ]
            if candidates:

                def _expand():
                    resolver = build_console_reference_resolver()
                    chunks, records, offsets = [], [], []
                    previous = 0
                    for candidate in candidates:
                        expansion = expand_references(
                            candidate.raw,
                            resolve=resolver,
                            git_runner=run_git_reference,
                        )
                        chunks.extend(
                            (
                                clean_draft[previous : candidate.start],
                                expansion.expanded_text,
                            )
                        )
                        records.extend(expansion.records)
                        offsets.append(
                            (
                                candidate.end,
                                len(expansion.expanded_text) - len(candidate.raw),
                            )
                        )
                        previous = candidate.end
                    chunks.append(clean_draft[previous:])
                    return ReferenceExpansion("".join(chunks), records), tuple(offsets)

                expansion, reference_offsets = await asyncio.to_thread(_expand)
                executed_draft_text = expansion.expanded_text
                reference_records = tuple(expansion.records)
        except Exception:  # noqa: BLE001 - references must never block a send
            _owner.logger.opt(exception=True).warning(
                "@-reference expansion failed; sending the raw draft"
            )
            executed_draft_text = clean_draft
            reference_records = ()
            reference_offsets = ()
    return clean_draft, executed_draft_text, reference_records, reference_offsets
