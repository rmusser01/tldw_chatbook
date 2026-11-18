"""Explicit local scope precedence for immutable native rule revisions."""

from .models import RuleBinding, RuleRevision, RuleScope


def resolve_effective_rules(
    bindings: tuple[RuleBinding, ...],
    revisions: tuple[RuleRevision, ...],
    *,
    chat: RuleScope,
    workspace: RuleScope | None,
    global_scope: RuleScope,
) -> tuple[RuleRevision, ...]:
    """Select each logical rule once; exclusions mask inherited revisions."""
    by_revision = {(r.rule_id, r.revision): r for r in revisions}
    selected: dict[str, RuleBinding] = {}
    for scope in (chat, workspace, global_scope):
        if scope is None:
            continue
        for binding in bindings:
            if binding.scope == scope and binding.rule_id not in selected:
                selected[binding.rule_id] = binding
    effective = []
    for rule_id in sorted(selected):
        binding = selected[rule_id]
        if binding.state != "enabled" or binding.revision is None:
            continue
        rule = by_revision.get((rule_id, binding.revision))
        if rule is not None:
            effective.append(rule)
    return tuple(effective)
