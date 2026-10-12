"""Local Guardian self-monitoring subsystem (ADR-204).

Pattern rules over the user's own typed Console prompts, humane awareness
notices with escalation, crisis resources, and (Task 2+) trend analysis.
Off by default with zero footprint when disabled.

This package's modules are imported lazily by their call sites (ADR-097
boot-census ratchet): nothing here is imported at app module scope.
"""
