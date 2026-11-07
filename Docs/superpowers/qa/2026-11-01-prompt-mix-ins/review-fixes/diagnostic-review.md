# Diagnostic statement review before inventory regeneration

Compared all three changed owners using `--statements ... --since 2acbcd42f3e7b07fd122f5f6dc6b22e7385f10c5`; exact output is diagnostic-statements.txt.

Controller: one new constant Preview-unavailable message, with no formatted value; the old exception statement only re-indents. Keyword store: five changes remove keyword names, UUIDs, entity IDs and exception messages/tracebacks. Remaining fields are exception class names and the fixed Media/Prompt entity label from the two KeywordStoreSpec constructors. Prompts Interop: two new constant mix-in import outcome messages avoid source name/path/body; three ordinary-prompt diagnostics only re-indent around the new discriminator and remain explicitly legacy-unreviewed path candidates. No new sink, source environment surface, logging destination or secret/path/body interpolation is introduced. Existing ordinary-prompt path logging is not endorsed or repaired by this scope.

The inventory update acknowledges these intended, reviewed owner statements; no suppression or census weakening is used.
