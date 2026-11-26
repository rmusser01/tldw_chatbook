-- TASK-42544: drop v56's partial idx_console_trace_response_revision. v72's
-- full idx_console_trace_response_links_semantic_revision_exact covers the
-- same column, and with no sqlite_stat1 it wins every product query on it
-- (the exact locator, the semantic-revision policy join, the voice-promotion
-- locator check). The partial index served only a `semantic_revision_id IS
-- NOT NULL` range scan no product query issues, yet every response-link
-- write maintained it.
DROP INDEX idx_console_trace_response_revision;
