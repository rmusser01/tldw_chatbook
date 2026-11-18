DROP INDEX idx_canvas_revisions_canvas_sequence;
CREATE INDEX idx_canvas_revisions_canvas_sequence
  ON canvas_revisions(canvas_id, sequence, id);
CREATE UNIQUE INDEX uq_canvas_revisions_single_root
  ON canvas_revisions(canvas_id) WHERE parent_revision_id IS NULL;
