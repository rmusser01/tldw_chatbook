-- ChaChaNotes v86 -> v87: trace segments owned by a spawned child run (TASK-41310).
--
-- A spawned child's provider calls belong to the parent's owner and turn, but
-- the child's request history is its own. Each child run gets one root-less
-- segment, linked here to the parent's owner by the child's run identity. The
-- owner guards below resolve a linked segment to that owner's root lineage, so
-- a child segment is owned exactly like its owner's root. GC keeps a link (and
-- its segment) while the owner is rooted, and collects it after detachment.
--
-- Guarantees held in SQL: the owner is attached, the segment is empty and
-- root-less (never an owner root, fork parent or fork child), and every call on
-- the segment carries the link's owner, turn and run. turn_id and run_id are
-- runtime identities with no table of their own, so they cannot be tied to the
-- owner's conversation here; the runtime derives them from the spawning run.

CREATE TABLE console_trace_child_segments(
  segment_id TEXT PRIMARY KEY NOT NULL
    REFERENCES console_trace_segments(segment_id),
  owner_id TEXT NOT NULL REFERENCES console_trace_owners(owner_id),
  turn_id TEXT NOT NULL CHECK(length(turn_id) > 0),
  run_id TEXT NOT NULL UNIQUE CHECK(length(run_id) > 0),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER console_trace_child_segments_insert_guard
BEFORE INSERT ON console_trace_child_segments
WHEN NOT EXISTS (
  SELECT 1 FROM console_trace_owners AS owner
   WHERE owner.owner_id = NEW.owner_id AND owner.attached = 1
) OR NOT EXISTS (
  SELECT 1 FROM console_trace_segments AS segment
   WHERE segment.segment_id = NEW.segment_id
     AND segment.parent_segment_id IS NULL
) OR EXISTS (
  SELECT 1 FROM console_trace_owners WHERE root_segment_id = NEW.segment_id
) OR EXISTS (
  SELECT 1 FROM console_trace_segments WHERE parent_segment_id = NEW.segment_id
) OR EXISTS (
  SELECT 1 FROM console_trace_surface_nodes WHERE segment_id = NEW.segment_id
) OR EXISTS (
  SELECT 1 FROM console_trace_calls WHERE segment_id = NEW.segment_id
) OR EXISTS (
  SELECT 1 FROM console_trace_events WHERE segment_id = NEW.segment_id
)
BEGIN
  SELECT RAISE(
    ABORT,
    'child segment must link an empty root-less segment to an active owner'
  );
END;

CREATE TRIGGER console_trace_child_segments_no_update
BEFORE UPDATE ON console_trace_child_segments BEGIN
  SELECT RAISE(ABORT, 'console_trace_child_segments is append-only');
END;

CREATE TRIGGER console_trace_child_segments_no_delete
BEFORE DELETE ON console_trace_child_segments
WHEN console_trace_gc_delete_authorized('console_trace_child_segments') <> 1
BEGIN SELECT RAISE(ABORT, 'trace GC deletion authorization required'); END;

-- Owner guards: a linked child segment resolves to its owner's root lineage.

DROP TRIGGER console_trace_calls_owner_lineage;
CREATE TRIGGER console_trace_calls_owner_lineage
BEFORE INSERT ON console_trace_calls
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT COALESCE((
      SELECT owner.root_segment_id
        FROM console_trace_child_segments AS link
        JOIN console_trace_owners AS owner
          ON owner.owner_id = link.owner_id
       WHERE link.segment_id = NEW.segment_id
    ), NEW.segment_id), 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.owner_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM effective_owner AS owner
   WHERE owner.owner_id = NEW.owner_id
     AND owner.attached = 1
) OR EXISTS (
  SELECT 1 FROM console_trace_child_segments AS link
   WHERE link.segment_id = NEW.segment_id
     AND (link.turn_id <> NEW.turn_id OR link.run_id <> NEW.run_id)
)
BEGIN
  SELECT RAISE(ABORT, 'provider call must use the active effective owner');
END;

DROP TRIGGER console_trace_events_owner_guard;
CREATE TRIGGER console_trace_events_owner_guard
BEFORE INSERT ON console_trace_events
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT COALESCE((
      SELECT owner.root_segment_id
        FROM console_trace_child_segments AS link
        JOIN console_trace_owners AS owner
          ON owner.owner_id = link.owner_id
       WHERE link.segment_id = NEW.segment_id
    ), NEW.segment_id), 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.conversation_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM effective_owner AS owner
   WHERE owner.attached = 1
     AND (
       NEW.semantic_revision_id IS NULL OR EXISTS (
         SELECT 1
           FROM console_trace_semantic_revisions AS revision
          WHERE revision.revision_id = NEW.semantic_revision_id
            AND revision.source_conversation_id = owner.conversation_id
       )
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'trace event requires one active effective owner and matching revision domain'
  );
END;

DROP TRIGGER console_trace_surface_nodes_owner_guard;
CREATE TRIGGER console_trace_surface_nodes_owner_guard
BEFORE INSERT ON console_trace_surface_nodes
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT COALESCE((
      SELECT owner.root_segment_id
        FROM console_trace_child_segments AS link
        JOIN console_trace_owners AS owner
          ON owner.owner_id = link.owner_id
       WHERE link.segment_id = NEW.segment_id
    ), NEW.segment_id), 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.conversation_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM effective_owner AS owner
   WHERE owner.attached = 1
     AND (
       NEW.semantic_revision_id IS NULL OR EXISTS (
         SELECT 1
           FROM console_trace_semantic_revisions AS revision
          WHERE revision.revision_id = NEW.semantic_revision_id
            AND revision.source_conversation_id = owner.conversation_id
       )
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'surface node requires one active effective owner and matching revision domain'
  );
END;

DROP TRIGGER console_trace_surface_replacements_owner_guard;
CREATE TRIGGER console_trace_surface_replacements_owner_guard
BEFORE INSERT ON console_trace_surface_replacements
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT COALESCE((
      SELECT owner.root_segment_id
        FROM console_trace_child_segments AS link
        JOIN console_trace_owners AS owner
          ON owner.owner_id = link.owner_id
       WHERE link.segment_id = NEW.segment_id
    ), NEW.segment_id), 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1 FROM effective_owner WHERE attached = 1
)
BEGIN
  SELECT RAISE(
    ABORT,
    'surface replacement requires one active effective owner'
  );
END;

DROP TRIGGER console_trace_response_links_owner_guard;
CREATE TRIGGER console_trace_response_links_owner_guard
BEFORE INSERT ON console_trace_response_links
WHEN NOT EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT COALESCE((
      SELECT owner.root_segment_id
        FROM console_trace_child_segments AS link
        JOIN console_trace_owners AS owner
          ON owner.owner_id = link.owner_id
       WHERE link.segment_id = call.segment_id
    ), call.segment_id), 0
      FROM console_trace_calls AS call
     WHERE call.call_id = NEW.call_id
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.owner_id, owner.conversation_id, owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1
    FROM console_trace_calls AS call
    JOIN effective_owner AS owner
      ON owner.owner_id = call.owner_id
   WHERE call.call_id = NEW.call_id
     AND owner.attached = 1
     AND (
       NEW.semantic_revision_id IS NULL OR EXISTS (
         SELECT 1
           FROM console_trace_semantic_revisions AS revision
          WHERE revision.revision_id = NEW.semantic_revision_id
            AND revision.source_conversation_id = owner.conversation_id
       )
     )
)
BEGIN
  SELECT RAISE(
    ABORT,
    'response link requires one active effective owner and matching revision domain'
  );
END;

-- A linked child segment can never become an owner root or a fork parent.
DROP TRIGGER console_trace_owners_empty_root_guard;
CREATE TRIGGER console_trace_owners_empty_root_guard
BEFORE INSERT ON console_trace_owners
WHEN
  EXISTS (
    SELECT 1 FROM console_trace_surface_nodes
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_surface_replacements
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_calls
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_events
     WHERE segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_segments
     WHERE parent_segment_id = NEW.root_segment_id
  ) OR
  EXISTS (
    SELECT 1 FROM console_trace_child_segments
     WHERE segment_id = NEW.root_segment_id
  )
BEGIN
  SELECT RAISE(ABORT, 'trace owner must attach to a distinct empty root segment');
END;

DROP TRIGGER console_trace_segments_parent_owner_guard;
CREATE TRIGGER console_trace_segments_parent_owner_guard
BEFORE INSERT ON console_trace_segments
WHEN NEW.parent_segment_id IS NOT NULL AND (
  EXISTS (
    SELECT 1 FROM console_trace_child_segments
     WHERE segment_id = NEW.parent_segment_id
  ) OR EXISTS (
  WITH RECURSIVE segment_ancestry(segment_id, depth) AS (
    SELECT NEW.parent_segment_id, 0
    UNION ALL
    SELECT segment.parent_segment_id, child.depth + 1
      FROM console_trace_segments AS segment
      JOIN segment_ancestry AS child
        ON child.segment_id = segment.segment_id
     WHERE segment.parent_segment_id IS NOT NULL
  ),
  effective_owner AS (
    SELECT owner.attached
      FROM segment_ancestry AS owned
      JOIN console_trace_owners AS owner
        ON owner.root_segment_id = owned.segment_id
     ORDER BY owned.depth
     LIMIT 1
  )
  SELECT 1 FROM effective_owner WHERE attached = 0
))
BEGIN
  SELECT RAISE(ABORT, 'child segment cannot extend a detached owner prefix');
END;
