# Verification refinement

The first native pair used `table.render_line()` for its numeric alignment check.
Its compact baseline records reported alignment even though the saved terminal and
SVG still painted the old header. The first pair's terminal comparisons isolated
the difference to the header row in all four views. Tests and native QA now read
the composed screen strips instead. All four stronger tests fail on merged dev;
the exact same final native runner reports all four baseline views misaligned and
all four fixed views aligned. The final fixed PNGs are pixel-identical to the
first fixed captures that were visually inspected. No product change was needed
after the original cache repair.

The native baseline's exit 1 means the expected alignment failure; its app itself
returns normally with code 0, and its lifecycle checks pass. Both final runs use
the same runner/journey hashes. Quick Look previews were used for visual inspection;
the earlier Cairo attempt substituted fonts and was excluded from visual evidence.
