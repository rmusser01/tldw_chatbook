"""Video generation backends: contracts, registry, config, validation, worker.

Parallel package to :mod:`tldw_chatbook.Image_Generation` (ADR-044): video
fields (duration, fps, ratio, reference assets) are first-class here rather
than stretched onto the image contracts. Generated videos are ephemeral and
name-referenced -- this package only produces bytes; storage is the
VideoStore's concern (task-3401.4).
"""
