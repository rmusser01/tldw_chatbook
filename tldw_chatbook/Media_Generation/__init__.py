"""One core for the modality-shared media-generation machinery (ADR-176).

The Image_Generation and Video_Generation packages keep their public
surfaces, frozen contracts, and per-modality adapters; the skeleton they
mirrored -- registry, config machinery, worker, request validation --
lives here once, parameterized by modality.
"""
