"""One core for the modality-shared media-generation machinery (ADR-176).

The Image_Generation and Video_Generation packages keep their public
surfaces, frozen contracts, and per-modality adapters; the skeleton they
mirrored -- the adapter registry, config machinery, and validation
helpers -- lives here once, parameterized by modality (ADR-176 records
why the workers and full validators stay modality-local).
"""
