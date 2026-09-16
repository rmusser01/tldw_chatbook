"""Run recovery without ordinary application startup."""

from .launcher import recovery_main

if __name__ == "__main__":
    raise SystemExit(recovery_main())
