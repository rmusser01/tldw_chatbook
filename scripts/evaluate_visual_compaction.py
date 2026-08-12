"""Run a confirmation-gated live visual-compaction model evaluation.

The command makes exactly two provider requests (text baseline and visual
transcript). Its output contains only synthetic corpus identity, hashes,
aggregate scores, provider usage, and latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = (
    REPOSITORY_ROOT
    / "Docs"
    / "superpowers"
    / "qa"
    / "visual-compaction-model-evaluation"
    / "corpus-v1.json"
)
DEFAULT_OUTPUT = DEFAULT_CORPUS.with_name("support-matrix.json")
# Keep parser choices as inert literals so --help/refusal cannot import app modules.
RENDERER_PROFILE_CHOICES = ("production_1024", "native_512_candidate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--renderer-profile",
        choices=RENDERER_PROFILE_CHOICES,
        default="production_1024",
        help=(
            "Select the versioned deterministic renderer. The native 512x512 "
            "candidate is evaluation-only until measured ADR-056 gates pass."
        ),
    )
    parser.add_argument(
        "--confirm-billable",
        action="store_true",
        help="Confirm exactly two provider requests may incur usage charges.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace this provider/model's existing support-matrix report.",
    )
    return parser


def validate_live_request(args: argparse.Namespace) -> Path:
    """Validate destructive and billable boundaries before loading app config."""

    if not args.confirm_billable:
        raise ValueError(
            "Refusing provider calls without --confirm-billable; this run makes "
            "exactly two potentially charged requests."
        )
    if not 1 <= args.max_output_tokens <= 16_384:
        raise ValueError("--max-output-tokens must be between 1 and 16384.")
    output = args.output.resolve()
    if output.exists():
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
            reports = data["reports"]
            if not isinstance(reports, list) or not all(
                isinstance(report, dict)
                and isinstance(report.get("provider"), str)
                and isinstance(report.get("model"), str)
                for report in reports
            ):
                raise TypeError("reports must identify provider/model pairs")
            identities = {(report["provider"], report["model"]) for report in reports}
        except (
            KeyError,
            OSError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "Existing output is not a readable visual support matrix."
            ) from exc
        if (args.provider, args.model) in identities and not args.replace:
            raise FileExistsError(
                "The support matrix already contains this provider/model. "
                "Pass --replace to rerun and replace that report."
            )
    return output


async def run_live(args: argparse.Namespace) -> int:
    output = validate_live_request(args)

    # Keep application imports behind the explicit charge/overwrite boundary.
    # Importing the package initializes config paths, so even ``--help`` and a
    # refused billable run must reach neither this point nor the user profile.
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_visual_evaluation import (
        build_visual_support_matrix,
        evaluate_visual_compaction_model,
        load_visual_evaluation_corpus,
        load_visual_support_matrix,
        resolve_visual_evaluation_model,
    )
    from tldw_chatbook.Chat.console_visual_transcript import (
        resolve_evaluation_renderer_profile,
    )
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.model_capabilities import get_model_capabilities
    from tldw_chatbook.Utils.atomic_file_ops import atomic_write_text

    # The parser choices intentionally remain inert literals. Re-resolve after
    # the billable/overwrite guard so a stale CLI allowlist cannot reach a call.
    resolve_evaluation_renderer_profile(args.renderer_profile)
    corpus = load_visual_evaluation_corpus(args.corpus)
    config = load_settings(force_reload=True)
    gateway = ConsoleProviderGateway(config_provider=lambda: config)
    try:
        resolution = await resolve_visual_evaluation_model(
            gateway=gateway,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            max_output_tokens=args.max_output_tokens,
        )
        if not resolution.ready:
            raise RuntimeError(resolution.visible_copy or "Provider is not ready.")
        capabilities = get_model_capabilities().get_model_capabilities(
            resolution.provider,
            resolution.model or "",
        )
        max_images = capabilities.get("max_images")
        report = await evaluate_visual_compaction_model(
            gateway=gateway,
            resolution=resolution,
            corpus=corpus,
            evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
            vision_available=capabilities.get("vision") is True,
            max_images=(
                max_images
                if isinstance(max_images, int) and not isinstance(max_images, bool)
                else 0
            ),
            max_output_tokens=args.max_output_tokens,
            renderer_profile_id=args.renderer_profile,
        )
    finally:
        await gateway.aclose()

    existing_reports = ()
    if output.exists():
        existing_reports = load_visual_support_matrix(output).reports
    retained_reports = tuple(
        existing
        for existing in existing_reports
        if (existing.provider, existing.model) != (report.provider, report.model)
    )
    matrix = build_visual_support_matrix((*retained_reports, report))
    atomic_write_text(output, matrix.to_json() + "\n", encoding="utf-8")
    measured = "measured" if report.measured_usage_complete else "estimated"
    print(
        f"Wrote {output}: {report.provider}/{report.model}, "
        f"{report.token_reduction_ratio:.1%} reduction ({measured}), "
        f"recommendation={report.recommendation}."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run_live(args))
    except (FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
