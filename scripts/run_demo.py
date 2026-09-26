#!/usr/bin/env python3
"""Run the allowlisted InSite engine against fictional synthetic timelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PORTFOLIO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PORTFOLIO_ROOT / "src"
# Prefer this checked-in snapshot even if a developer has another
# insite_analytics checkout installed or present in the working directory.
sys.path.insert(0, str(SRC_ROOT))

from insite_analytics.daily_brief import BriefConfig, build_daily_brief  # noqa: E402
from insite_analytics.personal_context_synthetic import (  # noqa: E402
    TIMEZONE,
    build_personal_context_fixture,
)


def _assert_snapshot_import() -> None:
    import insite_analytics

    package_file = Path(insite_analytics.__file__).resolve()
    if not package_file.is_relative_to(SRC_ROOT.resolve()):
        raise RuntimeError("the demo imported insite_analytics outside its source snapshot")


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _quick_config() -> BriefConfig:
    """Small smoke settings; these do not preserve benchmark power."""
    return BriefConfig(
        v3_min_units=3,
        v3_block_days=1,
        v3_max_candidates=2,
        v3_search_depth=1,
        v3_search_bins=3,
        v3_search_beam_width=12,
        v3_search_shortlist=8,
        v3_window_widths=(3,),
        v3_analogue_neighbors=4,
        v3_min_analogue_neighbors=2,
        v3_bootstrap_draws=40,
    )


def _run_one(*, scenario: str, days: int, seed: int, config: BriefConfig) -> dict[str, Any]:
    records, truth = build_personal_context_fixture(days=days, seed=seed, scenario=scenario)
    brief = build_daily_brief(
        records,
        as_of=truth["asOf"],
        timezone=TIMEZONE,
        config=config,
    )
    # Keep no source observations or truth manifest in the output. The engine
    # document itself contains only analysis results and synthetic evidence.
    return brief


def _save(brief: dict[str, Any], *, scenario: str, days: int, seed: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"personal-context-{scenario}-{days}d-seed{seed}.json"
    payload = json.dumps(
        brief,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )
    destination.write_text(payload + "\n", encoding="utf-8")
    return destination


def _summarize(brief: dict[str, Any], *, mode: str, scenario: str, days: int, seed: int, destination: Path) -> None:
    counts = brief.get("analysisDiagnostics", {})
    print(
        f"mode={mode} scenario={scenario} days={days} seed={seed} "
        f"status={brief.get('status', 'unknown')} "
        f"items={len(brief.get('items', []))} evidence={len(brief.get('evidence', {}))} "
        f"accepted_records={counts.get('acceptedRecords', 'unknown')} "
        f"output={destination.parent.name}/{destination.name}"
    )


def run_quick(output_dir: Path) -> None:
    days, seed, scenario = 14, 131, "full"
    brief = _run_one(scenario=scenario, days=days, seed=seed, config=_quick_config())
    destination = _save(brief, scenario=scenario, days=days, seed=seed, output_dir=output_dir)
    _summarize(brief, mode="quick-smoke", scenario=scenario, days=days, seed=seed, destination=destination)


def run_full(output_dir: Path) -> None:
    seed, days = 131, 120
    for scenario in ("full", "null_clock"):
        brief = _run_one(scenario=scenario, days=days, seed=seed, config=BriefConfig())
        destination = _save(brief, scenario=scenario, days=days, seed=seed, output_dir=output_dir)
        _summarize(brief, mode="full-default-config", scenario=scenario, days=days, seed=seed, destination=destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("quick", "full"),
        nargs="?",
        default="quick",
        help="quick is a reduced smoke run; full runs the 120-day seed-131 scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PORTFOLIO_ROOT / "outputs",
        help="local directory for synthetic engine evidence JSON (default: portfolio outputs/)",
    )
    args = parser.parse_args(argv)
    _assert_snapshot_import()
    if args.mode == "quick":
        run_quick(args.output_dir)
    else:
        run_full(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
