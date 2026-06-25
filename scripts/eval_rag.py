#!/usr/bin/env python3
"""Run RAG quality evaluation against Phoenix.

Usage
-----
    uv run python scripts/eval_rag.py

Requires
--------
- Phoenix running at ``PHOENIX_COLLECTOR_ENDPOINT`` (default http://localhost:6006/v1/traces)
- ``OPENAI_API_KEY`` set for LLM-based evaluators
- Standard env config (``.env`` or env vars) for the RAG pipeline
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``app`` is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import get_settings  # noqa: E402 — sys.path is set above
from app.eval import run_evaluation  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()

    phoenix_url = settings.phoenix_collector_endpoint.replace("/v1/traces", "")
    logger.info("Connecting to Phoenix at %s …", phoenix_url)

    if not settings.phoenix_enabled:
        logger.warning("PHOENIX_ENABLED is false — evaluation will not be recorded.")
        sys.exit(1)

    try:
        summary = run_evaluation(settings)
    except RuntimeError as exc:
        logger.error("❌ %s", exc)
        sys.exit(1)

    evaluated = summary["evaluated"]
    logger.info("✅ Evaluation complete — %d test case(s) processed.", evaluated)
    logger.info("   Experiment ID: %s", summary.get("experiment_id", "?"))

    scores = summary.get("scores", {})
    for name, data in scores.items():
        avg = data.get("avg", "?")
        vals = data.get("values", [])
        logger.info("   %-22s  avg=%-6s  n=%d  values=%s", name, avg, len(vals), vals)

    logger.info("\n📊 View results at %s", phoenix_url)


if __name__ == "__main__":
    main()
