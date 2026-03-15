"""CLI entry point for the document ingestion pipeline.

Usage:
    python -m scripts.run_ingestion [--pdf-dir PATH] [--skip-captioning]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline
from src.utils.logging import get_logger


def main() -> None:
    """Run the ingestion pipeline with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="MenoGuide Document Ingestion Pipeline"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing PDFs (default: data/raw/)",
    )
    parser.add_argument(
        "--skip-captioning",
        action="store_true",
        help="Skip table/image captioning (faster for testing)",
    )
    args = parser.parse_args()

    logger = get_logger("run_ingestion")
    logger.info("starting_ingestion_cli")

    pipeline = IngestionPipeline()
    stats = pipeline.run(
        pdf_dir=args.pdf_dir,
        skip_captioning=args.skip_captioning,
    )

    print("\n=== Ingestion Complete ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
