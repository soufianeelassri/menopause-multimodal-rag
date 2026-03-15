"""Export evaluation results to CSV for reporting.

Usage:
    python -m scripts.export_metrics --input evaluation_results.json --output metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def export_to_csv(input_path: Path, output_path: Path) -> None:
    """Export JSON evaluation results to CSV.

    Args:
        input_path: Path to evaluation_results.json.
        output_path: Path for output CSV file.
    """
    data = json.loads(input_path.read_text(encoding="utf-8"))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Aggregate metrics
        writer.writerow(["Metric", "Value"])
        for metric, value in data["aggregate_metrics"].items():
            writer.writerow([metric, f"{value:.4f}"])
        writer.writerow(["num_queries", data["num_queries"]])
        writer.writerow([])

        # By-type breakdown
        if data.get("by_type"):
            writer.writerow(["Query Type", "Count"])
            for qtype, info in data["by_type"].items():
                writer.writerow([qtype, info["count"]])

    print(f"Metrics exported to {output_path}")


def main() -> None:
    """Run metrics export from the command line."""
    parser = argparse.ArgumentParser(
        description="Export RAGAS evaluation results to CSV",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics.csv"),
        help="Path for output CSV file",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found. Run evaluation first.")
        return

    export_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
