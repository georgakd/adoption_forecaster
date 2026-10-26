#!/usr/bin/env python3
"""
Output Validation Script for GitHub Actions

This script compares output files between two directories to ensure no regressions.
It reads pre-generated outputs from two directories and compares them.

Usage:
    python outputs_validator_ga.py --base-dir compare-outputs/main --compare-dir compare-outputs/pr --pred-year 2026 --last-year 2024 --show-diffs
"""

import sys
import argparse

from adoption_forecaster.testing.validation_core import validate_outputs_from_dirs
from adoption_forecaster.config.logging_config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate output files between two directories (for GitHub Actions)")
    parser.add_argument("--base-dir", required=True, help="Base directory containing output files")
    parser.add_argument("--compare-dir", required=True, help="Compare directory containing output files")
    parser.add_argument("--pred-year", type=int, required=True, help="Prediction year to compare")
    parser.add_argument("--last-year", type=int, help="Last year (optional, for logging purposes)")
    parser.add_argument("--show-diffs", action="store_true", help="Show detailed differences")

    args = parser.parse_args()
    success = validate_outputs_from_dirs(args.base_dir, args.compare_dir, args.pred_year, args.show_diffs)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
