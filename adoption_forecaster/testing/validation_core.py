"""
Core Validation logic - shared between local and GA use cases

This module contains pure comparison functions that can be used by both
local and GitHub Actions validation scripts.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from adoption_forecaster.config.logging_config import get_logger

logger = get_logger(__name__)


def get_output_files(directory):
    """Get all CSV output files from a directory (supports Education and Age_Gender subdirectories)"""
    base_dir = Path(directory)

    if not base_dir.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    all_files = []

    for module in ["Education", "Age_Gender"]:
        module_dir = base_dir / module
        if module_dir.exists():
            for csv_file in module_dir.glob("*.csv"):
                all_files.append(csv_file)
            output_dir = module_dir / "output"
            if output_dir.exists():
                for csv_file in output_dir.glob("*.csv"):
                    all_files.append(csv_file)
        else:
            for csv_file in base_dir.glob(f"{module}_*.csv"):
                all_files.append(csv_file)

    if not all_files:
        for csv_file in base_dir.glob("*.csv"):
            all_files.append(csv_file)
        for csv_file in base_dir.rglob("*.csv"):
            if csv_file not in all_files:
                all_files.append(csv_file)

    logger.info(f"Found {len(all_files)} output files in {directory}: {[f.name for f in all_files]}")
    return all_files


def compare_two_files(file1, file2, tolerance=1e-4):
    """Compare two files and return True if they are the same"""
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        if not df1.shape == df2.shape:
            return False

        df1_numeric = df1.apply(pd.to_numeric, errors="coerce")
        df2_numeric = df2.apply(pd.to_numeric, errors="coerce")

        # Use tolerance-based comparison
        return np.allclose(df1_numeric, df2_numeric, rtol=tolerance, atol=tolerance, equal_nan=True)

    except Exception as e:
        logger.error(f"Error comparing files: {e}")
        return False


def show_file_diff(file1, file2, file_name, pred_year, tolerance=1e-4):
    """Show detailed differences between two files"""
    logger.info(f"\n--- Differences in {file_name} ---")
    logger.info(f"Using tolerance: {tolerance}")
    logger.info(f"Comparing prediction years: {pred_year-1} and {pred_year}")

    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        logger.info(f"Shape comparison: {df1.shape} vs {df2.shape}")

        if not df1.shape == df2.shape:
            logger.error("Different shapes!")

        logger.info("Converting columns to numeric...")
        df1_numeric = df1.apply(pd.to_numeric, errors="coerce")
        df2_numeric = df2.apply(pd.to_numeric, errors="coerce")

        pred_cols = []
        for col in df1.columns:
            if (col.startswith(str(pred_year - 1)) or col.startswith(str(pred_year))) and pd.api.types.is_numeric_dtype(df1_numeric[col]):
                pred_cols.append(col)

        existing_pred_cols = [col for col in pred_cols if col in df1.columns and col in df2.columns]

        if not existing_pred_cols:
            logger.warning(f"No prediction columns found for years {pred_year-1} and {pred_year}")
            logger.info(f"Available columns: {list(df1.columns)}")
            return

        logger.info(f"Comparing prediction columns: {existing_pred_cols}")

        for col in existing_pred_cols:
            logger.info(f"Checking column '{col}'")

            # Use tolerance-based comparison
            are_close = np.isclose(df1_numeric[col], df2_numeric[col], rtol=tolerance, atol=tolerance, equal_nan=True)

            if not are_close.all():
                logger.error(f"Column '{col}' differs (tolerance: {tolerance})")
                diff_count = (~are_close).sum()
                logger.info(f"   {diff_count} values differ out of {len(df1)}")

                diff_mask = ~are_close
                diff_indices = df1_numeric[diff_mask].index[:3]  # Get first 3 indices

                if len(diff_indices) > 0:
                    logger.info("   Sample differences:")
                    for idx in diff_indices:
                        val1 = df1_numeric.loc[idx, col]
                        val2 = df2_numeric.loc[idx, col]
                        actual_diff = abs(val1 - val2)
                        logger.info(f"     Row {idx}: {val1} vs {val2} (diff: {actual_diff})")
            else:
                logger.info(f"Column '{col}' identical (within tolerance)")

        # Check for missing prediction columns
        missing_cols = [col for col in pred_cols if col not in df1.columns or col not in df2.columns]
        if missing_cols:
            logger.warning(f"Missing prediction columns: {missing_cols}")

    except Exception as e:
        logger.error(f"Error comparing CSV files: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")


def validate_outputs_from_dirs(base_dir, compare_dir, pred_year=2026, show_diffs=False):
    """Main validation function - compares outputs from two directories"""
    logger.info(f"Validating outputs: {base_dir} vs {compare_dir}")
    logger.info(f"Comparing prediction years: {pred_year-1} and {pred_year}")

    base_dir_path = Path(base_dir)
    compare_dir_path = Path(compare_dir)

    if not base_dir_path.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return False

    if not compare_dir_path.exists():
        logger.error(f"Compare directory does not exist: {compare_dir}")
        return False

    # Get all output files from both directories
    base_files = get_output_files(base_dir)
    compare_files = get_output_files(compare_dir)

    if not base_files:
        logger.error(f"No output files found in base directory: {base_dir}")
        return False

    if not compare_files:
        logger.error(f"No output files found in compare directory: {compare_dir}")
        return False

    logger.info(f"Found {len(base_files)} files in base directory")
    logger.info(f"Found {len(compare_files)} files in compare directory")

    # Create a mapping of base files to compare files based on filename
    base_file_map = {}

    for base_file in base_files:
        base_name = base_file.name
        found_match = False

        for compare_file in compare_files:
            if compare_file.name == base_name:
                base_file_map[base_file] = compare_file
                logger.info(f"Matched: {base_name}")
                found_match = True
                break

        if not found_match:
            logger.warning(f"No match found for base file: {base_name}")
            logger.warning(f"  Available compare files: {[f.name for f in compare_files]}")

    logger.info(f"Created {len(base_file_map)} file pairs for comparison")

    if not base_file_map:
        logger.error("No matching files found between directories!")
        return False

    # Compare each pair of files
    all_same = True
    for base_file, compare_file in base_file_map.items():
        file_name = base_file.name

        logger.info(f"Comparing: {file_name}")

        if compare_two_files(base_file, compare_file):
            logger.info(f"{file_name} - IDENTICAL")
        else:
            logger.error(f"{file_name} - DIFFERENT!")
            all_same = False

            if show_diffs:
                show_file_diff(base_file, compare_file, file_name, pred_year)

    if all_same:
        logger.info("All output files are identical!")
    else:
        logger.warning("WARNING: Some output files are different!")
        return False

    return True
