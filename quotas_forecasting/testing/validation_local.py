#!/usr/bin/env python3
"""
Local Validation Script

This script compares output files between any two Git branches locally.
It runs predictions on both branches and compares the results.

Usage:
    python validate_local.py --base-branch main --compare-branch feature-branch --pred-year 2026 --last-year 2024 --show-diffs
    python validate_local.py --base-branch develop --compare-branch feature-branch --pred-year 2026 --last-year 2024 --show-diffs
    python validate_local.py --base-branch branch1 --compare-branch branch2 --pred-year 2026 --last-year 2024 --show-diffs
"""

import sys
import subprocess
import shutil
import argparse
from pathlib import Path

from quotas_forecasting.config.logging_config import get_logger
from quotas_forecasting.testing.validation_core import compare_two_files, show_file_diff

logger = get_logger(__name__)


def run_command(command, cwd=None, show_output=False):
    """Run command and return success status with output"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=cwd, check=False)
        if show_output and result.stderr:
            logger.error(f"Error output: {result.stderr}")
        if show_output and result.stdout:
            logger.info(f"Standard output: {result.stdout}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        if show_output:
            logger.error(f"Command failed: {e}")
        return False, "", str(e)


def get_current_branch():
    """Get current git branch"""
    success, stdout, stderr = run_command(["git", "branch", "--show-current"])
    if not success:
        logger.error(f"Error getting current branch: {stderr}")
        sys.exit(1)
    return stdout.strip()


def stash_changes():
    """Stash uncommitted changes"""
    success, stdout, stderr = run_command(["git", "status", "--porcelain"])
    if success and stdout.strip():
        logger.info("Stashing uncommitted changes...")
        success, _, stderr = run_command(["git", "stash", "push", "-m", "validation_stash"])
        if not success:
            logger.error(f"Error stashing changes: {stderr}")
            return False
    return True


def restore_changes():
    """Restore stashed changes"""
    success, stdout, stderr = run_command(["git", "stash", "list"])
    if success and "validation_stash" in stdout:
        logger.info("Restoring stashed changes...")
        success_pop, stdout_pop, stderr_pop = run_command(["git", "stash", "pop"])
        if not success_pop:
            logger.error(f"Error restoring stashed changes: {stderr_pop}")
            return False
        return True
    else:
        return True


def checkout_branch(branch):
    """Checkout branch"""
    logger.info(f"Checking out {branch}...")
    success, _, stderr = run_command(["git", "checkout", branch])
    if not success:
        logger.error(f"Error checking out {branch}: {stderr}")
        return False
    return True


def run_predictions(pred_year, last_year):
    """Run prediction pipeline"""
    logger.info(f"Running predictions (pred_year={pred_year}, last_year={last_year})...")
    success, stdout, stderr = run_command(
        [
            "python",
            "main.py",
            "--module",
            "both",
            "--mode",
            "local",
            "--pred-year",
            str(pred_year),
            "--last-year",
            str(last_year),
        ],
        show_output=False,
    )

    if not success:
        logger.error("Predictions failed!")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        return False

    logger.info("Predictions completed successfully")
    return True


def sanitize_branch_name(branch_name):
    """Sanitize branch name for use in file paths"""
    sanitized = branch_name.replace("/", "_")
    sanitized = sanitized.replace("\\", "_")
    sanitized = sanitized.replace("..", "_")
    return sanitized


def save_module_to_testing_folder(module_name, branch_suffix):
    """Save module outputs to testing folder with branch suffix"""
    source_dir = Path(f"{module_name}/output")
    testing_dir = Path(f"{module_name}/output/testing")

    if not source_dir.exists():
        logger.warning(f"Source directory {source_dir} does not exist")
        return []

    testing_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file_path in source_dir.glob("*.csv"):
        if file_path.is_file() and not file_path.name.startswith("."):
            name_parts = file_path.stem, file_path.suffix
            new_name = f"{name_parts[0]}_{branch_suffix}{name_parts[1]}"
            target_path = testing_dir / new_name

            shutil.copy2(file_path, target_path)
            saved_files.append(str(target_path))

    return saved_files


def save_outputs_to_testing_folder(branch_name):
    """Save branch outputs to testing folder with branch suffix"""
    sanitized_branch = sanitize_branch_name(branch_name)
    all_files = []

    education_files = save_module_to_testing_folder("Education", sanitized_branch)
    if education_files:
        all_files.extend(education_files)

    age_gender_files = save_module_to_testing_folder("Age_Gender", sanitized_branch)
    if age_gender_files:
        all_files.extend(age_gender_files)

    return all_files


def validate_outputs_from_testing_folders(base_branch, compare_branch, base_files, compare_files, pred_year, show_diffs=False):
    """Validate outputs from testing folders, matching files by removing branch suffixes"""
    logger.info(f"Validating outputs: {base_branch} vs {compare_branch}")
    logger.info(f"Comparing prediction years: {pred_year-1} and {pred_year}")

    sanitized_base = sanitize_branch_name(base_branch)
    sanitized_compare = sanitize_branch_name(compare_branch)

    base_file_map = {}

    if len(base_files) != len(compare_files):
        logger.error("WARNING: Different number of files found!")
        logger.error(f"  Base branch has {len(base_files)} files")
        logger.error(f"  Compare branch has {len(compare_files)} files")

    for base_file_path in base_files:
        base_file = Path(base_file_path)
        base_name = base_file.name

        original_name = base_name.replace(f"_{sanitized_base}.csv", "").replace(f"_{sanitized_base}", "")
        logger.info(f"Looking for match for base file: {base_name} -> original: {original_name}")

        found_match = False
        for compare_file_path in compare_files:
            compare_file = Path(compare_file_path)
            compare_name = compare_file.name
            expected_compare_name = f"{original_name}_{sanitized_compare}.csv"

            if compare_name == expected_compare_name:
                base_file_map[base_file] = compare_file
                logger.info(f"  Found match: {compare_name}")
                found_match = True
                break

        if not found_match:
            logger.warning(f"  No match found for {base_name}")
            logger.warning(f"  Expected: {original_name}_{sanitized_compare}.csv")
            logger.warning(f"  Available: {[Path(f).name for f in compare_files]}")

    logger.info(f"Created {len(base_file_map)} file pairs for comparison")

    if not base_file_map:
        logger.error("No matching files found between branches!")
        return False

    all_same = True
    for base_file, compare_file in base_file_map.items():
        base_name = base_file.name
        original_name = base_name.replace(f"_{sanitized_base}.csv", "").replace(f"_{sanitized_base}", "")
        file_name = f"{original_name}.csv"

        if "testing" not in str(base_file) or "testing" not in str(compare_file):
            logger.error("ERROR: Comparing files outside testing directories!")
            logger.error(f"  Base file: {base_file}")
            logger.error(f"  Compare file: {compare_file}")
            continue

        logger.info(f"Comparing: {Path(base_file).name} vs {Path(compare_file).name}")

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


def validate_outputs_local(base_branch, compare_branch, pred_year, last_year, show_diffs=False):
    """Main validation function for local use - handles git checkout and runs predictions"""
    logger.info(f"Validating outputs: {base_branch} vs {compare_branch}")
    logger.info(f"Comparing prediction years: {pred_year-1} and {pred_year}")

    original_branch = get_current_branch()
    if not original_branch:
        logger.error("Could not determine current branch")
        return False

    if original_branch != compare_branch:
        logger.warning(f"Current branch ({original_branch}) does not match ({compare_branch})")
        logger.warning(f"Switch to ({compare_branch}) and then run the validation.")  # so that it runs even with uncommitted changes
        return False

    try:
        # step1 run develop branch with uncommitted changed
        logger.info(f"=== Testing {compare_branch} (with uncommitted changes) ===")
        if not run_predictions(pred_year, last_year):
            logger.error(f"Failed to run predictions on {compare_branch}")
            return False

        compare_files = save_outputs_to_testing_folder(compare_branch)
        logger.info(f"Found {len(compare_files)} compare files: {[Path(f).name for f in compare_files]}")

        # step2 stash changes before checkout main
        logger.info("Stashing uncommitted changed...")
        if not stash_changes():
            logger.warning("Could not shash, continuing anyway...")

        # step3 checkout main and run
        logger.info(f"=== Testing {base_branch} ===")
        if not checkout_branch(base_branch):
            logger.error(f"Failed to checkout {base_branch}")
            return False

        if not run_predictions(pred_year, last_year):
            logger.error("Failed to run predictions on {base_branch}")
            return False

        base_files = save_outputs_to_testing_folder(base_branch)
        logger.info(f"Found {len(base_files)} base files: {[Path(f).name for f in base_files]}")

        # step4 compare outputs
        logger.info("=== Comparing Files ===")
        success = validate_outputs_from_testing_folders(base_branch, compare_branch, base_files, compare_files, pred_year, show_diffs)

        return success

    finally:
        # step5 restore original state
        if not checkout_branch(original_branch):
            logger.error(f"Failed to restore original branch {original_branch}")

        if not restore_changes():
            logger.warning("Could not restore stashed changes")


def main():
    parser = argparse.ArgumentParser(description="Validate output files between Git branches locally")
    parser.add_argument("--base-branch", required=True, help="Base branch name")
    parser.add_argument("--compare-branch", required=True, help="Compare branch name")
    parser.add_argument("--pred-year", type=int, required=True, help="Prediction year to compare")
    parser.add_argument("--last-year", type=int, required=True, help="Last year of available data")
    parser.add_argument("--show-diffs", action="store_true", help="Show detailed differences")

    args = parser.parse_args()
    success = validate_outputs_local(args.base_branch, args.compare_branch, args.pred_year, args.last_year, args.show_diffs)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
