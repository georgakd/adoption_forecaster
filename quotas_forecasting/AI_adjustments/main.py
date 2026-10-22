"""
Main entry point for AI-driven projection corrections.

Loads projection CSV files, applies AI corrections using hard rules and market
researcher feedback, and saves adjusted projections to output directory.

Supports education and age_gender modules with optional RAG feedback integration.
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Tuple
from quotas_forecasting.AI_adjustments.client import AIClient
from quotas_forecasting.AI_adjustments.utils import get_projection_columns, get_historical_columns, validate_historical_data, report_changes
from quotas_forecasting.AI_adjustments.agent import apply_corrections
from quotas_forecasting.AI_adjustments.utils import load_feedback, get_feedback_csv_path
from quotas_forecasting.AI_adjustments.feedback_rag import FeedbackRAGRetriever
from quotas_forecasting.AI_adjustments.gcp_secrets import get_openai_api_key
from quotas_forecasting.config.config_loader import load_config


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Directory setup
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the main script with specified environment.")
    parser.add_argument(
        "--module",
        choices=["education", "age_gender"],
        required=True,
        help="Module to process: education or age_gender",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "prod"],
        required=True,
        help="Execution mode: local or production",
    )
    parser.add_argument(
        "--adjusted-year",
        type=int,
        required=True,
        help="Target year for adjusting predictions",
    )
    return parser.parse_args()


def select_input_file() -> Optional[Path]:
    """Prompt user to select an input CSV file."""
    input_files = list(INPUT_DIR.glob("*.csv"))

    if not input_files:
        logger.warning(f"\n[WARNING] No CSV files found in '{INPUT_DIR}/' directory.")
        logger.info(f"Please place your prediction CSV files in: {INPUT_DIR.absolute()}")
        return None

    logger.info(f"\nFound {len(input_files)} CSV file(s):")
    for i, file in enumerate(input_files, 1):
        logger.info(f"  {i}. {file.name}")

    choice = input(f"\nSelect file number (1-{len(input_files)}): ")
    try:
        return input_files[int(choice) - 1]
    except (ValueError, IndexError) as e:
        logger.exception(f"Invalid choice: {e}. Exiting.")
        return None


def load_and_validate_data(selected_file: Path, module: str, adjusted_year: int) -> Optional[Tuple[pd.DataFrame, str, Optional[str], list, list]]:
    """Load CSV file and validate data structure."""
    try:
        df = pd.read_csv(selected_file)

        logger.info(f"\nLoaded {selected_file.name}")
        logger.info(f"Shape: {df.shape} (rows x columns)")
        logger.info(f"Columns: {', '.join(df.columns.tolist())}")

        country_col = df.columns[0]
        individual_col = df.columns[1] if module == "age_gender" else None

        projection_columns = get_projection_columns(df, module, adjusted_year)
        historical_columns = get_historical_columns(df, country_col, individual_col, projection_columns)

        if not projection_columns:
            logger.warning(f"\n[WARNING] No projection columns (containing {adjusted_year} or {adjusted_year - 1}) found!")
            return None

        logger.info(f"\n  Projection columns (to be adjusted): {', '.join(projection_columns)}")
        logger.info(f"  Historical columns: {', '.join(historical_columns)}")

        return df, country_col, individual_col, projection_columns, historical_columns

    except Exception as e:
        logger.exception(f"Error reading file: {e}")
        return None


def get_custom_instruction() -> Optional[str]:
    """Prompt user for custom correction instructions."""
    logger.info("\n" + "-" * 60)
    logger.info("Using rules from rules.py for corrections.")
    logger.info("(To provide custom instructions, modify rules.py or pass instruction parameter)")

    usr_input = input("\nEnter custom correction instructions (press Enter to use default rules from rules.py): ").strip()
    return usr_input if usr_input else None


def save_output(df_modified: pd.DataFrame, selected_file: Path) -> Path:
    """Save modified DataFrame to output directory."""
    output_filename = f"modified_{selected_file.name}"
    output_path = OUTPUT_DIR / output_filename
    df_modified.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("\n" + "=" * 60)
    logger.info("Corrections applied and saved!")
    logger.info(f"  Original: {selected_file}")
    logger.info(f"  Modified: {output_path}")
    logger.info("=" * 60)

    return output_path


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Quotas Forecasting AI - Projection Corrector")
    logger.info("=" * 60)

    args = parse_arguments()

    config = load_config()
    project_id = config["defaults"]["project_id"]
    secret_name = config["secrets"]["openai_api_key"]
    open_api_key = get_openai_api_key(project_id, secret_name)
    client = AIClient(api_key=open_api_key)

    # mode = args.mode
    module = args.module
    adjusted_year = args.adjusted_year
    model = "gpt-4o"

    selected_file = select_input_file()
    if selected_file is None:
        return

    data_result = load_and_validate_data(selected_file, module, adjusted_year)
    if data_result is None:
        return

    df, country_col, individual_col, projection_columns, historical_columns = data_result

    feedback_retriever = None
    feedback_csv = get_feedback_csv_path(INPUT_DIR, module, adjusted_year)
    if feedback_csv is not None and feedback_csv.exists():
        try:
            feedback_data = load_feedback(str(feedback_csv), module=module)
            feedback_retriever = FeedbackRAGRetriever(feedback_data, openai_api_key=open_api_key)
            logger.info(f"[INFO] Loaded education feedback for {len(feedback_data)} countries")
        except Exception as e:
            logger.warning(f"[WARNING] Error loading feedback: {e}. Proceeding without feedback.")
    else:
        logger.info("[INFO] No feedback file found, proceeding without feedback")

    instruction = get_custom_instruction()

    try:
        df_modified, adjustment_log = apply_corrections(
            df,
            projection_columns,
            country_col,
            individual_col,
            adjusted_year,
            client=client,
            module=module,
            model=model,
            instruction=instruction,
            feedback_retriever=feedback_retriever,
        )

        validate_historical_data(df, df_modified, historical_columns)
        save_output(df_modified, selected_file)
        if adjustment_log:
            import json

            adjustment_log_path = OUTPUT_DIR / f"adjustment_log_{selected_file.stem}.json"
            with open(adjustment_log_path, "w", encoding="utf-8") as f:
                json.dump(adjustment_log, f, indent=2, ensure_ascii=False)
                logger.info(f"\n[INFO] Adjustment log saved: {adjustment_log_path}")
                logger.info(f"[INFO] Total adjustments: {len(adjustment_log)}")

        report_changes(df, df_modified, projection_columns, country_col, individual_col, module)

    except Exception as e:
        logger.exception(f"\n[ERROR] Error: {e}")


if __name__ == "__main__":
    main()
