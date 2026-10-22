"""
This module serves as the main entry point for internet usage predictions.
It orchestrates the prediction pipeline for both Education and Age_Gender modules, including:
- Loading and preprocessing data for different demographic segments
- Handling predictions for specific target markets using proxy countries
- Managing quota-based forecasting for different demographic groups
- Saving final predictions in standardized format

Key Features:
- Module-agnostic processing (Education and Age_Gender)
- Target market specific proxy country configurations
- Quota-based forecasting with error learning
- Automatic data validation and clipping
- Standardized output formatting
"""

import argparse
from quotas_forecasting.forecasting.processors import get_processor
from quotas_forecasting.forecasting.predictors import get_predictor
from quotas_forecasting.data_handlers.loader import Loader
from quotas_forecasting.data_handlers.file_writer import FileWriter
from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.fetch_data.fetch_data_from_Eurostat import EurostatDataFetcher
from quotas_forecasting.correction.llm import MissingApiKeyError
from quotas_forecasting.correction.pipeline import NoSourceStoreError, prepare_correction
import sys

# Set up logging with color support
from quotas_forecasting.config.logging_config import get_logger

logger = get_logger(__name__)


def run_single_module(module_name, mode, bucket_name, project_id, pred_year, last_year, config=None, correction_store=None, correction_llm=None):
    """
    Helper function to run a single module with proper setup.

    Args:
        module_name: Name of the module ("education" or "age_gender")
        mode: Execution mode ("local" or "prod")
        bucket_name: GCS bucket name
        project_id: GCP project ID
        pred_year: Prediction year from args
        last_year: Last year from args
        config: Configuration object (loaded once in main)
        correction_store: Source store for the correction agent (built from config when correction is enabled and this is None)
        correction_llm: LLM client for the correction agent (built from config when correction is enabled and this is None)
    """
    # with correction enabled, a missing NVIDIA_API_KEY or source store fails here, before any forecasting work starts
    correction_store, correction_llm = prepare_correction(config, correction_store, correction_llm)

    # Create module-specific instances
    namespace = module_name.title()  # "education" -> "Education", "age_gender" -> "Age_Gender"

    loader = Loader(
        namespace=namespace,
        mode=mode,
        config=config,
    )
    writer = FileWriter(
        namespace=namespace,
        mode=mode,
        pred_year=pred_year,
        config=config,
    )
    processor = get_processor(module_name, config=config)
    predictor = get_predictor(
        module_name,
        mode=mode,
        pred_year=pred_year,
        last_year=last_year,
        loader=loader,
        config=config,
    )

    # Run the module
    if module_name == "education":
        processor.run_education_pipeline(
            loader, writer, predictor, pred_year, last_year, correction_store=correction_store, correction_llm=correction_llm
        )
    elif module_name == "age_gender":
        processor.run_age_gender_pipeline(
            loader, writer, predictor, pred_year, last_year, correction_store=correction_store, correction_llm=correction_llm
        )
    else:
        raise ValueError(f"Unknown module: {module_name}")


if __name__ == "__main__":
    try:
        config = load_config()
        defaults = config["defaults"]
        validation = config["validation"]
    except Exception as e:
        logger.info(f"Error loading configuration: {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run the main script with specified environment.")
    parser.add_argument(
        "--module",
        choices=["education", "age_gender", "both"],
        default="both",
        help="Module to run (default: both)",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "prod"],
        required=True,
        help="Execution mode: local or production",
    )
    parser.add_argument(
        "--pred-year",
        type=int,
        required=True,
        help="Target year for generating predictions",
    )
    parser.add_argument(
        "--last-year",
        type=int,
        required=True,
        help="Last year of available data",
    )

    args = parser.parse_args()

    if args.pred_year <= args.last_year:
        raise ValueError("--pred-year must be greater than --last-year")

    # Validate year difference using config
    year_diff = args.pred_year - args.last_year
    required_diff = validation["year_difference"]
    if year_diff != required_diff:
        raise ValueError(
            f"Difference between --pred-year ({args.pred_year}) and --last-year ({args.last_year}) must be exactly {required_diff} years, got {year_diff}"
        )

    # correction agent dependencies are built once, before any fetching or forecasting
    try:
        correction_store, correction_llm = prepare_correction(config, None, None)
    except (MissingApiKeyError, NoSourceStoreError, ValueError) as e:
        logger.error(f"Correction agent cannot start: {e}")
        sys.exit(1)

    # fetch Eurostat data
    if config["defaults"]["auto_fetch"]:
        try:
            logger.info("Starting Eurostat data fetching...")
            fetcher = EurostatDataFetcher(last_year_of_observations=args.last_year, config=config)
            success = fetcher.fetch_data()
            if success:
                logger.info("Eurostat data fetching completed successfully")
            else:
                logger.error("Eurostat data fetching failed")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error fetching Eurostat data: {e}")
            sys.exit(1)
    else:
        logger.info("Auto-fetch disabled, skipping Eurostat data fetching...")

    bucket_name = config["defaults"]["bucket_name"]
    project_id = config["defaults"]["project_id"]

    pred_year = args.pred_year
    last_year = args.last_year
    mode = args.mode
    module = args.module

    # module execution
    modules = ["education", "age_gender"] if module == "both" else [module]
    for module_name in modules:
        logger.info(f"Running {module_name} module...")
        run_single_module(
            module_name,
            mode,
            bucket_name,
            project_id,
            pred_year,
            last_year,
            config,
            correction_store=correction_store,
            correction_llm=correction_llm,
        )

    if module == "both":
        logger.info("Both modules completed successfully!")
