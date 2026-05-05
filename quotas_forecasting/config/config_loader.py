"""
configuration loader for Quotas Forecasting.
This module provides a way to load configuration settings from the YAML config file.
"""

import yaml
import os

from quotas_forecasting.config.logging_config import get_logger

logger = get_logger(__name__)


CONFIG_DIR = os.path.dirname(__file__)
config_path = os.path.join(CONFIG_DIR, "config.yaml")


def load_config(config_path: str = config_path):
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")

            # Override with environment variables for Cloud Run compatibility
            if os.getenv("GOOGLE_CLOUD_PROJECT"):
                config["defaults"]["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT")
                logger.info(f"Overriding project_id with environment variable: {os.getenv('GOOGLE_CLOUD_PROJECT')}")

            if os.getenv("GCS_BUCKET_NAME"):
                config["defaults"]["bucket_name"] = os.getenv("GCS_BUCKET_NAME")
                logger.info(f"Overriding bucket_name with environment variable: {os.getenv('GCS_BUCKET_NAME')}")

            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise
