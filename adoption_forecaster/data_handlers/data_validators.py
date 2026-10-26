"""
Data Validator Module

This module provides data validation functionality for the adoption forecaster system.
It includes methods for validating data quality, consistency, and completeness.
"""

import pandas as pd

from adoption_forecaster.config.logging_config import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    A class for validating data quality and structure.
    """

    def __init__(self, config=None):
        """Initialize DataValidator"""
        self.config = config

    @staticmethod
    def validate_dataframe(df, name="DataFrame"):
        """
        Basic validation for a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to validate
            name (str): Name of the DataFrame for logging

        Returns:
            bool: True if validation passes, False otherwise
        """
        if df is None:
            logger.error(f"{name} is None")
            return False

        if df.empty:
            logger.warning(f"{name} is empty")
            return False

        if df.isnull().all().all():
            logger.error(f"{name} contains only null values")
            return False

        logger.info(f"{name} validation passed. Shape: {df.shape}")
        return True

    @staticmethod
    def validate_required_columns(df, required_columns, name="DataFrame"):
        """
        Validate that required columns are present in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to validate
            required_columns (list): List of required column names
            name (str): Name of the DataFrame for logging

        Returns:
            bool: True if all required columns are present, False otherwise
        """
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"{name} missing required columns: {missing_columns}")
            return False

        logger.info(f"{name} has all required columns: {required_columns}")
        return True

    @staticmethod
    def validate_numeric_columns(df, numeric_columns, name="DataFrame"):
        """
        Validate that specified columns contain numeric data.

        Args:
            df (pd.DataFrame): DataFrame to validate
            numeric_columns (list): List of column names that should be numeric
            name (str): Name of the DataFrame for logging

        Returns:
            bool: True if all specified columns are numeric, False otherwise
        """
        non_numeric_columns = []

        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_columns.append(col)

        if non_numeric_columns:
            logger.error(f"{name} has non-numeric columns: {non_numeric_columns}")
            return False

        logger.info(f"{name} has all numeric columns: {numeric_columns}")
        return True

    @staticmethod
    def validate_year_range(df, min_year=2000, max_year=2030, name="DataFrame"):
        """
        Validate that year columns are within expected range.

        Args:
            df (pd.DataFrame): DataFrame to validate
            min_year (int): Minimum expected year
            max_year (int): Maximum expected year
            name (str): Name of the DataFrame for logging

        Returns:
            bool: True if year range is valid, False otherwise
        """
        year_columns = [col for col in df.columns if isinstance(col, int) and min_year <= col <= max_year]

        if not year_columns:
            logger.warning(f"{name} has no year columns in range {min_year}-{max_year}")
            return False

        logger.info(f"{name} has {len(year_columns)} year columns in valid range")
        return True
