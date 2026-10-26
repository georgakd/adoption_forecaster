"""
Core Processing Module

This module provides the main processing classes for education and age_gender modules.
It focuses on the core pipeline execution and delegates specialized functionality
to dedicated handler modules.

Key features:
- Core pipeline execution for education and age_gender modules
- Integration with specialized handlers for proxy countries and missing data
- Main prediction pipeline orchestration
- File saving and output management
"""

import pandas as pd
import re
import numpy as np
from adoption_forecaster.data_handlers.proxy_handlers import ProxyHandler
from adoption_forecaster.data_handlers.missing_data_handlers import MissingDataHandler
from adoption_forecaster.data_handlers.data_validators import DataValidator
from adoption_forecaster.config.logging_config import get_logger
from adoption_forecaster.correction.pipeline import maybe_correct


pd.set_option("future.no_silent_downcasting", True)


logger = get_logger(__name__)


class EducationProcessor:
    """
    A class for implementing education-related internet usage data processing.

    This class handles the education module's core pipeline execution and delegates
    specialized functionality to dedicated handler modules.
    """

    def __init__(self, module_type="education", config=None):
        """Initialize EducationProcessor"""
        self.module_type = module_type
        if config is None:
            raise ValueError("EducationProcessor requires config parameter - it cannot be None")
        self.config = config
        self.education_config = config["modules"]["education"]

        self.proxy_handler = ProxyHandler(config=self.config)
        self.missing_data_handler = MissingDataHandler(config=self.config)
        self.validator = DataValidator(config=self.config)
        self.age_groups = self.education_config["settings"]["age_groups"]

    def load_and_process_education_data(self, loader, regex_patterns, country_mapping, age_group):
        """
        Load and process data for 16-64  and 16-74 age groups.

        Args:
            loader: Data loader instance
            regex_patterns (dict): Regex patterns for column standardization
            country_mapping (dict): Country name mapping
            last_year_of_observations (int): Last year of available data

        Returns:
            internet_users_by_edu_ratio_full containing processed dataframes
        """
        return self.missing_data_handler.load_and_process_education_data(
            loader,
            regex_patterns,
            country_mapping,
            age_group,
        )

    def run_education_pipeline(
        self, loader, writer, predictor, year_of_predictions, last_year_of_observations, correction_store=None, correction_llm=None
    ):
        """Run education pipeline for all age groups (correction_store/correction_llm are used only when correction.enabled is true)"""
        education_data = {}

        regex_patterns = self.education_config["processing"]["regex_patterns"]
        country_mapping = self.education_config["processing"]["country_mapping"]

        for age_group in self.age_groups:
            try:
                # Load and process data
                logger.info(f"Loading {age_group} data for predictions up to {year_of_predictions}")
                education_df = self.load_and_process_education_data(
                    loader,
                    regex_patterns,
                    country_mapping,
                    age_group=age_group,
                )

                start_year = self.config["prediction"]["time_series"]["start_year"]
                education_df = education_df[range(start_year, last_year_of_observations + 1)]
                logger.info(f"Successfully loaded {age_group} data")

            except Exception as e:
                logger.info(f"Could not load {age_group} data: {e}")

            # since we dont have target markets, it will go straight to else
            target_markets = self.config.get("target_markets", {})
            edu_levels = self.education_config["settings"]["education_levels"]
            if target_markets:
                target_countries = list(target_markets.keys())
                logger.info(f"Applying LASSO interpolation for target countries: {target_countries}")
                education_df_filled = self.proxy_handler.apply_LASSO_for_the_three_countries_with_missing_data(
                    education_df, target_countries, edu_levels, last_year_of_observations
                )
            else:
                logger.info("No target markets configured - skipping LASSO interpolation for specific countries")
                education_df_filled = education_df

            try:
                education_df_merged = self.missing_data_handler.merge_education_levels(education_df_filled).astype(float).replace(0, np.nan)
                # Run predictions
                education_df_merged = predictor.predict(education_df_merged, last_year_of_observations, year_of_predictions)
                # correction agent (no-op unless correction.enabled): cited corrections to forecast cells, before saving
                education_df_merged = maybe_correct(
                    education_df_merged, "education", last_year_of_observations, self.config, correction_store, correction_llm
                )

                education_data[age_group] = education_df_merged
                logger.info(f"Successfully completed {age_group} education pipeline. Final shape: {education_data[age_group].shape}")

            except Exception as e:
                logger.error(f"Failed to run {age_group} education prediction: {e}")
                education_data[age_group] = None

        # Save all predictions
        self._save_education_predictions(education_data, writer, year_of_predictions)

        return education_data

    def _save_education_predictions(
        self,
        education_data,
        writer,
        year_of_predictions,
    ):
        """
        Save education predictions to files for all age groups.
        """
        output_dir = self.education_config["output"]["directory"]
        start_year = year_of_predictions - self.config["prediction"]["preprocessing"]["historical_years_back"]
        year_diff = self.config["validation"]["year_difference"]
        last_year_of_observations = year_of_predictions - year_diff

        # Pre-calculate config keys to avoid repeated access
        output_config = self.education_config["output"]

        # Process each age group
        for age_group in self.age_groups:
            df_complete = education_data.get(age_group)

            if df_complete is None:
                logger.info(f"Skipping {age_group} predictions - data not available")
                continue

            logger.info(f"Processing {age_group} predictions...")

            available_years = [col for col in df_complete.columns if isinstance(col, int) and start_year <= col <= year_of_predictions]

            try:
                df_unstacked = df_complete[available_years].unstack(level=1)
                df_unstacked.columns = [f"{year}_{edu}" for year, edu in df_unstacked.columns]

                pred_columns = [col for col in df_unstacked.columns if str(year_of_predictions) in col]
                df_preds_only = df_unstacked[pred_columns]

                # Save predictions only
                writer.save_predictions(
                    df_preds_only,
                    year_of_predictions,
                    output_dir,
                    output_config[f"predictions_only_{age_group}"],
                    last_year_of_observations=last_year_of_observations,
                )
                # Save past data + predictions
                writer.save_predictions(
                    df_unstacked,
                    year_of_predictions,
                    output_dir,
                    output_config[f"past_data_and_predictions_{age_group}"],
                    last_year_of_observations=last_year_of_observations,
                )

            except Exception as e:
                logger.exception(f"Error saving {age_group} predictions: {e}")


class AgeGenderProcessor:
    """
    A class for implementing gender and age group internet usage data processing.

    This class handles the age_gender module's data preprocessing, standardization,
    and pipeline execution for gender and age-specific internet usage predictions.
    """

    def __init__(self, module_type="age_gender", config=None):
        """Initialize AgeGenderProcessor"""
        self.module_type = module_type
        if config is None:
            raise ValueError("AgeGenderProcessor requires config parameter - it cannot be None")

        self.module_type = module_type
        self.config = config
        self.config_agegender = config["modules"]["age_gender"]

    def extract_age_groups(self, text):
        """
        Extract age groups from text and format them.

        Args:
            text (str): Text containing age group information (e.g., "16 to 24 years old")

        Returns:
            list: Formatted age groups (e.g., ["16_24"])

        Notes:
            - Extracts age ranges from text like "16 to 24 years old"
            - Returns formatted strings like "16_24"
        """
        age_groups = re.findall(r"(\d+)\s+to\s+(\d+)\s+years\s+old", text)
        formatted_age_groups = ["{}_{}".format(lower, upper) for lower, upper in age_groups]
        return formatted_age_groups

    def extract_gender(self, row):
        """
        Extract gender information from a row's 'Individual type' field.

        Args:
            row (pd.Series): DataFrame row containing 'Individual type' field

        Returns:
            pd.Series: Updated row with extracted gender information

        Notes:
            - Looks for Male/Female patterns in text
            - Case insensitive matching
            - Handles plural forms (Males/Females)
        """
        individual_type = row["Individual type"].upper().replace(",", "")
        if re.search(r"\bMALE(?:S)?\b", individual_type):
            row["gender"] = "male"
        elif re.search(r"\bFEMALE(?:S)?\b", individual_type):
            row["gender"] = "female"
        return row

    def split_composite_index(self, row):
        """
        Split composite index into country and Individual type components.

        Args:
            row (pd.Series): DataFrame row containing 'composite_index' field

        Returns:
            pd.Series: Series containing split components:
                - country: Extracted country name
                - Individual type: Formatted demographic information

        Notes:
            - Splits on spaces, taking all but last two parts as country
            - Formats individual type with gender first, then age range
            - Converts age group format from '16_24' to '16 to 24'
        """
        parts = row["composite_index"].split()
        country = " ".join(parts[:-2])
        indiv_list = parts[-2:]  # last two elements i.e. age group (e.g. 16_24) and gender male/female
        indiv_list.reverse()  # reverse list to bring gender first
        individual_type = ", ".join(indiv_list)
        individual_type = individual_type.replace("_", " to ").capitalize()
        return pd.Series([country, individual_type])

    def run_age_gender_pipeline(
        self, loader, writer, predictor, year_of_predictions, last_year_of_observations, correction_store=None, correction_llm=None
    ):
        """
        Run the complete gender and age prediction pipeline.

        Args:
            loader: Data loader instance
            writer: File writer instance
            predictor: Prediction model instance
            year_of_predictions: Target year for predictions
            last_year_of_observations: Last year of available data
            correction_store: Source store for the correction agent (used only when correction.enabled is true)
            correction_llm: LLM client for the correction agent (used only when correction.enabled is true)
        """
        country_mapping = self.config["modules"]["education"]["processing"]["country_mapping"]
        df_age_gender_path = self.config_agegender["files"]["age_gender"]

        # get base filenames (without .csv extension)
        df_age_gender_base = df_age_gender_path.replace(".csv", "")

        logger.info(f"Loading age_gender data: {df_age_gender_base}")
        df_age_gender = loader.read_csv_timestamped(df_age_gender_base)

        df_age_gender = MissingDataHandler.standardize_countries(df_age_gender, country_column="country", country_mapping=country_mapping)
        start_year = self.config["modules"]["age_gender"]["settings"]["start_year"]
        df_age_gender = df_age_gender[["country", "Individual type"] + [str(i) for i in range(start_year, last_year_of_observations + 1)]]
        logger.info(f"Age_gender data loaded successfully. Shape: {df_age_gender.shape}")
        # Run the predictor with pre-loaded data
        frame = predictor.run_with_data(df_age_gender, last_year_of_observations)
        # correction agent (no-op unless correction.enabled): cited corrections to forecast cells, before the rename/fillna/save
        frame = maybe_correct(frame, "age_gender", last_year_of_observations, self.config, correction_store, correction_llm)
        previous_year_offset = self.config["prediction"]["preprocessing"]["previous_year_offset"]
        # convert all columns to integer type
        frame.rename(
            columns={
                f"{year_of_predictions-previous_year_offset}": int(year_of_predictions - previous_year_offset),
                f"{year_of_predictions}": int(year_of_predictions),
            },
            inplace=True,
        )

        countries_to_drop = []
        output_dir = self.config_agegender["output"]["directory"]
        past_data_start_year = self.config["prediction"]["time_series"]["past_data_start_year"]  # noqa: F841

        frame = frame[~(frame["country"].isin(countries_to_drop))].fillna(self.config["prediction"]["preprocessing"]["fill_missing_with"])
        frame = frame.set_index("country")  # change index to country to avoid having the pd index in output file

        # Save predictions only
        writer.save_predictions(
            frame[["Individual type", year_of_predictions]],
            year_of_predictions,
            output_dir,
            self.config_agegender["output"]["predictions_only"],
            countries_to_drop,
            last_year_of_observations=last_year_of_observations,
        )

        # Save predictions with past years
        writer.save_predictions(
            frame,
            year_of_predictions,
            output_dir,
            self.config_agegender["output"]["past_data_and_predictions"],
            countries_to_drop,
            last_year_of_observations=last_year_of_observations,
        )


def get_processor(module, config=None):
    """
    Factory function to get the appropriate processor for the specified module.

    Args:
        module (str): Module type ("education" or "age_gender")

    Returns:
        Processor instance: EducationProcessor or AgeGenderProcessor

    Raises:
        ValueError: If module is not recognized
    """
    if module == "education":
        return EducationProcessor(module_type=module, config=config)
    elif module == "age_gender":
        return AgeGenderProcessor(module_type=module, config=config)
    else:
        raise ValueError(f"Unknown module: {module}. Must be 'education' or 'age_gender'")
