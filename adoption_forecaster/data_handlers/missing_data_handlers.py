"""
Missing Data Handler Module

This module handles data loading and processing operations for education-related data.
It provides functionality for loading and processing data for different age groups
and education levels.
"""

import pandas as pd
import numpy as np
import re
from adoption_forecaster.forecasting.utils import preprocess_df

from adoption_forecaster.config.logging_config import get_logger

logger = get_logger(__name__)


class MissingDataHandler:
    """
    A class for handling missing data operations in education-related data processing.
    """

    def __init__(self, config=None):
        """Initialize MissingDataHandler"""
        self.config = config
        self.education_config = config["modules"]["education"]

    @staticmethod
    def standardize_columns_with_regex(df, regex_patterns, upper_age="64"):
        """
        Standardize column names based on regex patterns.
        Retain only integer columns for education-related columns since in some files provided by GR we have Education
        columns with strings(low,medium,high) and integers (1,2,3) for education level.

        Args:
        df (pd.DataFrame): Dataframe to standardize.
        regex_patterns (dict): Dictionary with standardized column names as keys and regex patterns as values.
        upper_age: 64 or 74. This is because education is given to us in oen sheet containing age grousp 15-64 and 15-74
        Returns:
        pd.DataFrame: Dataframe with standardized column names.
        """
        if "age_range" in df.columns:
            age_range_value = f"16-{upper_age}"
            df = df[df["age_range"] == age_range_value].drop(columns=["age_range"])
        elif "Age group" in df.columns:
            df = df[df["Age group"].str.contains(upper_age)]
        if "Individual type" in df.columns:
            df = df[df["Individual type"].str.contains("75")]
        new_column_names = {}
        for col in df.columns:
            for standard_col, pattern in regex_patterns.items():
                if re.search(pattern, col, re.IGNORECASE):
                    if standard_col == "Education level":
                        if pd.api.types.is_integer_dtype(df[col]):
                            new_column_names[col] = standard_col
                    else:
                        new_column_names[col] = standard_col
                    break
        df = df.copy()
        df.rename(columns=new_column_names, inplace=True)
        return df

    @staticmethod
    def standardize_countries(df, country_column, country_mapping):
        """
        Standardize country names based on a provided mapping.

        Args:
        df (pd.DataFrame): Dataframe with country names to standardize.
        country_column (str): Name of the column containing country names.
        country_mapping (dict): Dictionary with standardized country names as keys and lists of variants as values.

        Returns:
        pd.DataFrame: Dataframe with standardized country names.

        Raises:
        KeyError: If the specified country column is not found in the dataframe.
        """
        try:
            df[country_column] = df[country_column].replace({v: k for k, vs in country_mapping.items() for v in vs})
        except KeyError:
            raise KeyError(f"country column '{country_column}' not found in DataFrame.")
        return df

    @staticmethod
    def get_online_pop_education_distribution(df):
        """
        Calculate the distribution of online population across education levels.

        Args:
            df (pd.DataFrame): DataFrame with online population data

        Returns:
            pd.DataFrame: Distribution ratios of online population by education level

        Notes:
            - Handles special cases where total equals individual education level
            - Returns NaN for invalid distributions
            - Calculates ratios for each country and year combination
        """
        midx = pd.IndexSlice
        online_pop_by_edu_ratios = pd.DataFrame(index=df.index, columns=df.columns)
        for cntry, group in df.groupby(level=0):
            for column in df.columns:
                total = df[column].loc[cntry].sum()  # total of internet users per country per year for 65-74
                if total in [
                    group.loc[:, column].at[midx[cntry, 1]],
                    group.loc[:, column].at[midx[cntry, 2]],
                    group.loc[:, column].at[midx[cntry, 3]],
                ]:
                    (
                        online_pop_by_edu_ratios.loc[(cntry, 1), column],
                        online_pop_by_edu_ratios.loc[(cntry, 2), column],
                        online_pop_by_edu_ratios.loc[(cntry, 3), column],
                    ) = (np.nan, np.nan, np.nan)
                else:
                    online_pop_by_edu_ratios.loc[(cntry, 1), column] = group.loc[:, column].at[midx[cntry, 1]] / total
                    online_pop_by_edu_ratios.loc[(cntry, 2), column] = group.loc[:, column].at[midx[cntry, 2]] / total
                    online_pop_by_edu_ratios.loc[(cntry, 3), column] = group.loc[:, column].at[midx[cntry, 3]] / total
        return online_pop_by_edu_ratios

    @staticmethod
    def merge_education_levels(df):
        """
        Merge education levels 1 and 2 into 'secondary' and level 3 into 'tertiary'.

        Args:
            df (pd.DataFrame): DataFrame with education levels 1, 2, and 3

        Returns:
            pd.DataFrame: DataFrame with merged education levels

        Notes:
            - Level 1 and 2 are combined into 'secondary'
            - Level 3 becomes 'tertiary'
            - Sums values for combined levels
        """
        df_reset_index = df.reset_index()
        df_reset_index["education"] = df_reset_index["education"].replace({1: "secondary", 2: "secondary", 3: "tertiary"})
        df_grouped = df_reset_index.groupby(["country", "education"], as_index=False).sum()
        df_final = df_grouped.set_index(["country", "education"])
        return df_final

    def load_and_process_education_data(
        self,
        loader,
        regex_patterns,
        country_mapping,
        age_group,
    ):
        """
        Main function to load and process all education-related data for both age groups.

        Args:
            loader: Data loader instance
            regex_patterns (dict): Patterns for standardizing column names
            country_mapping (dict): Mapping for standardizing country names
            age_group(str): Age group identifier (e.g., "16_64", "16_74")
        Returns:
            pd.DataFrame: Processed DataFrame with education level ratios

        Notes:
            - Loads and processes data for education levels 1, 2, and 3
            - Handles population ratios and total population data
            - Calculates online population distributions
            - Returns final ratios in secondary/tertiary format
        """

        # get base filenames from config (without .csv extension)
        total_population_base = self.education_config["files"]["total_population_by_age_range"]
        population_ratios_base = self.education_config["files"]["population_ratios"]
        internet_penetration_population_base = self.education_config["files"]["internet_penetration_population"]
        upper_age = age_group.split("_")[1]

        # Load raw data
        total_pops = (
            loader.read_csv_timestamped(total_population_base)
            .pipe(
                MissingDataHandler.standardize_columns_with_regex,
                regex_patterns=regex_patterns,
                upper_age=upper_age,
            )
            .pipe(
                MissingDataHandler.standardize_countries,
                country_column="country",
                country_mapping=country_mapping,
            )
            .pipe(
                preprocess_df,
                df_name="total_population",
                index_col="country",
            )
        )

        pop_by_edu_ratio = (
            loader.read_csv_timestamped(population_ratios_base)
            .pipe(
                MissingDataHandler.standardize_columns_with_regex,
                regex_patterns=regex_patterns,
                upper_age=upper_age,
            )
            .pipe(
                MissingDataHandler.standardize_countries,
                country_column="country",
                country_mapping=country_mapping,
            )
            .pipe(
                preprocess_df,
                df_name="population_ratios_by_education",
                div=100,
                index_col=["country", "education"],
            )
        )

        int_pop_by_edu_ratio = (
            loader.read_csv_timestamped(internet_penetration_population_base)
            .pipe(
                MissingDataHandler.standardize_columns_with_regex,
                regex_patterns=regex_patterns,
                upper_age=upper_age,
            )
            .pipe(
                MissingDataHandler.standardize_countries,
                country_column="country",
                country_mapping=country_mapping,
            )
            .pipe(
                preprocess_df,
                df_name="internet_penetration_population_by_education",
                div=100,
                index_col=["country", "education"],
            )
        )

        # create dataframe with internet penetration universes per country, per educ level
        int_pop_by_edu_abs = MissingDataHandler._int_pop_by_edu_abs(int_pop_by_edu_ratio, pop_by_edu_ratio, total_pops)

        idx_cast = int
        df_processed = preprocess_df(int_pop_by_edu_abs, "df_individuals")  # preprocessing

        df_processed.columns = [idx_cast(i) for i in df_processed.columns.tolist()]

        logger.info(f"{age_group} data columns after processing: {sorted(df_processed.columns.tolist())}")

        full_data_abs = df_processed.copy()

        internet_users_by_edu_ratio_full = MissingDataHandler.calculate_internet_users_ratio(full_data_abs)

        return internet_users_by_edu_ratio_full

    @staticmethod
    def calculate_internet_users_ratio(full_data_abs):
        """
        Calculate internet users ratio by education level relative to total population.

        The input file contains ratios = internet users with ed. level X / total pop. with ed. level X
        This function converts to ratios = internet users with ed. level X / total population

        Args:
            full_data_abs (pd.DataFrame): DataFrame with absolute numbers of internet users

        Returns:
            pd.DataFrame: DataFrame with calculated ratios
        """
        midx = pd.IndexSlice
        result_df = pd.DataFrame(index=full_data_abs.index, columns=full_data_abs.columns)

        for cntry, group in full_data_abs.groupby(level=0):
            for column in full_data_abs.columns:
                total = group[column].sum()  # total of internet users per country per year

                level_1_val = group.loc[:, column].at[midx[cntry, 1]]
                level_2_val = group.loc[:, column].at[midx[cntry, 2]]
                level_3_val = group.loc[:, column].at[midx[cntry, 3]]

                # Avoid division by zero: if total is zero or NaN, set ratios to NaN
                if total == 0 or pd.isna(total):
                    level_1_ratio = pd.NA
                    level_2_ratio = pd.NA
                    level_3_ratio = pd.NA
                else:
                    level_1_ratio = level_1_val / total
                    level_2_ratio = level_2_val / total
                    level_3_ratio = level_3_val / total

                result_df.loc[(cntry, 1), column] = level_1_ratio
                result_df.loc[(cntry, 2), column] = level_2_ratio
                result_df.loc[(cntry, 3), column] = level_3_ratio

        return result_df

    @staticmethod
    def _pop_by_edu_abs(pop_by_edu_ratio, total_pops):
        """
        Convert education level population percentages to absolute numbers.

        Args:
            pop_by_edu_ratio (pd.DataFrame): Population ratios by education level
            total_pops (pd.DataFrame): Total population numbers

        Returns:
            pd.DataFrame: Absolute population numbers by education level
        """

        pop_by_edu_abs = pd.DataFrame()
        for edu, group in pop_by_edu_ratio.groupby(level=1):
            grp = group.copy().droplevel(1, axis=0)
            abs_pop = grp.loc[total_pops.index, :] * total_pops
            abs_pop.loc[:, "education"] = edu
            abs_pop.set_index("education", append=True, inplace=True)
            abs_pop.reorder_levels(["country", "education"], axis=0)
            pop_by_edu_abs = pd.concat([abs_pop, pop_by_edu_abs], sort=True)

        return pop_by_edu_abs

    @staticmethod
    def _int_pop_by_edu_abs(int_pop_by_edu_ratio, pop_by_edu_ratio, total_pops):
        """
        Convert internet penetration percentages to absolute population numbers.

        Args:
            int_pop_by_edu_ratio (pd.DataFrame): Internet penetration ratios by education level
            pop_by_edu_ratio (pd.DataFrame): Population ratios by education level
            total_pops (pd.DataFrame): Total population numbers

        Returns:
            pd.DataFrame: Absolute numbers of internet users by education level
        """

        pop_by_edu_abs = MissingDataHandler._pop_by_edu_abs(pop_by_edu_ratio, total_pops)

        int_pop_by_edu_abs = pd.DataFrame()
        for _, group in pop_by_edu_abs.groupby(level=1):
            abs_pop = group * int_pop_by_edu_ratio.loc[group.index, :]
            int_pop_by_edu_abs = pd.concat([abs_pop, int_pop_by_edu_abs], sort=True)

        return int_pop_by_edu_abs
