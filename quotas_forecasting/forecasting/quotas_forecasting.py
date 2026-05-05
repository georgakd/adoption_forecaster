"""
This module contains the Quotas_Forecasting class for implementing quota-based forecasting
for gender and age group predictions.

This class combines multiple forecasting methods and handles the entire prediction pipeline
including data preprocessing, model training, prediction generation, and results combination.
"""

import numpy as np
import pandas as pd
import warnings
from quotas_forecasting.config.logging_config import get_logger
from quotas_forecasting.forecasting.utils import (
    init_preds,
    preprocess_df,
    extract_age_groups,
    extract_gender,
    fit_predict_theta,
    train_lasso,
    backtest_model,
    prediction_gender_data,
    fit_predict_logistic,
    fit_predict_logistic_older_age_groups,
    adjust_100prc_predictions_cases,
    age_groups_tweak,
)

# Suppress specific warnings from statsmodels Holt-Winters
warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning, module="statsmodels")


logger = get_logger(__name__)


class Quotas_Forecasting:
    """
    A class for implementing quota-based forecasting for gender and age group predictions.

    This class combines multiple forecasting methods and handles the entire prediction pipeline
    including data preprocessing, model training, prediction generation, and results combination.

    Attributes:
        last_year_of_observations (int): Last year of available data
        year_of_predictions (int): Target year for predictions
        df_individuals (pd.DataFrame): Individual-level data
        df_age_gender (pd.DataFrame): Age and gender group data
        target_markets (dict): Mapping of target markets to proxy markets

    Key Methods:
        run: Execute the complete forecasting pipeline
        set_target_market: Configure target markets and their proxies
        combine_forecasts: Combine predictions from multiple methods
        get_forecast_results: Retrieve final prediction results
    """

    def __init__(
        self,
        df_age_gender_path=None,
        df_age_gender=None,
        last_year_of_observations=None,
        year_of_predictions=None,
        loader=None,
        config=None,
    ):
        """
        Initialize the Quota_Forecasting class.

        Args:
            df_age_gender_path (str, optional): Path to gender/age data CSV (legacy)
            df_age_gender (pd.DataFrame, optional): Pre-loaded age_gender data
            last_year_of_observations (int): Last year of available data
            year_of_predictions (int): Target year for predictions
            loader (Loader, optional): Loader instance for reading files
            config (dict, optional): Configuration dictionary
        """
        self.config = config

        self.last_year_of_observations = last_year_of_observations
        self.year_of_predictions = year_of_predictions

        if df_age_gender is not None:
            self.df_age_gender = df_age_gender
        else:
            raise ValueError("Must provide pre-loaded age_gender DataFrame")

        self.target_markets = {}
        self.training_and_prediction_years = {}

    def run(self):
        """
        Execute the complete forecasting pipeline.

        Returns:
            pd.DataFrame: Final forecast results
        """
        self.combine_forecasts()
        return self.get_forecast_results()

    def set_target_market(self, target_markets):
        """
        Sets target markets and their corresponding proxy markets.

        Args:
            target_markets (dict): A dictionary where keys are target markets and values are lists of proxy markets.
        """
        for target_market, proxy_markets in target_markets.items():
            self.target_markets[target_market] = proxy_markets

    def determine_training_and_prediction_years(self, df, group_by_columns):
        """
        Determines the training and prediction years for each target market based on the available data.

        Args:
            df (pd.DataFrame): The DataFrame containing the time series data
            group_by_columns (list): List of column names to group the data by

        Returns:
            dict: Dictionary mapping target markets to their year ranges

        Notes:
            - Considers data from 2010 onwards
            - Handles missing data and determines valid year ranges
            - Sets prediction period after the last available data
        """
        if not self.target_markets:
            logger.info("No target markets configured - skipping year determination")
            return {}

        target_years = {}
        for target_market in self.target_markets.keys():
            # Initialize variables to store the global start and end years for each target market
            start_year_candidates = []
            end_year_candidates = []

            for index, row in df[df["country"] == target_market].iterrows():
                # Limit the DataFrame columns to the last year of observations
                limited_row = row.drop(["country", "Individual type", "composite_index"] + group_by_columns)
                min_training_year = self.config["prediction"]["time_series"]["min_training_year"]

                # adding lower limit from config (most time series have consistent data starting from min_training_year) and upper limit last_year_of_observation,
                # this is to allow this to work with the go back two years function.
                limited_years = limited_row.index[
                    (limited_row.index.astype(int) <= self.last_year_of_observations) & (limited_row.index.astype(int) >= min_training_year)
                ].astype(int)
                limited_values = limited_row[limited_years.astype(str)].values.astype(float)
                non_nan_indices = np.where(~np.isnan(limited_values))[0]
                if non_nan_indices.size == 0:
                    continue  # Skip if all values are NaN

                start_year = limited_years[non_nan_indices[0]]
                end_year = limited_years[non_nan_indices[-1]]

                start_year_candidates.append(start_year)
                end_year_candidates.append(end_year)

            global_start_year = max(start_year_candidates) if start_year_candidates else None
            global_end_year = min(end_year_candidates) if end_year_candidates else None

            predict_start_year = global_end_year + 1
            predict_end_year = self.last_year_of_observations

            if global_start_year is not None and global_end_year is not None:
                target_years[target_market] = {
                    "start_year": global_start_year,
                    "end_year": global_end_year,
                    "predict_start_year": predict_start_year,
                    "predict_end_year": predict_end_year,
                }

        return target_years

    def update_training_and_prediction_years(self):
        """
        Update the training and prediction years for age/gender data.

        Notes:
            - Updates training_and_prediction_years attribute
            - Uses age_group and gender grouping columns
        """
        if not self.target_markets:
            logger.info("No target markets configured - skipping year updates")
            return

        self.training_and_prediction_years.update(self.determine_training_and_prediction_years(self.df_age_gender, ["age_group", "gender"]))

    def extend_time_series_for_group(self, df, group_by_columns):
        """
        Extend time series for each group using LASSO regression.

        Args:
            df (pd.DataFrame): Input data to extend
            group_by_columns (list): Columns to group by

        Returns:
            pd.DataFrame: Extended time series data

        Notes:
            - Uses LASSO regression for prediction
            - Handles missing data through interpolation
            - Applies predictions to each group separately
        """
        results = []
        if not self.target_markets:
            logger.info("No target markets configured - skipping time series extension")
            return df, pd.DataFrame()

        if not hasattr(self, "training_and_prediction_years") or not self.training_and_prediction_years:
            logger.info("Training and prediction years not initialized - initializing now")
            if group_by_columns == ["age_group", "gender"]:
                self.update_training_and_prediction_years()
            else:
                logger.warning(f"Unknown group_by_columns: {group_by_columns}")
                return df, pd.DataFrame()

        target_years = self.training_and_prediction_years
        if not target_years:
            logger.info("No target years available - skipping time series extension")
            return df, pd.DataFrame()

        for target_market, proxy_markets in self.target_markets.items():
            for name, group in df[df["country"].isin(proxy_markets)].groupby(group_by_columns):
                if target_market not in target_years:
                    raise ValueError(f"Market {target_market} not in target market list")

                year_info = target_years[target_market]
                start_year = year_info["start_year"]
                predict_start_year = year_info["predict_start_year"]
                predict_end_year = year_info["predict_end_year"]

                if predict_start_year >= predict_end_year:  # if condition is true it means time series does not need extension
                    return df, ""

                for year in range(predict_start_year, predict_end_year + 1):
                    train_columns = [str(y) for y in range(start_year, year)]

                    X_train = group[train_columns].dropna(axis=1, how="all").astype(float)
                    y_train = group[str(year - 1)].dropna().astype(float)
                    lasso = train_lasso(X_train, y_train, config=self.config)

                    mse_backtest, mae_backtest, mape_backtest = backtest_model(lasso, X_train, y_train)
                    results.append(
                        {
                            "target_market": target_market,
                            "group": name,
                            "year": year,
                            "MSE_backtest": mse_backtest,
                            "MAE_backtest": mae_backtest,
                            "MAPE_backtest": mape_backtest,
                        }
                    )

                    if len(group_by_columns) == 1:
                        target_series = df[(df["country"] == target_market) & (df[group_by_columns[0]] == name)].copy()
                    else:
                        target_series = df[
                            (df["country"] == target_market) & (df[group_by_columns[0]] == name[0]) & (df[group_by_columns[1]] == name[1])
                        ].copy()

                    if target_series.empty:
                        continue

                    X_test = target_series[train_columns].astype(float)
                    scaling_factor = self.config["prediction"]["preprocessing"]["scaling_factor"]
                    X_test_scaled = X_test / scaling_factor
                    y_pred_full_scaled = lasso.predict(X_test_scaled)
                    clip_predictions = self.config["prediction"]["preprocessing"]["clip_predictions"]
                    min_prediction = self.config["prediction"]["preprocessing"]["min_prediction"]
                    max_prediction = self.config["prediction"]["preprocessing"]["max_prediction"]

                    if clip_predictions:
                        y_pred_full = np.clip(
                            y_pred_full_scaled * scaling_factor,
                            min_prediction,
                            max_prediction,
                        )
                    else:
                        y_pred_full = y_pred_full_scaled * scaling_factor

                    target_series[str(year)] = y_pred_full[0]

                    if len(group_by_columns) == 1:
                        df.loc[
                            (df["country"] == target_market) & (df[group_by_columns[0]] == name),
                            str(year),
                        ] = y_pred_full[0]

                    else:
                        df.loc[
                            (df["country"] == target_market) & (df[group_by_columns[0]] == name[0]) & (df[group_by_columns[1]] == name[1]),
                            str(year),
                        ] = y_pred_full[0]

        results_df = pd.DataFrame(results)
        logger.info(
            f"Minimum MAE: {results_df.groupby('target_market')['MAE_backtest'].min()},\n"
            f"Maximum MAE: {results_df.groupby('target_market')['MAE_backtest'].max()},\n"
            f"Mean MAE: {results_df.groupby('target_market')['MAE_backtest'].mean()}"
        )
        logger.info(
            f"Minimum MAE: {results_df.groupby('target_market')['MSE_backtest'].min()},\n"
            f"Maximum MAE: {results_df.groupby('target_market')['MSE_backtest'].max()},\n"
            f"Mean MAE: {results_df.groupby('target_market')['MSE_backtest'].mean()}"
        )
        return df, results_df

    def _validate_methodology(self, methodology):
        """Validate methodology against enum"""
        valid_methodologies = self.config["defaults"]["methodology_enum"]
        if not valid_methodologies:
            raise ValueError("No methodology_enum defined in config file")
            exit()
        if methodology not in valid_methodologies:
            raise ValueError(f"Invalid methodology '{methodology}' in config file. " f"Valid options: {', '.join(valid_methodologies)}")
            exit()

    def create_train_data(self):
        """
        Create training data for model fitting.

        Returns:
            pd.DataFrame: Processed training data

        Notes:
            - Extracts age groups and gender information
            - Standardizes data format
            - Handles missing values
        """
        self.df_age_gender["age_group"] = self.df_age_gender["Individual type"].apply(
            lambda x: "_".join(map(str, extract_age_groups(x)))
        )  # Get age groups
        self.df_age_gender = self.df_age_gender.apply(extract_gender, axis=1)  # Get gender
        self.df_age_gender["composite_index"] = (
            self.df_age_gender["country"] + " " + self.df_age_gender["age_group"] + " " + self.df_age_gender["gender"]
        )

        fill_missing_with = self.config["prediction"]["preprocessing"]["fill_missing_with"]

        self.df_age_gender = self.df_age_gender.replace(fill_missing_with, np.nan)
        # extend time series
        self.df_age_gender, backtest_results_age_gender = self.extend_time_series_age_gender()
        df_processed_gender = preprocess_df(self.df_age_gender, "df_age_gender", div=100)
        df_processed_gender.columns = [int(i) for i in df_processed_gender.columns.tolist()]  # Columns to int (year columns)

        # Initialize prediction index
        n_predictions = self.config["modules"]["age_gender"]["settings"]["n_predictions"]
        preds, pred_idx = init_preds(df_processed_gender, n_predictions)
        self.pred_idx = pred_idx

        gender_data_start_year = self.config["prediction"]["time_series"]["gender_data_start_year"]

        self.train_columns_gender_split = (
            pd.period_range(start=str(gender_data_start_year), end=str(self.last_year_of_observations), freq="Y").to_timestamp().year.values
        )

        self.full_data_gender = df_processed_gender.copy()
        self.full_data_gender.set_index(self.df_age_gender["composite_index"], inplace=True)
        self.train_data_gender = self.full_data_gender[self.train_columns_gender_split]

    def extend_time_series_age_gender(self):
        """
        Extend time series for gender and age group data.

        Notes:
            - Updates training years for gender/age dataset
            - Extends time series using group-specific approach
        """
        self.update_training_and_prediction_years()
        return self.extend_time_series_for_group(self.df_age_gender, ["age_group", "gender"])

    def go_back_two_years(self):
        """
        Shift prediction window back by two years for backtesting.

        Notes:
            - Adjusts last_year_of_observations
            - Updates training and prediction years
        """
        adjustment_year_offset = self.config["prediction"]["preprocessing"]["adjustment_year_offset"]
        self.last_year_of_observations = self.last_year_of_observations - adjustment_year_offset
        self.year_of_predictions = self.year_of_predictions - adjustment_year_offset

    def go_forward_two_years(self):
        """
        Shift prediction window forward by two years.

        Notes:
            - Restores original last_year_of_observations
            - Updates training and prediction years
        """
        adjustment_year_offset = self.config["prediction"]["preprocessing"]["adjustment_year_offset"]
        self.last_year_of_observations = self.last_year_of_observations + adjustment_year_offset
        self.year_of_predictions = self.year_of_predictions + adjustment_year_offset

    def combine_weighted(self, weights):
        """
        Combine predictions using weighted average.

        Args:
            weights (list): List of weights for each prediction method

        Returns:
            pd.DataFrame: Combined predictions

        Notes:
            - Applies weights to different prediction methods
            - Ensures weights sum to 1
            - Clips final predictions to valid range
        """
        previous_year_offset = self.config["prediction"]["preprocessing"]["previous_year_offset"]

        self.merged_all[str(self.year_of_predictions - previous_year_offset) + "_combined"] = np.sum(
            (
                weights
                * self.merged_all[
                    [
                        str(self.year_of_predictions - previous_year_offset) + "_rw",
                        str(self.year_of_predictions - previous_year_offset) + "_theta",
                        str(self.year_of_predictions - previous_year_offset) + "_logistic",
                    ]
                ].values
            ),
            axis=1,
        )

        self.merged_all[str(self.year_of_predictions) + "_combined"] = np.sum(
            (
                weights
                * self.merged_all[
                    [
                        str(self.year_of_predictions) + "_rw",
                        str(self.year_of_predictions) + "_theta",
                        str(self.year_of_predictions) + "_logistic",
                    ]
                ].values
            ),
            axis=1,
        )

        self.merged_all["composite_index"] = self.merged_all["country"] + " " + self.merged_all["age"] + " " + self.merged_all["gender"]

    def _generate_theta_predictions(self):
        """
        Generate theta method predictions for all observations.

        Returns:
            np.array: Theta predictions for all observations
        """
        theta_preds = self.train_data_gender.apply(
            lambda x: fit_predict_theta(x, self.pred_idx, config=self.config, last_year_of_observations=self.last_year_of_observations),
            axis=1,
        )
        return np.array(theta_preds)

    def _generate_logistic_predictions_younger_age_groups(self):
        """
        Generate logistic predictions for younger age groups and extract minimum values.

        Returns:
            tuple: (logistic_preds_younger_age_groups, min_values) where:
                - logistic_preds_younger_age_groups: Predictions for younger age groups
                - min_values: Minimum values per country-gender combination
        """
        # First predict for younger age groups so that we can get the max value for the older age groups
        older_age_groups = self.config["modules"]["age_gender"]["settings"]["older_age_groups"]
        younger_age_groups_df = self.train_data_gender[~self.train_data_gender.index.str.contains("|".join(older_age_groups))]

        logistic_preds_younger_age_groups = younger_age_groups_df.apply(
            lambda x: fit_predict_logistic(
                x, np.arange(self.last_year_of_observations + 1, self.year_of_predictions + 1), self.last_year_of_observations, self.config
            ),
            axis=1,
        )

        # Extract the max value from the younger age groups to use in the older age groups
        logistic_preds_younger_age_groups_copy = logistic_preds_younger_age_groups.copy()
        # logistic_preds_younger_age_groups_copy['country'] = logistic_preds_younger_age_groups_copy.index.str.split(" ").str[0]
        logistic_preds_younger_age_groups_copy["country"] = (
            logistic_preds_younger_age_groups_copy.index.str.strip().str.replace(r"\s+", " ", regex=True).str.rsplit(" ", n=2).str[0]
        )
        logistic_preds_younger_age_groups_copy["gender"] = logistic_preds_younger_age_groups_copy.index.str.split(" ").str[-1]
        logistic_preds_younger_age_groups_copy.reset_index(drop=True, inplace=True)
        min_values = logistic_preds_younger_age_groups_copy.groupby(["country", "gender"]).min()

        return logistic_preds_younger_age_groups, min_values

    def _generate_logistic_predictions_older_age_groups(self, min_values):
        """
        Generate logistic predictions for older age groups using minimum values from younger age groups.

        Args:
            min_values: Minimum values per country-gender combination from younger age groups

        Returns:
            pd.DataFrame: Logistic predictions for older age groups
        """
        older_age_groups = self.config["modules"]["age_gender"]["settings"]["older_age_groups"]

        # Predict for the older age groups
        # Add a new column to the older age groups dataframe with the max value obtained by the younger age groups
        older_age_groups_df = self.train_data_gender[self.train_data_gender.index.str.contains("|".join(older_age_groups))]
        parts = older_age_groups_df.index.to_series().str.rsplit(" ", n=2, expand=True)
        parts.columns = ["country", "age_group", "gender"]
        older_age_groups_df_with_keys = older_age_groups_df.join(parts[["country", "gender"]])
        older_age_groups_df_with_K = older_age_groups_df_with_keys.join(min_values[self.year_of_predictions].rename("K"), on=["country", "gender"])
        older_age_groups_df_with_K = older_age_groups_df_with_K.drop(columns=["country", "gender"])

        # Now predict
        logistic_preds_older_age_groups = older_age_groups_df_with_K.apply(
            lambda x: fit_predict_logistic_older_age_groups(
                x, np.arange(self.last_year_of_observations + 1, self.year_of_predictions + 1), self.last_year_of_observations, self.config
            ),
            axis=1,
        )

        return logistic_preds_older_age_groups

    def _process_logistic_predictions(self, logistic_preds_younger_age_groups, logistic_preds_older_age_groups):
        """
        Process and sort logistic predictions from both younger and older age groups.

        Args:
            logistic_preds_younger_age_groups: Predictions for younger age groups
            logistic_preds_older_age_groups: Predictions for older age groups

        Returns:
            pd.DataFrame: Processed and sorted logistic predictions
        """
        logistic_preds_all = pd.concat([logistic_preds_younger_age_groups, logistic_preds_older_age_groups], axis=0)

        # Sort the logistic predictions to match the index sorting we have in the other prediction dataframes
        parts = logistic_preds_all.index.to_series().str.rsplit(" ", n=2, expand=True)
        parts.columns = ["country", "age_group", "gender"]
        parts["age_start"] = parts["age_group"].str.split("_").str[0].astype(int)
        gender_cat = pd.Categorical(parts["gender"].str.lower(), categories=["male", "female"], ordered=True)
        logistic_preds = (
            logistic_preds_all.assign(country=parts["country"], gender=gender_cat, age_start=parts["age_start"])
            .sort_values(by=["country", "gender", "age_start"])
            .drop(columns=["country", "gender", "age_start"])
        )

        logistic_preds_df = logistic_preds
        logistic_preds_df.columns = [f"{year}_logistic" for year in logistic_preds.columns]
        logistic_preds_df["country_age"] = self.full_data_gender.index.str.split(" ").str[:-1].str.join(" ")
        logistic_preds_df["gender"] = logistic_preds.index.str.split(" ").str[-1]

        return logistic_preds_df

    def _generate_random_walk_predictions(self):
        """
        Generate random walk predictions for two consecutive years.

        Returns:
            pd.DataFrame: Random walk predictions with proper indexing
        """
        rw_preds_first_year = self.train_data_gender.apply(lambda x: prediction_gender_data(x, config=self.config), axis=1)
        rw_preds_first_year.name = f"{self.year_of_predictions-1}_rw"
        df_copy = self.train_data_gender.copy()
        df_copy[f"{self.year_of_predictions-1}"] = rw_preds_first_year.values
        rw_preds_second_year = df_copy.apply(lambda x: prediction_gender_data(x, config=self.config), axis=1)
        rw_preds_second_year.name = f"{self.year_of_predictions}_rw"
        rw_preds_df = pd.merge(
            rw_preds_first_year,
            rw_preds_second_year,
            how="inner",
            on="composite_index",
            suffixes=(f"{self.year_of_predictions-1}_rw", f"{self.year_of_predictions}_rw"),
        )
        rw_preds_df.index = rw_preds_first_year.index

        # Extract gender information from the index of rw_preds to maintain correct order
        rw_preds_df["country_age"] = rw_preds_first_year.index.str.split(" ").str[:-1].str.join(" ")
        rw_preds_df["gender"] = rw_preds_first_year.index.str.split(" ").str[-1]

        return rw_preds_df

    def _merge_and_preprocess_predictions(self, theta_preds, logistic_preds_df, rw_preds_df):
        """
        Merge all prediction dataframes and apply preprocessing (clipping, country name fixes).

        Args:
            theta_preds: Theta predictions array
            logistic_preds_df: Logistic predictions dataframe
            rw_preds_df: Random walk predictions dataframe

        Returns:
            pd.DataFrame: Merged and preprocessed predictions
        """
        theta_preds = np.array(theta_preds)
        theta_preds_df = pd.DataFrame()

        clip_predictions = self.config["prediction"]["preprocessing"]["clip_predictions"]
        min_prediction = self.config["prediction"]["preprocessing"]["min_prediction"]
        max_prediction = self.config["prediction"]["preprocessing"]["max_prediction"]

        if clip_predictions:
            theta_preds_df[str(self.last_year_of_observations + 1) + "_theta"] = np.clip(theta_preds[:, 0], min_prediction, max_prediction)
            theta_preds_df[str(self.year_of_predictions) + "_theta"] = np.clip(theta_preds[:, 1], min_prediction, max_prediction)
            rw_preds_df[str(self.last_year_of_observations + 1) + "_rw"] = np.clip(
                rw_preds_df[f"{self.last_year_of_observations + 1}_rw"].values, min_prediction, max_prediction
            )
            rw_preds_df[str(self.year_of_predictions) + "_rw"] = np.clip(
                rw_preds_df[f"{self.year_of_predictions}_rw"].values, min_prediction, max_prediction
            )
        else:
            theta_preds_df[str(self.last_year_of_observations + 1) + "_theta"] = theta_preds[:, 0]
            theta_preds_df[str(self.year_of_predictions) + "_theta"] = theta_preds[:, 1]
            # Note: This line seems to have a bug in the original code - using rw_preds_first_year.values twice
            # Keeping the original logic but this should probably be reviewed
            rw_preds_df[str(self.last_year_of_observations + 1) + "_rw"] = rw_preds_df[f"{self.last_year_of_observations + 1}_rw"].values
            rw_preds_df[str(self.year_of_predictions) + "_rw"] = rw_preds_df[f"{self.year_of_predictions}_rw"].values

        theta_preds_df["country_age"] = self.full_data_gender.index.str.split(" ").str[:-1].str.join(" ")
        theta_preds_df["gender"] = self.full_data_gender.index.str.split(" ").str[-1]

        merged_all = pd.merge(
            logistic_preds_df,
            rw_preds_df,
            how="inner",
            on=["country_age", "gender"],
            suffixes=("_logistic", "_rw"),
        ).merge(theta_preds_df, on=["country_age", "gender"])

        # Use the gender information from rw_preds_df to maintain correct order
        merged_all["country"] = merged_all["country_age"].str.strip().str.replace(r"\s+", " ", regex=True).str.rsplit(" ", n=1).str[0]
        merged_all["age"] = merged_all["country_age"].str.extract(r"(\d{2}_\d{2})")

        return merged_all

    def generate_forecasts(self):
        """
        Generate forecasts using multiple methods.

        Returns:
            tuple: (predictions, errors) containing:
                - predictions: Dictionary of predictions from each method
                - errors: Dictionary of error metrics for each method

        Notes:
            - Implements multiple forecasting methods
            - Calculates error metrics for each method
        """
        theta_preds = self._generate_theta_predictions()
        logistic_preds_younger_age_groups, min_values = self._generate_logistic_predictions_younger_age_groups()
        logistic_preds_older_age_groups = self._generate_logistic_predictions_older_age_groups(min_values)
        logistic_preds_df = self._process_logistic_predictions(logistic_preds_younger_age_groups, logistic_preds_older_age_groups)
        rw_preds_df = self._generate_random_walk_predictions()
        self.merged_all = self._merge_and_preprocess_predictions(theta_preds, logistic_preds_df, rw_preds_df)
        return self.merged_all

    def combine_forecasts(self):
        """
        Combine forecasts from multiple methods.

        Returns:
            pd.DataFrame: Combined forecasts
        """
        # Create training data first to initialize necessary attributes
        self.create_train_data()
        self.generate_forecasts()

        previous_year_offset = self.config["prediction"]["preprocessing"]["previous_year_offset"]
        theta_weight = self.config["modules"]["age_gender"]["settings"]["theta_weight"]
        random_walk_weight = self.config["modules"]["age_gender"]["settings"]["random_walk_weight"]
        logistic_weight = self.config["modules"]["age_gender"]["settings"]["logistic_weight"]

        self.merged_all[str(self.year_of_predictions - previous_year_offset) + "_combined"] = (
            self.merged_all[str(self.year_of_predictions - previous_year_offset) + "_logistic"] * logistic_weight
            + self.merged_all[str(self.year_of_predictions - previous_year_offset) + "_theta"] * theta_weight
            + self.merged_all[str(self.year_of_predictions - previous_year_offset) + "_rw"] * random_walk_weight
        )

        self.merged_all[str(self.year_of_predictions) + "_combined"] = (
            self.merged_all[str(self.year_of_predictions) + "_logistic"] * logistic_weight
            + self.merged_all[str(self.year_of_predictions) + "_theta"] * theta_weight
            + self.merged_all[str(self.year_of_predictions) + "_rw"] * random_walk_weight
        )

        methodology = self.config["defaults"]["methodology"]
        self._validate_methodology(methodology)
        logger.info(f"Modeling Methodology used: {methodology}")
        if methodology == "ceiling":
            self.merged_all = adjust_100prc_predictions_cases(self.merged_all, self.train_data_gender, self.year_of_predictions, self.config)
        if methodology == "age_groups":
            self.merged_all = age_groups_tweak(self.merged_all, self.year_of_predictions, self.config)
        if methodology == "both":
            # We first apply the ceiling effect and the the age groups rule. If we first apply the age tweak for an older age group for which we have three consecutive
            # years at 100%, then the ceiling effect would change the tweaked value back to 100, and the age group rule would be again violated.
            # On the contrary, when we first apply the ceiling rule, we either (a) lower percentages below 100% or (b) raise them to 100% in the case
            # of three consecutive 100%. The age group rule will then further lower any percentage below its initial value, so the ceiling rule will not
            # be violated, as it states "allow 100% only if the last three consecutve years are 100%", e.g. it is sufficient condition to have 100% but it does not impose a 100%.
            self.merged_all = adjust_100prc_predictions_cases(self.merged_all, self.train_data_gender, self.year_of_predictions, self.config)
            self.merged_all = age_groups_tweak(self.merged_all, self.year_of_predictions, self.config)

        self.merged_all["composite_index"] = self.merged_all["country"] + " " + self.merged_all["age"] + " " + self.merged_all["gender"]

        return self.merged_all

    def get_forecast_results(self):
        """
        Get final forecast results.

        Returns:
            pd.DataFrame: Final combined predictions

        Notes:
            - Formats results for output
            - Applies any final transformations
            - Ensures consistent data structure
        """
        previous_year_offset = self.config["prediction"]["preprocessing"]["previous_year_offset"]

        self.merged_all = self.merged_all[
            [
                "composite_index",
                str(self.year_of_predictions - previous_year_offset) + "_combined",
                str(self.year_of_predictions) + "_combined",
            ]
        ]

        self.merged_all.rename(
            columns={
                "composite_index": "composite_index",
                str(self.year_of_predictions - previous_year_offset) + "_combined": str(self.year_of_predictions - previous_year_offset),
                str(self.year_of_predictions) + "_combined": str(self.year_of_predictions),
            },
            inplace=True,
        )

        self.final_results = self.full_data_gender.merge(self.merged_all, left_index=True, right_on="composite_index")
        first_column = self.merged_all[["composite_index"]]
        self.final_results.drop(columns=["composite_index"], inplace=True)
        self.final_results = pd.concat([first_column, self.final_results], axis=1)

        return self.final_results
