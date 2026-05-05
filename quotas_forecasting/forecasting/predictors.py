"""
This module implements prediction models and algorithms for forecasting internet usage data.
It provides functionality for both Education and Age_Gender modules with unified interfaces.

Key features:
- Random walk prediction with smoothing
- Theta forecasting with configurable parameters
- LASSO regression for proxy-based predictions
- Quotas-based forecasting with error learning
- Ensemble model combining multiple prediction methods
- Visualization tools for prediction results
"""

import numpy as np
import pandas as pd
import warnings

# Suppress specific warnings from statsmodels Holt-Winters
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from quotas_forecasting.forecasting.quotas_forecasting import Quotas_Forecasting
from quotas_forecasting.forecasting.utils import (
    preprocess,
    predict_random_walk,
    fit_predict_theta,
    split_composite_index,
)
from quotas_forecasting.config.logging_config import get_logger


logger = get_logger(__name__)
warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning, module="statsmodels")


# ============================================================================
# EDUCATION MODULE PREDICTOR
# ============================================================================


class EducationPredictor:
    """
    A class for implementing education-related internet usage predictions.

    This class handles the education module's specific prediction logic including
    education level processing, LASSO regression for proxy countries, and
    predictions for both 16-64 and 16-74 age groups.
    """

    def __init__(self, mode="local", pred_year=2025, last_year=2023, loader=None, config=None):
        """
        Initialize the EducationPredictor.

        Args:
            mode (str): "local" or "prod" - determines data source
            pred_year (int): Target year for predictions
            last_year (int): Last year of available data
            loader (Loader): Loader instance for reading files
            config (dict): Configuration dictionary
        """
        self.mode = mode
        self.pred_year = pred_year
        self.last_year = last_year
        self.loader = loader
        self.config = config

    def train_test_split_(self, df, target_country, proxy_countries, edu_level, last_year):
        """
        Split data into training and test sets for LASSO regression.

        Args:
            df (pd.DataFrame): Input data
            target_country (str): country to predict for
            proxy_countries (list): List of proxy countries to use
            edu_level (str): Education level to consider ("secondary" or "tertiary")
            last_year (int): Last year of data to use

        Returns:
            tuple: (X_train, y_train, X_test) containing:
                - X_train (pd.DataFrame): Training features from proxy countries
                - y_train (pd.Series): Training targets from proxy countries
                - X_test (pd.DataFrame): Test features (synthetic based on proxy countries)

        Notes:
            - Filters data for specified education level
            - Interpolates missing values in both directions
            - Creates synthetic test set if target country is missing
        """
        df = df[(df.index.get_level_values(1) == edu_level)]
        df = df.reset_index(level=1, drop=True)
        df = df.apply(lambda x: x.interpolate(limit_direction="backward"), axis=1)
        df = df.apply(lambda x: x.interpolate(limit_direction="backward"), axis=1)  # ?

        # Get training data from proxy countries
        X_train = df[(df.index.get_level_values(0).isin(proxy_countries))][range(df.columns[0], last_year + 1)]

        # For y_train, we need to use the last available year in the dataset
        # since we're predicting for future years that don't exist yet
        available_years = df.columns
        last_available_year = max(available_years)
        y_train = df[(df.index.get_level_values(0).isin(proxy_countries))][last_available_year]

        # Check if target country exists in the data
        if target_country in df.index.get_level_values(0):
            # Target country exists, use its data
            X_test = df[(df.index.get_level_values(0) == target_country)][range(df.columns[0], last_year + 1)]
        else:
            # Target country doesn't exist, create synthetic test set
            # Use the average of proxy countries as a baseline
            X_test = X_train.mean(axis=0).to_frame().T
            X_test.index = [target_country]

        return X_train, y_train, X_test

    def train_test_split_UK(self, df, first_year_of_obs_for_UK, last_year_of_observations, eduLevel, countries):
        """
        Prepares training and test datasets specifically for UK predictions.

        Args:
            df: Input DataFrame with all countries' data
            first_year_of_obs_for_UK: Starting year for UK observations
            last_year_of_observations: Last year of available observations
            eduLevel: Education level to filter for
            countries: List of countries to include in training

        Returns:
            tuple: (X_train, y_train, X_test) prepared datasets
        """
        # filter for 2010 onwards
        years_with_oobservations = np.arange(first_year_of_obs_for_UK, last_year_of_observations + 1)
        # Create a training set for the UK only, selecting proximate countries
        df = preprocess(df.T.astype("float")).T
        condition_countries = df.index.get_level_values(0).isin(countries)
        condition_eduLevel = df.index.get_level_values(1) == eduLevel
        X_train = df[condition_countries & condition_eduLevel][years_with_oobservations]
        y_train = df[condition_countries & condition_eduLevel][last_year_of_observations + 1]
        X_test = df[(df.index.get_level_values(0) == "United Kingdom") & (df.index.get_level_values(1) == eduLevel)]
        # interpolate for the missing values (* must transpose first *)
        X_train = preprocess(X_train.T.astype("float")).T
        # drop education index
        X_train = X_train.droplevel("education")
        X_test = X_test.droplevel("education")
        X_test = X_test[years_with_oobservations]
        return X_train, y_train, X_test

    def Lasso(self, X_train, y_train, X_test):
        """
        Fit LASSO regression model and make predictions.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
            X_test (pd.DataFrame): Test features

        Returns:
            float: Predicted value from LASSO regression

        Notes:
            - Uses LassoCV with cross-validation for optimal regularization
            - Number of CV splits is min(5, number of samples)
            - Returns rounded prediction to 6 decimal places
        """
        n_splits = min(5, len(X_train))  # Ensure n_splits doesn't exceed the number of samples
        lasso_config = self.config["prediction"]["lasso"]

        reg = LassoCV(cv=KFold(n_splits=n_splits), random_state=lasso_config["random_state"], max_iter=lasso_config["max_iterations"])

        reg.fit(np.array(X_train), np.array(y_train).reshape(y_train.shape[0], 1))

        prediction = np.round(reg.predict(np.array(X_test).reshape(1, -1))[0], self.config["prediction"]["preprocessing"]["prediction_precision"])
        return prediction

    def combine_regressors(self, X_train, y_train, X_test, Lasso_flag):
        """
        Combines multiple regression models for ensemble prediction.

        Args:
            X_train: Training features
            y_train: Training target values
            X_test: Test features
            Lasso_flag: Whether to include Lasso Regression

        Returns:
            float: Average prediction from enabled models
        """
        avg = 0
        n_regr = 0
        if Lasso_flag:
            prediction_level_3 = self.Lasso(X_train, y_train, X_test)
            avg += prediction_level_3
            n_regr += 1
        return avg / n_regr

    def backtest_UK(
        self,
        countries,
        eduLevel,
        df,
        X_train,
        X_test,
        last_year_of_observations,
        Lasso_flag,
        target_country="United Kingdom",  # Default for backward compatibility
    ):
        """
        Performs backtesting of predictions for UK data.

        Args:
            countries: List of countries to include in training
            eduLevel: Education level to test
            df: Input DataFrame
            X_train: Training features
            X_test: Test features
            last_year_of_observations: Last year of observations
        """
        df = preprocess(df.T.astype("float")).T
        condition_countries = df.index.get_level_values(0).isin(countries)
        condition_eduLevel = df.index.get_level_values(1) == eduLevel

        backtest_start_year = self.config["prediction"]["time_series"]["backtest_start_year"]
        min_training_year = self.config["prediction"]["time_series"]["min_training_year"]

        for year in np.arange(backtest_start_year, last_year_of_observations + 1):
            X_train_ = X_train[np.arange(min_training_year, year)]
            y_train_ = df[condition_countries & condition_eduLevel][year]
            X_test_ = X_test.loc[target_country][np.arange(min_training_year, year)]
            y_test_ = df.loc[target_country, eduLevel][year]  # noqa: F841

            n_regr = 0
            avg = 0
            if Lasso_flag:
                prediction = round(self.Lasso(X_train_, y_train_, X_test_), 6)
                avg += prediction
                n_regr += 1

            avg = avg / n_regr

    def predict(self, df, last_year_of_observations, year_of_predictions):
        """
        Main prediction function for all education levels

        Args:
            df: Input DataFrame with historical data (should be complete after missing data handling)
            last_year_of_observations: Last year of available observations
        """
        logger.info(f"Starting education predictions for years {last_year_of_observations + 1} to {self.pred_year}")
        logger.info(f"Input data shape: {df.shape}")

        edu_levels = self.config["modules"]["education"]["settings"]["education_levels"]  # noqa: F841
        start_year = self.config["prediction"]["time_series"]["start_year"]
        df.index = df.index.map(lambda x: f"{x[0]}_{x[1]}")
        df.index.name = "composite_index"

        rw_preds_first_year = df.apply(lambda x: predict_random_walk(x, config=self.config), axis=1)
        rw_preds_first_year.name = f"{year_of_predictions-1}_rw"
        df_copy = df.copy()
        df_copy[f"{year_of_predictions-1}"] = rw_preds_first_year.values
        rw_preds_second_year = df_copy.apply(lambda x: predict_random_walk(x, config=self.config), axis=1)
        rw_preds_second_year.name = f"{year_of_predictions}_rw"
        rw_preds_df = pd.merge(
            rw_preds_first_year,
            rw_preds_second_year,
            how="inner",
            on="composite_index",
            suffixes=(f"{year_of_predictions-1}_rw", f"{year_of_predictions}_rw"),
        )
        rw_preds_df.index = rw_preds_first_year.index

        preds = pd.DataFrame(index=df.index, columns=range(start_year, year_of_predictions + 1))
        pred_idx = list(preds.columns)
        theta_preds = df.apply(
            lambda x: fit_predict_theta(x, pred_idx, config=self.config, last_year_of_observations=last_year_of_observations),
            axis=1,
        )  # theta forecast for individual observations
        theta_preds.columns = [f"{year}_theta" for year in theta_preds.columns]

        merged_all = pd.merge(
            rw_preds_df,
            theta_preds,
            how="inner",
            on="composite_index",
            suffixes=("_rw", "_theta"),
        )

        # split composite index into country and education for merged_all with predictions and df with historic data
        merged_all = merged_all.reset_index(names="composite_index")
        parts = merged_all["composite_index"].str.rsplit("_", n=1, expand=True)
        parts.columns = ["country", "education"]
        merged_all = merged_all.join(parts).drop(columns=["composite_index"]).set_index(["country", "education"])

        df = df.reset_index(names="composite_index")
        parts = df["composite_index"].str.rsplit("_", n=1, expand=True)
        parts.columns = ["country", "education"]
        df = df.join(parts).drop(columns=["composite_index"]).set_index(["country", "education"])

        # for each of the prediction years,
        for i in [1, 2]:
            df_final = pd.DataFrame(
                {
                    "country": merged_all.index.get_level_values(0).values,
                    "education": merged_all.index.get_level_values(1).values,
                    "ground_truths_last_year_of_observations": df[last_year_of_observations].values,
                    "theta_preds": merged_all[f"{last_year_of_observations + i}_theta"],
                    "random_walk_preds": merged_all[f"{last_year_of_observations + i}_rw"],
                }
            )
            df_final.set_index(["country", "education"], inplace=True)

            theta_weight = self.config["modules"]["education"]["settings"]["theta_weight"]
            random_walk_weight = self.config["modules"]["education"]["settings"]["random_walk_weight"]

            # Combine predictions with configurable weights
            df_final[f"averaged_model_{last_year_of_observations + i}"] = (
                theta_weight * df_final["theta_preds"] + random_walk_weight * df_final["random_walk_preds"]
            )

            df[f"{last_year_of_observations + i}"] = df_final[f"averaged_model_{last_year_of_observations + i}"]

        df.columns = [int(i) for i in df.columns]

        for year_col in [last_year_of_observations + 1, last_year_of_observations + 2]:
            for country in df.index.get_level_values(0).unique():
                try:
                    secondary_val = df.loc[(country, "secondary"), year_col]
                    tertiary_val = df.loc[(country, "tertiary"), year_col]
                    # check for missing values
                    if pd.isna(secondary_val) or pd.isna(tertiary_val):
                        error_msg = (
                            f"Missing prediction value for {country} {year_col}: "
                            f"secondary={secondary_val}, tertiary={tertiary_val}. "
                            f"Cannot normalize missing values."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    total = secondary_val + tertiary_val
                    if total <= 0:
                        error_msg = (
                            f"Invalid total for {country} {year_col}: "
                            f"secondary={secondary_val}, tertiary={tertiary_val}, total={total}. "
                            f"Cannot normalize with zero or negative total."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Normalize: S --> S/(S+T), T --> T/(S+T)
                    new_secondary = secondary_val / total
                    new_tertiary = tertiary_val / total

                    df.loc[(country, "secondary"), year_col] = new_secondary
                    df.loc[(country, "tertiary"), year_col] = new_tertiary

                    # make sure sum is exactly 1 (fail fast if not)
                    verify_sum = new_secondary + new_tertiary
                    if abs(verify_sum - 1.0) > 1e-10:
                        error_msg = (
                            f"Normalization failed for {country} {year_col}: "
                            f"sum = {verify_sum:.10f} (expected 1.0). "
                            f"secondary={new_secondary:.10f}, tertiary={new_tertiary:.10f}. "
                            f"This violates secondary + tertiary = 1 hard constraint."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                except (KeyError, IndexError) as e:
                    error_msg = f"KeyError/IndexError for {country} {year_col}: {e}. " f"Cannot access prediction values for normalization."
                    logger.error(error_msg)
                    raise ValueError(error_msg) from e

        return df


# ============================================================================
# age_gender MODULE PREDICTORS
# ============================================================================


class AgeGenderPredictor:
    """
    Predictor class for age_gender module that uses Quotas_Forecasting internally.

    This class serves as a wrapper around the Quotas_Forecasting class,
    providing a unified interface for the age_gender module predictions.
    """

    def __init__(self, mode="local", pred_year=2025, last_year=2023, loader=None, config=None):
        """
        Initialize the AgeGenderPredictor.

        Args:
            mode (str): "local" or "prod" - determines data source
            pred_year (int): Target year for predictions
            last_year (int): Last year of available data
            loader (Loader): Loader instance for reading files
            config (dict): Configuration dictionary
        """
        self.mode = mode
        self.pred_year = pred_year
        self.last_year = last_year
        self.loader = loader
        self.config = config

    def run_with_data(self, df_age_gender, last_year_of_observations):
        """
        Run the Age_Gender prediction pipeline using pre-loaded data.
        This method accepts pre-loaded DataFrames (consistent with education pipeline).

        Args:
            df_age_gender (pd.DataFrame): Pre-loaded age_gender data
            **kwargs: Additional arguments

        Returns:
            pd.DataFrame: Prediction results
        """
        target_markets = self.config.get("target_markets", {})

        qf = Quotas_Forecasting(
            df_age_gender=df_age_gender, last_year_of_observations=last_year_of_observations, year_of_predictions=self.pred_year, config=self.config
        )

        if target_markets:
            qf.set_target_market(target_markets)

        frame = qf.run()

        frame[["country", "Individual type"]] = frame.apply(split_composite_index, axis=1)
        columns_reshuffle = list(frame.columns[-2:]) + list(frame.columns[1:-2])
        frame = frame[columns_reshuffle]

        return frame


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def get_predictor(module_type, mode="local", pred_year=None, last_year=None, loader=None, config=None):
    """
    Factory function to get the appropriate predictor for the module type.

    Args:
        module_type (str): "education" or "age_gender"
        mode (str): "local" or "prod"
        pred_year (int): Target year for predictions
        last_year (int): Last year of available data
        loader (Loader): Loader instance for reading files
        config (dict): Configuration dictionary to pass to predictors

    Returns:
        EducationPredictor or AgeGenderPredictor: Appropriate predictor instance
    """
    if module_type == "education":
        return EducationPredictor(
            mode=mode,
            pred_year=pred_year,
            last_year=last_year,
            loader=loader,
            config=config,
        )
    elif module_type == "age_gender":
        return AgeGenderPredictor(
            mode=mode,
            pred_year=pred_year,
            last_year=last_year,
            loader=loader,
            config=config,
        )
    else:
        raise ValueError(f"Unknown module type: {module_type}. Must be 'education' or 'age_gender'")
