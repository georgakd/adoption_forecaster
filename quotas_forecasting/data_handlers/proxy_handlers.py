"""
Proxy Handler Module

This module handles proxy country identification and data interpolation operations.
It provides functionality for finding proxy countries based on correlation analysis
and applying various interpolation methods to fill missing data.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from quotas_forecasting.config.logging_config import get_logger

logger = get_logger(__name__)


class ProxyHandler:
    """
    A class for handling proxy country operations and data interpolation.
    """

    def __init__(self, config=None):
        """Initialize ProxyHandler"""
        self.config = config

    def correlation_heatmap(self, df, edu_level):
        """
        Generate and optionally display a correlation heatmap.

        Args:
            df (pd.DataFrame): Input data for correlation analysis
            edu_level (str): Education level to analyze

        Returns:
            pd.DataFrame: Correlation matrix

        Notes:
            - Calculates correlations between all countries
        """
        correlation = df[df.index.get_level_values(1) == edu_level].reset_index().drop("education", axis=1).set_index("country").transpose().corr()

        return correlation

    def get_proxy_countries(self, df, countries_with_missing_data, edu_levels):
        """
        Identify proxy countries for each country with missing data.

        Args:
            df (pd.DataFrame): Input data containing all countries
            countries_with_missing_data (list): List of countries needing proxies
            edu_levels (list): Education levels to analyze

        Returns:
            dict: Dictionary mapping countries to their proxy countries for each education level

        Notes:
            - Uses correlation analysis to find suitable proxies
            - Applies different correlation thresholds based on data availability
            - Returns nested dictionary {country: {edu_level: [proxy_countries]}}
        """
        corr_sec = self.correlation_heatmap(df, edu_levels[0])
        corr_ter = self.correlation_heatmap(df, edu_levels[1])

        # Get countries with missing data from config if not provided
        if not countries_with_missing_data:
            markets_config = self.get_target_markets()
            countries_with_missing_data = list(markets_config.keys())

        # Get correlation thresholds from config
        high_threshold = self.config["modules"]["education"]["settings"]["correlation_threshold"]
        low_threshold = self.config["modules"]["education"]["settings"]["correlation_threshold_low"]

        iceland_sec_proxy_countries = self.find_proxy_countries_using_correlation(
            countries_with_missing_data[0],
            corr_sec,
            high_threshold,
            edu_levels[0],
            set_exclude_countries=set(countries_with_missing_data),
        )
        north_m_sec_proxy_countries = self.find_proxy_countries_using_correlation(
            countries_with_missing_data[1],
            corr_sec,
            low_threshold,
            edu_levels[0],
            set_exclude_countries=set(countries_with_missing_data),
        )
        uni_kin_sec_proxy_countries = self.find_proxy_countries_using_correlation(
            countries_with_missing_data[2],
            corr_sec,
            high_threshold,
            edu_levels[0],
            set_exclude_countries=set(countries_with_missing_data),
        )

        iceland_ter_proxy_countries = self.find_proxy_countries_using_correlation(
            countries_with_missing_data[0],
            corr_ter,
            high_threshold,
            edu_levels[1],
            set_exclude_countries=set(countries_with_missing_data),
        )
        north_m_ter_proxy_countries = self.find_proxy_countries_using_correlation(
            countries_with_missing_data[1],
            corr_ter,
            low_threshold,
            edu_levels[1],
            set_exclude_countries=set(countries_with_missing_data),
        )
        uni_kin_ter_proxy_countries = self.find_proxy_countries_using_correlation(
            countries_with_missing_data[2],
            corr_ter,
            high_threshold,
            edu_levels[1],
            set_exclude_countries=set(countries_with_missing_data),
        )

        sec_proxy_countries = [
            iceland_sec_proxy_countries,
            north_m_sec_proxy_countries,
            uni_kin_sec_proxy_countries,
        ]
        ter_proxy_countries = [
            iceland_ter_proxy_countries,
            north_m_ter_proxy_countries,
            uni_kin_ter_proxy_countries,
        ]
        proxy_countries_dict = {
            edu_levels[0]: sec_proxy_countries,
            edu_levels[1]: ter_proxy_countries,
        }

        return proxy_countries_dict

    def find_proxy_countries_using_correlation(
        self,
        country_with_missing_data,
        correlations,
        threshold,
        edu_level,
        set_exclude_countries,
    ):
        """
        Find suitable proxy countries based on correlation thresholds.

        Args:
            country_with_missing_data (str): country for which proxies are needed
            correlations (pd.DataFrame): Correlation matrix between countries
            threshold (float): Minimum correlation threshold for proxy selection
            edu_level (str): Education level being analyzed
            set_exclude_countries (set): Set of countries to exclude from proxy selection

        Returns:
            list: List of proxy countries meeting the correlation threshold

        Notes:
            - Excludes countries specified in set_exclude_countries
            - Returns countries with correlation above threshold
            - Sorts proxies by correlation strength
        """
        logger.info(f"{country_with_missing_data} proxy countries for {edu_level} education level")
        country_with_missing_data_corrs = correlations.loc[country_with_missing_data]
        proxy_countries = set(country_with_missing_data_corrs[country_with_missing_data_corrs > threshold].index)
        proxy_countries = proxy_countries - set_exclude_countries
        country_with_missing_data_corrs = pd.DataFrame(country_with_missing_data_corrs)
        country_with_missing_data_corrs.columns = ["correlation"]

        logger.info(f"\n{country_with_missing_data_corrs[country_with_missing_data_corrs['correlation'] > threshold]}")

        return proxy_countries

    def extrapolate_with_Lasso(
        self,
        df,
        countries_with_missing_data,
        last_year_of_observations_dict,
        proxy_countries_dict,
        last_year_of_observations,
        edu_levels,
    ):
        """
        Extrapolate missing data using LASSO regression with proxy countries.

        Args:
            df (pd.DataFrame): Input data
            countries_with_missing_data (list): Countries needing extrapolation
            last_year_of_observations_dict (dict): Last available year for each country
            proxy_countries_dict (dict): Proxy countries for each target country
            last_year_of_observations (int): Overall last year of observations
            edu_levels (list): Education levels to process

        Returns:
            pd.DataFrame: DataFrame with extrapolated values for missing data

        Notes:
            - Uses LASSO regression for prediction
            - Applies predictions only to missing years
            - Maintains data integrity for non-missing years
        """

        def train_test_split_(df, target_country, proxy_countries, edu_level, last_year):
            """
            Split data into training and test sets for LASSO regression.
            """
            df = df[(df.index.get_level_values(1) == edu_level)]
            df = df.reset_index(level=1, drop=True)
            df = df.infer_objects(copy=False)
            df = df.apply(lambda x: x.interpolate(limit_direction="backward"), axis=1)
            df = df.apply(lambda x: x.interpolate(limit_direction="backward"), axis=1)
            X_train = df[(df.index.get_level_values(0).isin(proxy_countries))][range(df.columns[0], last_year + 1)]
            y_train = df[(df.index.get_level_values(0).isin(proxy_countries))][last_year + 1]
            X_test = df[(df.index.get_level_values(0) == target_country)][range(df.columns[0], last_year + 1)]

            valid_indices = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_train = X_train[valid_indices]
            y_train = y_train[valid_indices]

            if len(X_train) == 0 or len(y_train) == 0:
                return None, None, X_test

            return X_train, y_train, X_test

        def Lasso(X_train, y_train, X_test):
            """
            Fit LASSO regression model and make predictions.
            """
            X_train_np = np.array(X_train)
            y_train_np = np.array(y_train).ravel()

            n_splits = min(self.config["prediction"]["lasso"]["default_cv_folds"], len(X_train))
            reg = LassoCV(
                cv=KFold(n_splits=n_splits),
                random_state=self.config["prediction"]["lasso"]["random_state"],
                max_iter=self.config["prediction"]["lasso"]["max_iterations"],
            )
            reg.fit(X_train_np, y_train_np)

            X_test_np = np.array(X_test).reshape(1, -1)

            if np.isnan(X_test_np).any():
                X_test_series = pd.Series(X_test_np[0])
                X_test_interpolated = X_test_series.interpolate(
                    method=self.config["prediction"]["preprocessing"]["interpolation_method"],
                    limit_direction=self.config["prediction"]["preprocessing"]["interpolation_direction"],
                )
                X_test_np = X_test_interpolated.values.reshape(1, -1)

            prediction = np.round(
                reg.predict(X_test_np)[0],
                self.config["prediction"]["preprocessing"]["prediction_precision"],
            )
            return prediction

        # this method uses the prediction of 2020 to predict for 2021 then for 2023 etc, which is not ideal. Alternatives?
        for edu_level in edu_levels:
            for j, target_country in enumerate(countries_with_missing_data):
                for last_year in range(
                    last_year_of_observations_dict[edu_level][j],
                    last_year_of_observations,
                ):
                    X_train, y_train, X_test = train_test_split_(
                        df,
                        target_country,
                        proxy_countries_dict[edu_level][j],
                        edu_level,
                        last_year,
                    )

                    if X_train is None or y_train is None:
                        logger.warning(f"No valid training data for {target_country}, education level {edu_level}, year {last_year + 1}")
                        continue

                    prediction = Lasso(X_train, y_train, X_test)
                    df.at[(target_country, edu_level), last_year + 1] = prediction
        return df

    def apply_LASSO_for_the_three_countries_with_missing_data(self, df, countries_with_missing_data, edu_levels, last_year_of_observations):
        """
        Apply LASSO regression to predict missing data for specific countries.

        Args:
            df (pd.DataFrame): Input data
            countries_with_missing_data (list): Countries needing predictions
            edu_levels (list): Education levels to process
            last_year_of_observations (int): Last year of available data

        Returns:
            pd.DataFrame: DataFrame with filled missing data

        Notes:
            - Specifically handles Iceland, North Macedonia, and UK
            - Uses proxy countries for prediction
            - Returns complete dataset with filled values
        """
        proxy_countries_dict = self.get_proxy_countries(df, countries_with_missing_data, edu_levels)
        last_year_of_observations_sec = [
            df.loc[countries_with_missing_data[0], edu_levels[0]].last_valid_index(),
            df.loc[countries_with_missing_data[1], edu_levels[0]].last_valid_index(),
            df.loc[countries_with_missing_data[2], edu_levels[0]].last_valid_index(),
        ]
        last_year_of_observations_ter = [
            df.loc[countries_with_missing_data[0], edu_levels[1]].last_valid_index(),
            df.loc[countries_with_missing_data[1], edu_levels[1]].last_valid_index(),
            df.loc[countries_with_missing_data[2], edu_levels[1]].last_valid_index(),
        ]
        last_year_of_observations_dict = {
            edu_levels[0]: last_year_of_observations_sec,
            edu_levels[1]: last_year_of_observations_ter,
        }

        online_pop_by_edu_ratios_st_extrap = self.extrapolate_with_Lasso(
            df,
            countries_with_missing_data,
            last_year_of_observations_dict,
            proxy_countries_dict,
            last_year_of_observations,
            edu_levels,
        )

        return online_pop_by_edu_ratios_st_extrap


    def get_target_markets(self):
        """get target_markets from config, returns empty dict if not present"""
        return self.config.get("target_markets", {})
