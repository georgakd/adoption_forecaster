"""
Shared utility functions for data processing.
Used by both Education and Age_Gender modules.
"""

import pandas as pd
import numpy as np
import datetime as dt
import re
import random
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sktime.forecasting.theta import ThetaForecaster as _ThetaForecaster
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
from quotas_forecasting.config.logging_config import get_logger

# Suppress specific warnings from statsmodels Holt-Winters
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module="statsmodels",
)


logger = get_logger(__name__)


def init_preds(data, n_steps):
    """
    Initialize a predictions DataFrame with expanded index based on input data.

    Args:
        data (pd.DataFrame): Input data to base predictions on
        n_steps (int): Number of future steps to predict

    Returns:
        tuple: (preds, pred_idx) where:
            - preds (pd.DataFrame): Empty DataFrame with expanded index
            - pred_idx (list): List of prediction indices
    """
    cols = _expand_idx(data.columns, n_steps)
    preds = pd.DataFrame(index=data.index, columns=cols)
    pred_idx = preds.columns.tolist()
    return preds, pred_idx


def _cast(idx, cast):
    """
    Attempt to cast an index to a specified type.

    Args:
        idx: Index to cast
        cast (str): Type to cast to ('date', 'int', or 'float')

    Returns:
        tuple: (cast_index, success_flag) where:
            - cast_index: Casted index if successful, None if failed
            - success_flag (bool): True if casting succeeded, False otherwise
    """
    try:
        if cast == "date":
            idx_cast = [pd.to_datetime(d, infer_datetime_format=True) for d in idx]
        elif cast == "int":
            idx_cast = [int(d) for d in idx]
        elif cast == "float":
            idx_cast = [float(d) for d in idx]
        return idx_cast, True
    except Exception as e:
        logger.warning(f"Failed to cast index as: {e}")
        return None, False


def _expand_idx(idx, n_steps):
    """
    Expand an index by n_steps into the future, maintaining the same interval pattern.

    Args:
        idx: Index to expand
        n_steps (int): Number of steps to expand by

    Returns:
        list: Expanded index

    Raises:
        TypeError: If index cannot be auto-cast to a supported type
    """
    recast = False
    cast = ["date", "int", "float"]
    dtype = str(idx.dtype)
    ctypes = [c in dtype for c in cast]
    if not any(ctypes):
        for c in cast:
            idx_cast, success = _cast(idx, c)
            if success:
                break
            if c == cast[-1]:
                raise TypeError(f"Unable to auto cast index, please provide an index that is of type {cast}")
    else:
        c = cast[np.argmax(ctypes)]
        idx_cast = idx.tolist()
        recast = True
    if c == "date":
        avg_diff = np.mean(np.diff([d.timestamp() for d in idx_cast]))
    else:
        avg_diff = np.mean(np.diff(idx_cast))
    last_idx = np.max(idx_cast)
    for n in np.arange(n_steps):
        if c == "date":
            last_idx = dt.datetime.fromtimestamp(last_idx.timestamp() + avg_diff)
        else:
            last_idx += avg_diff
        idx_cast.append(last_idx)
    if recast:
        idx_cast, _ = _cast(idx_cast, c)
    return idx_cast


def preprocess_df(df, df_name, div=None, index_col=None):
    """
    Preprocess a dataframe: fix schema, standardize column names, standardize country names,
    set index columns, process data, and handle division if specified.

    Args:
        df (pd.DataFrame): Dataframe to preprocess
        df_name (str): Name of the dataframe (for error messages)
        div (int/float, optional): Value to divide the dataframe by
        index_col (str/list, optional): Column(s) to set as index
    Returns:
        pd.DataFrame: Preprocessed dataframe

    Notes:
        - Handles comma removal in numeric values
        - Converts year columns to integers
        - Optionally divides values by specified factor
        - Sets specified columns as index
    """
    if index_col:
        if isinstance(index_col, str):
            index_col = [index_col]

        for idx_col in index_col:
            assert idx_col in df.columns, f"Index: {idx_col} not found in df {df_name}"

        df.set_index(index_col, inplace=True)

    # Convert to string and replace commas
    df = df.astype(str)
    df = df.apply(lambda x: x.str.replace(",", ""))

    # for col in df.columns:
    #     df.loc[:, col] = df.loc[:, col].str.replace(",", "")

    # Process the data
    df_processed = df.replace(":", np.nan).dropna(how="all", axis=0).dropna(how="all", axis=1).filter(regex="^20[0-9][0-9]$").astype(float).copy()

    df_processed.columns = [int(c) for c in df_processed.columns]  # this is not needed
    if isinstance(div, (int, float)):
        return df_processed / div
    return df_processed


def preprocess(series):
    """
    Interpolate missing values in a time series between first and last valid indices.

    Args:
        series (pd.Series): Time series to preprocess

    Returns:
        pd.Series: Preprocessed time series with interpolated values
    """
    start_idx = series.first_valid_index()
    end_idx = series.last_valid_index()
    series = series.loc[start_idx:end_idx].interpolate()
    return series


def extract_age_groups(text):
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


def extract_gender(row):
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


def split_composite_index(row):
    """
    Split composite index into Country and Individual type components.

    Args:
        row (pd.Series): DataFrame row containing 'composite_index' field

    Returns:
        pd.Series: Series containing split components:
            - Country: Extracted country name
            - Individual type: Formatted demographic information

    Notes:
        - Splits on spaces, taking all but last two parts as country
        - Formats individual type with gender first, then age range
        - Converts age group format from '16_24' to '16 to 24'
    """
    parts = row["composite_index"].split()
    country = " ".join(parts[:-2])  # All parts except the last two
    indiv_list = parts[-2:]  # last two elements i.e. age group (e.g. 16_24) and gender male/female
    indiv_list.reverse()  # reverse list to bring gender first
    individual_type = ", ".join(indiv_list)
    individual_type = individual_type.replace("_", " to ").capitalize()
    return pd.Series([country, individual_type])


# ============================================================================
# PREDICTION UTILITY FUNCTIONS
# ============================================================================


def predict_random_walk(series, config=None):
    """
    Predict the next value using a smoothed random walk approach.

    Args:
        series (pd.Series): Time series data to predict from
        config (dict): Configuration dictionary

    Returns:
        float: Predicted value based on smoothed random walk

    Notes:
        1. Applies preprocessing to fill NaN values through interpolation
        2. Calculates differences between consecutive years
        3. Uses median of last three years' differences for prediction
    """

    series = preprocess(series.astype("float"))

    lookback_period = config["prediction"]["random_walk"]["lookback_period"]
    smoothed_series_drift = series.diff()[-lookback_period:].median()
    random_walk_pred_individuals_smoothed = series.values[-1] + smoothed_series_drift
    return random_walk_pred_individuals_smoothed


def fit_predict_theta(series, pred_idx, config=None, last_year_of_observations=None):
    """
    Fit and predict using the Theta forecasting method.

    Args:
        series (pd.Series): Time series data to fit and predict from
        pred_idx (list): Index for predictions
        config (dict): Configuration dictionary

    Returns:
        pd.Series: Last two predictions from the Theta model

    Notes:
        - Uses ThetaForecaster without seasonality
        - Preprocesses data and handles missing values
        - Clips predictions to [0, 100] range
        - Returns only last two predictions
    """
    series = preprocess(series.astype("float"))

    # Create a copy for fitting with proper index for sktime compatibility
    series_for_fit = series.copy()
    min_prediction = config["prediction"]["preprocessing"]["min_prediction"]
    max_prediction = config["prediction"]["preprocessing"]["max_prediction"]
    # Convert index to RangeIndex if it's not already a compatible type
    if not isinstance(series_for_fit.index, (pd.RangeIndex, pd.PeriodIndex, pd.DatetimeIndex)):
        series_for_fit.index = pd.RangeIndex(
            start=last_year_of_observations - len(series_for_fit),
            stop=last_year_of_observations,
        )

    theta_config = config["prediction"]["theta"]
    return_last_n = theta_config["return_last_n"]
    # Filter pred_idx to only include future years (greater than the last valid index)
    future_pred_idx = [year for year in pred_idx if year > series.last_valid_index()]
    pred_idx_individuals = pd.Index(future_pred_idx)
    n_forward_steps = pred_idx_individuals.shape[0]
    forward_steps = np.arange(n_forward_steps) + 1
    thetaforecaster = _ThetaForecaster(deseasonalize=theta_config["deseasonalize"], sp=theta_config["seasonal_period"])
    thetaforecaster.fit(series_for_fit)
    theta_pred = thetaforecaster.predict(forward_steps).clip(lower=min_prediction, upper=max_prediction)

    theta_pred_with_years = pd.Series(theta_pred.values, index=future_pred_idx[: len(theta_pred)], name=series.name)

    return theta_pred_with_years[-return_last_n:]


def logistic_function(t, K, r, t0):
    """
    Standard logistic function.
    Args:
        t (array-like): Time index
        K (float): Carrying capacity
        r (float): Growth rate
        t0 (float): Midpoint of the S-curve
    Returns:
        np.array: Logistic values at each time point
    """
    # Handle overflow in exp by clipping the exponent to prevent numerical issues
    exponent = -r * (t - t0)

    # Clip exponent to prevent overflow (exp(-700) ≈ 0, exp(700) ≈ inf)
    # This maintains numerical stability while preserving function behavior
    exponent_clipped = np.clip(exponent, -700, 700)

    return K / (1 + np.exp(exponent_clipped))


def fit_predict_logistic(series, pred_idx, last_year_of_observations, config=None):
    """
    Fit logistic growth curve to a series and predict future values.
    Args:
        series (pd.Series): Historical time series
        pred_idx (list): Future indices for predictions
        K_guess (float, optional): Initial guess for carrying capacity. Defaults to 100.
        return_last_n (int): Number of last predictions to return
    Returns:
        pd.Series: Predicted values for future indices
    """
    series = preprocess(series.astype("float"))

    K_guess_factor = config["modules"]["age_gender"]["settings"]["K_guess_factor"]
    K_max = config["modules"]["age_gender"]["settings"]["K_max"]
    r_guess = config["modules"]["age_gender"]["settings"]["r_guess"]
    r_max = config["modules"]["age_gender"]["settings"]["r_max"]
    start_year = config["modules"]["age_gender"]["settings"]["start_year"]
    return_last_n = config["modules"]["age_gender"]["settings"]["n_predictions"]

    t = np.arange(start_year, last_year_of_observations + 1)
    y = series.values.astype(float)
    if len(t) != len(y):
        series = series.dropna()
        t = np.arange(start_year, series.last_valid_index() + 1)
        y = series.values.astype(float)

    K_guess = max(y) * K_guess_factor * K_max  # Slightly beyond current max

    try:
        # Initial guesses for r (growth rate) and t0 (midpoint)
        t0_guess = np.mean(np.arange(start_year, last_year_of_observations + 1))
        logger.debug(f"Fitted logistic params guess: K={K_guess:.2f}, r={r_guess:.4f}, t0={t0_guess:.2f}")
        popt, _ = curve_fit(
            logistic_function,
            t,
            y,
            p0=[K_guess, r_guess, t0_guess],
            bounds=([0, 0, 0], [K_max, r_max, np.inf]),
        )
        K, r, t0 = popt
        logger.debug(f"Fitted logistic params: K={K:.2f}, r={r:.4f}, t0={t0:.2f}")
    except Exception as e:
        logger.warning(f"Logistic curve fit failed: {e}. Returning last value.")
        return pd.Series([y[-1]] * return_last_n, index=pred_idx[-return_last_n:])

    # Predict future values
    t_future = list(pred_idx)
    y_future = logistic_function(t_future, K, r, t0)

    return pd.Series(y_future[-return_last_n:], index=pred_idx[-return_last_n:])


def fit_predict_logistic_older_age_groups(series, pred_idx, last_year_of_observations, config=None):
    """
    Fit logistic growth curve to a series and predict future values.
    Args:
        series (pd.Series): Historical time series
        pred_idx (list): Future indices for predictions
        K_guess (float, optional): Initial guess for carrying capacity. Defaults to 100.
        return_last_n (int): Number of last predictions to return
    Returns:
        pd.Series: Predicted values for future indices
    """
    K_max = series["K"]
    series = series.drop("K")
    series = preprocess(series.astype("float"))

    K_guess_factor = config["modules"]["age_gender"]["settings"]["K_guess_factor"]
    r_guess = config["modules"]["age_gender"]["settings"]["r_guess"]
    r_max = config["modules"]["age_gender"]["settings"]["r_max"]
    start_year = config["modules"]["age_gender"]["settings"]["start_year"]
    return_last_n = config["modules"]["age_gender"]["settings"]["n_predictions"]

    t = np.arange(start_year, last_year_of_observations + 1)
    y = series.values.astype(float)
    if len(t) != len(y):
        series = series.dropna()
        t = np.arange(start_year, series.last_valid_index() + 1)
        y = series.values.astype(float)

    K_guess = max(y) * K_guess_factor * K_max  # Slightly beyond current max

    try:
        # Initial guesses for r (growth rate) and t0 (midpoint)
        t0_guess = np.mean(np.arange(start_year, last_year_of_observations + 1))
        logger.debug(f"Fitted logistic params guess: K={K_guess:.2f}, r={r_guess:.4f}, t0={t0_guess:.2f}")
        popt, _ = curve_fit(
            logistic_function,
            t,
            y,
            p0=[K_guess, r_guess, t0_guess],
            bounds=([0, 0, 0], [K_max, r_max, np.inf]),
        )
        K, r, t0 = popt
        logger.debug(f"Fitted logistic params: K={K:.2f}, r={r:.4f}, t0={t0:.2f}")
    except Exception as e:
        logger.warning(f"Logistic curve fit failed: {e}. Returning last value.")
        return pd.Series([y[-1]] * return_last_n, index=pred_idx[-return_last_n:])

    # Predict future values
    t_future = list(pred_idx)
    y_future = logistic_function(t_future, K, r, t0)

    return pd.Series(y_future[-return_last_n:], index=pred_idx[-return_last_n:])


def age_groups_tweak(df, year_of_predictions, config):
    """
    Tweak the predictions for the age groups.
    When the combined prediction for age group 55-64 or 65-74 is more than the combined prediction of younger age groups, we need to tweak the predictions.
    We set the combined prediction for the older age groups to be that of logistic, as we have already ensured with fit_predict_logistic_older_age_groups()
    that the predictions for the older age groups are not more than the prediction of younger age groups.
    Args:
        df (pd.DataFrame): DataFrame containing predictions
        year_of_predictions (int): Year of predictions
    Returns:
        pd.DataFrame: DataFrame with tweaked predictions
    """

    df_tweaked = df.copy()
    previous_year_offset = config["prediction"]["preprocessing"]["previous_year_offset"]
    for year in [year_of_predictions - previous_year_offset, year_of_predictions]:
        younger_ages = config["modules"]["age_gender"]["settings"]["younger_age_groups"]
        older_ages = config["modules"]["age_gender"]["settings"]["older_age_groups"]
        younger_mask = df_tweaked["age"].isin(younger_ages)
        min_younger_predictions = (
            df_tweaked[younger_mask]
            .groupby(["country", "gender"])[f"{year}_combined"]
            .min()
            .reset_index()
            .rename(columns={f"{year}_combined": "min_younger_pred"})
        )
        df_tweaked = df_tweaked.merge(min_younger_predictions, on=["country", "gender"], how="left")

        tweak_counts = {}
        for age_group in older_ages:
            age_mask = df_tweaked["age"] == age_group
            needs_tweak = age_mask & (df_tweaked[f"{year}_combined"] > df_tweaked["min_younger_pred"])
            df_tweaked.loc[needs_tweak, f"{year}_combined"] = df_tweaked.loc[needs_tweak, f"{year}_logistic"]
            tweak_counts[age_group] = needs_tweak.sum()

        df_tweaked = df_tweaked.drop("min_younger_pred", axis=1)
        total_processed = len(df_tweaked[df_tweaked["age"].isin(older_ages)].groupby(["country", "gender"]))
        n_55_64_tweaked = tweak_counts["55_64"]
        n_65_74_tweaked = tweak_counts["65_74"]
        logger.info(f"for year {year}: age_groups_tweak summary:")
        logger.info(f"  Total groups processed: {total_processed}")
        logger.info(f"  Rows tweaked for 55-64: {n_55_64_tweaked}")
        logger.info(f"  Rows tweaked for 65-74: {n_65_74_tweaked}")
        logger.info(f"  Total rows tweaked: {n_55_64_tweaked + n_65_74_tweaked}")

    return df_tweaked


def adjust_100prc_predictions_cases(df_preds, df_train_data, year_of_predictions, config=None):
    """
    Implements two rules:
    (a) Adjust predictions to 100% if the last three years are 100% and the prediction is 100% (rw, theta) & 99.5% (logistic) to be 100%
    (b) Adjust predictions to be less than 100% if the last three years are not all 100% but the predictions are 100% (rw, theta) & 99.5% (logistic)
    Args:
        df (pd.DataFrame): DataFrame containing predictions
    Returns:
        pd.DataFrame: DataFrame with adjusted predictions
    """
    K_max = config["modules"]["age_gender"]["settings"]["K_max"]
    df_preds["composite_index"] = df_preds["country_age"] + " " + df_preds["gender"]
    df_preds.set_index("composite_index", inplace=True)

    # Initialize counters for logging
    rule_a_count = 0  # Set to 100%
    rule_b_count = 0  # Multiplied by random factor

    for year in [year_of_predictions - 1, year_of_predictions]:
        total_processed = 0
        for index in df_train_data.index:
            total_processed += 1
            last_valid_index = df_train_data.loc[index].last_valid_index()
            intermediate_valid_index = df_train_data.loc[index].drop(last_valid_index).last_valid_index()
            first_valid_index = df_train_data.loc[index].drop([last_valid_index, intermediate_valid_index]).last_valid_index()
            # Rule (a): Set to 100% if last three years are 100% and predictions are at max
            if (
                (df_train_data.loc[index][first_valid_index] == 1)
                and (df_train_data.loc[index][intermediate_valid_index] == 1)
                and (df_train_data.loc[index][last_valid_index] == 1)
                and (df_preds.loc[index][f"{year}_logistic"].round(3) >= K_max)
                and (df_preds.loc[index][f"{year}_rw"] == 1)
                and (df_preds.loc[index][f"{year}_theta"] == 1)
            ):
                df_preds.loc[index, f"{year}_combined"] = 1
                rule_a_count += 1
            # Rule (b): Multiply by random factor if last three years are not all 100% but predictions are at max
            if (
                (
                    df_train_data.loc[index][first_valid_index]
                    + df_train_data.loc[index][intermediate_valid_index]
                    + df_train_data.loc[index][last_valid_index]
                    != 3
                )
                and (df_preds.loc[index][f"{year}_logistic"].round(3) >= K_max)
                and (df_preds.loc[index][f"{year}_rw"] == 1)
                and (df_preds.loc[index][f"{year}_theta"] == 1)
            ):
                random_seed = 42  # for reproducibility
                random.seed(random_seed)
                df_preds.loc[index, f"{year}_combined"] = df_preds.loc[index, f"{year}_logistic"] * random.uniform(1, 1 / K_max)
                rule_b_count += 1
        # Log summary statistics
        logger.info(f"for year {year}: adjust_100prc_predictions_cases summary:")
        logger.info(f"  Total rows processed: {total_processed}")
        logger.info(f"  Rule (a) applied (set to 100%): {rule_a_count}")
        logger.info(f"  Rule (b) applied (set to <100%): {rule_b_count}")
        logger.info(f"  Total rows changed: {rule_a_count + rule_b_count}")

    df_preds.reset_index(inplace=True)
    df_preds.drop(columns=["composite_index"], inplace=True)

    return df_preds


def train_lasso(X_train, y_train, config=None):
    """
    Train LASSO regression model with cross-validation.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        config (dict): Configuration dictionary

    Returns:
        LassoCV: Fitted LASSO model

    Notes:
        - Scales input data by dividing by 100
        - Adapts cross-validation folds to data size
        - Handles convergence warnings
        - Uses maximum 10000 iterations
    """

    lasso_config = config["prediction"]["lasso"]
    scaling_factor = config["prediction"]["preprocessing"]["scaling_factor"]

    X_train_scaled = X_train.ffill().bfill() / scaling_factor
    y_train_scaled = y_train.ffill().bfill() / scaling_factor

    if X_train_scaled.shape[0] < lasso_config["min_samples_for_cv"]:
        cv = X_train_scaled.shape[0] - 1 if X_train_scaled.shape[0] - 1 > 1 else 1
    else:
        cv = lasso_config["default_cv_folds"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso = LassoCV(
            cv=cv,
            random_state=lasso_config["random_state"],
            max_iter=lasso_config["max_iterations"],
        ).fit(X_train_scaled, y_train_scaled)

    return lasso


def backtest_model(lasso, X_backtest, y_backtest, config=None):
    """
    Perform backtesting on trained LASSO model.

    Args:
        lasso (LassoCV): Trained LASSO model
        X_backtest (pd.DataFrame): Backtesting features
        y_backtest (pd.Series): True values for backtesting

    Returns:
        tuple: (mse, mae, mape) containing:
            - mse (float): Mean squared error
            - mae (float): Mean absolute error
            - mape (float): Mean absolute percentage error

    Notes:
        - Scales input data by dividing by 100
        - Clips predictions to [0, 100] range
        - Returns multiple error metrics
    """
    scaling_factor = config["prediction"]["preprocessing"]["scaling_factor"]
    clip_predictions = config["prediction"]["preprocessing"]["clip_predictions"]

    X_backtest_scaled = X_backtest / scaling_factor
    y_pred_backtest_scaled = lasso.predict(X_backtest_scaled)

    if clip_predictions:
        y_pred_backtest = np.clip(y_pred_backtest_scaled * scaling_factor, 0, 1)
    else:
        y_pred_backtest = y_pred_backtest_scaled * scaling_factor

    mse_backtest = mean_squared_error(y_backtest, y_pred_backtest)
    mae_backtest = mean_absolute_error(y_backtest, y_pred_backtest)
    mape_backtest = np.mean(np.abs((y_backtest - y_pred_backtest) / y_backtest)) * 100

    return mse_backtest, mae_backtest, mape_backtest


def prediction_gender_data(series, config=None):
    """
    Predict gender-specific data using random walk approach.

    Args:
        series (pd.Series): Time series data for gender predictions

    Returns:
        float: Predicted value using random walk method

    Notes:
        - Uses random walk forecasting for gender data
        - Applies preprocessing to handle missing values
        - Returns single predicted value
    """
    return random_walk_forecast(series, config=config)


def random_walk_forecast(series_individuals, config=None):
    """
    Perform random walk forecasting on individual-level data.

    Args:
        series_individuals (pd.Series): Time series data for individuals
        config (dict): Configuration dictionary

    Returns:
        float: Predicted value using random walk method

    Notes:
        - Uses smoothed random walk approach
        - Calculates median drift from last three years
        - Applies preprocessing for missing values
    """
    series = preprocess(series_individuals.astype("float"))
    lookback_period = config["prediction"]["random_walk"]["lookback_period"]
    smoothed_series_drift = series.diff()[-lookback_period:].median()
    random_walk_pred_individuals_smoothed = series.values[-1] + smoothed_series_drift
    return random_walk_pred_individuals_smoothed


###############OUTPUTS VALUES VALIDATION###############
def validate_values(df, module_name, file_path, pred_year):
    """Validate that all predictions are within [0,1] range"""

    pred_cols = [col for col in df.columns if (str(col).startswith(str(pred_year)) or str(col).startswith(str(pred_year - 1)))]

    if not pred_cols:
        logger.info(f"No prediction columns found for values check in {module_name}")
        return

    df_numeric = df[pred_cols].apply(pd.to_numeric, errors="coerce")

    if df_numeric.empty:
        logger.warning(f"No valid numeric prediction columns found for values check in {module_name}")
        return

    # check for values > 1
    invalid_rows = df_numeric.gt(1).any(axis=1)
    invalid_cnt = invalid_rows.sum()

    if invalid_cnt > 0:
        logger.warning(f"VALUES CHECK WARNING in {module_name}: {invalid_cnt} rows contain values > 1 in {file_path}")

        for col in df_numeric.columns:
            # get the actual invalid values
            invalid_mask = df_numeric[col] > 1
            invalid_values = df_numeric[col][invalid_mask]

            if not invalid_values.empty:
                logger.warning(f"Column {col}: {len(invalid_values)} values > 1 (max: {invalid_values.max():.6f})")

                for idx, value in invalid_values.items():
                    if any(col.lower() == "individual type" for col in df.columns):
                        # age_gender module - get individual type
                        country_rows = df[df.index == idx]
                        matching_row = country_rows[country_rows[col] == value]
                        if not matching_row.empty:
                            individual_type_col = next(col for col in df.columns if "individual type" in col.lower())
                            individual_type = matching_row[individual_type_col].iloc[0]
                            logger.warning(f"{idx} ({individual_type}): {value:.6f}")
                        else:
                            logger.warning(f"{idx}: {value:.6f}")

                    elif any("country" in col.lower() for col in df.columns):
                        # education module - get country
                        country_rows = df[df.index == idx]
                        matching_row = country_rows[country_rows[col] == value]
                        if not matching_row.empty:
                            country_col = next(col for col in df.columns if "country" in col.lower())
                            country = matching_row[country_col].iloc[0]
                            logger.warning(f"Row {idx} ({country}): {value:.6f}")
                        else:
                            logger.warning(f"Row {idx}: {value:.6f}")
                    else:
                        logger.warning(f"Row {idx}: {value:.6f}")

    elif invalid_cnt == 0:
        logger.info(f"{module_name}: All predictions values within valid range [0,1]")
