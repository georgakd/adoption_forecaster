"""
File I/O Operations for Quotas Forecasting Data

This module provides a unified FileWriter class for handling file operations across
multiple forecasting modules (Education and Age_Gender). It supports both local file
system and Google Cloud Storage operations for saving predictions and observations.

Supported Modules:
- Education: Secondary and tertiary education level internet usage predictions
- Age_Gender: Demographic-specific internet penetration predictions

Key Features:
- Dual storage modes: local and Google Cloud Storage
- Consistent file naming and directory structure
- Proper encoding handling and namespace organization
- Support for filtering countries and handling missing data
"""

import os
from google.cloud import storage
from google.api_core import exceptions as gapi_exceptions
from quotas_forecasting.config.logging_config import get_logger
from quotas_forecasting.services.gcp_clients import get_credentials
from quotas_forecasting.forecasting.utils import validate_values

logger = get_logger(__name__)


class FileWriter:
    """
    Handles file operations for saving forecasting predictions and observations.

    Supports both local filesystem and Google Cloud Storage, with automatic
    file naming, column renaming, and optional Pub/Sub notifications for
    downstream services. Works with Education and Age_Gender modules.

    Args:
        namespace (str): Module namespace ("Education" or "Age_Gender")
        mode (str): Storage mode ("local" or "prod")
        config (dict): Configuration dictionary
        pred_year (int): Prediction year for file naming
    """

    def __init__(self, namespace, mode="local", config=None, pred_year=None):
        """
            Initialize FileWriter for saving predictions and observations.

        Args:
            namespace (str): Module namespace ("Education" or "Age_Gender")
            mode (str): Storage mode - "local" for filesystem, "prod" for GCS. Defaults to "local"
            config (dict): Configuration dictionary (required)
            pred_year (int, optional): Prediction year for file naming

        Raises:
            ValueError: If config is None
        """
        self.config = config
        if self.config is None:
            raise ValueError("config is required for FileWriter")
        self.namespace = namespace
        self.mode = mode
        self.pred_year = pred_year
        self.bucket_name = self.config["defaults"]["bucket_name"]
        self.bucket_name_external = self.config["defaults"]["bucket_name_external"]
        self.dir_external = self.config["defaults"]["dir_external"]
        self.project_id = self.config["defaults"]["project_id"]
        self.project_id_external = self.config["pubsub"]["sub_project_id"]
        self.data_source = self.config["gcs"]["data_source"]
        self.service_acc = self.config["pubsub"]["service_account"]
        self.gcs_base_path = f"{self.data_source}/{self.namespace}"
        self.topic_name = self.config["pubsub"]["topic_name"]
        self.methodology = self.config["defaults"]["methodology"]

        if self.mode == "prod":
            from quotas_forecasting.services.message_serializer import MessageSerializer

            try:
                temp_serializer = MessageSerializer(config=self.config)
                self.credentials = get_credentials(self.config)

                self.client = storage.Client(project=self.project_id, credentials=self.credentials)
                self.bucket_int = self.client.bucket(self.bucket_name)

                if self.bucket_name_external and self.project_id_external:
                    self.client_ext = storage.Client(project=self.project_id_external, credentials=self.credentials)
                    self.bucket_ext = self.client_ext.bucket(self.bucket_name_external)
                else:
                    self.client_ext = None
                    self.bucket_ext = None

                # create message serializer object
                if self.topic_name:
                    self.message_serializer = temp_serializer
                else:
                    self.message_serializer = None
            except Exception as e:
                logger.error(f"Failed to initialize FileWriter in prod mode: Error: {e}")
                raise

        else:
            self.bucket_name_external = None
            self.message_serializer = None

    def _generate_filename(self, base_filename, destination="internal", year_of_prediction=None):
        """
        Generate appropriate filename based on destination.

        Args:
            base_filename: Base filename without extension
            destination: "internal" or "external"
            year_of_prediction: Year for predictions (only needed for external)

        Returns:
            str: Generated filename with appropriate format
        """
        if destination == "external":
            return f"{year_of_prediction}_{base_filename}_{self.methodology}.csv"
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d")
            return f"{year_of_prediction}_{base_filename}_{self.methodology}_{timestamp}.csv"

    def _get_gcs_path(self, output_dir, filename):
        """Convert local path to GCS path"""
        clean_output_dir = output_dir.lstrip("./")
        return f"{self.gcs_base_path}/{clean_output_dir}/{filename}"

    def save_to_external_bucket(self, df, output_dir, filename, encoding="utf-8-sig"):
        """Save file to external bucket of Universe service"""
        if not self.bucket_ext:
            logger.warning("External bucket not configured, skipping")
            return False

        try:
            csv_content = df.to_csv(encoding=encoding)

            blob_path = f"{self.dir_external}/{filename}"
            blob = self.bucket_ext.blob(blob_path)

            blob.upload_from_string(csv_content, content_type=f"text/csv; charset={encoding}")
            logger.info(f"Saved {filename} to external bucket: gs://{self.bucket_name_external}/{blob_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save {filename} to external bucket: {str(e)}")
            return False

    def _get_local_path(self, output_dir, filename):
        """Convert output_dir to local module-specific path"""
        clean_output_dir = output_dir.lstrip("./")
        local_data_dir = self.config["defaults"].get("local_data_dir")
        if local_data_dir:
            return os.path.join(local_data_dir, self.namespace, clean_output_dir, filename)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(script_dir, ".."))

        return os.path.join(base_path, self.namespace, clean_output_dir, filename)

    def _rename_columns_for_output(self, df, filename, year_of_prediction, last_year_of_observations):
        """
        Rename columns before writing to csv.

        -Education: years<=last_year -> "eurostat-{year}", years>last_year -> "prediction-{year}
        -Age-Gender: years<=last_year -> "{year}_{edu}" → "eurostat_{edu}_{year}" (if historical) or "prediction_{year}_{edu}" (if prediction)

        Args:
            df: Dataframe to rename
            filename: Output filename
            year_of_prediction: Prediction year (e.g. 2026)
            last_year_of_observations: Last year of observed data (e.g. 2024)

        Returns:
            Dataframe with renamed columns
        """
        df_torename = df.copy()

        column_mappping = {}

        is_edu = "Education" in filename
        is_ag = "Age_Gender" in filename

        for col in df_torename.columns:
            new_col = col

            if is_ag:
                if isinstance(col, (int, str)) and str(col).isdigit():
                    year = int(col)
                    if year <= last_year_of_observations:
                        new_col = f"eurostat-{year}"
                    else:
                        new_col = f"prediction-{year}"
            elif is_edu:
                if isinstance(col, str) and "_" in col:
                    try:
                        parts = col.split("_", 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            year = int(parts[0])
                            edu_level = parts[1]

                            if year <= last_year_of_observations:
                                new_col = f"eurostat-{edu_level}-{year}"
                            else:
                                new_col = f"prediction-{edu_level}-{year}"
                    except (ValueError, IndexError):  # if fails keep original name
                        pass

            if new_col != col:
                column_mappping[col] = new_col
        if column_mappping:
            df_torename.rename(columns=column_mappping, inplace=True)
        return df_torename

    def save_last_observations(self, df, output_dir, filename):
        """Save the last observations from the dataset to a local file or GCS."""
        if self.mode == "local":
            return self._save_local_last_observations(df, output_dir, filename)
        else:
            return self._save_gcp_last_observations(df, output_dir, filename)

    def save_predictions(
        self,
        predictions,
        year_of_prediction,
        output_dir,
        filename,
        countries_to_drop=None,
        last_year_of_observations=None,
    ):
        """Save predictions to local file or GCS"""
        if "past_data" not in filename:
            validate_values(predictions, self.namespace, filename, self.pred_year)

        if last_year_of_observations is None:
            year_diff = self.config["validation"]["year_difference"]
            last_year_of_observations = year_of_prediction - year_diff

        predictions = self._rename_columns_for_output(predictions, filename, year_of_prediction, last_year_of_observations)
        predictions = predictions.replace(":", "")  # in processing we replace missing values with :

        if self.mode == "local":
            return self._save_local_predictions(predictions, year_of_prediction, output_dir, filename, countries_to_drop)
        else:
            return self._save_gcp_predictions(predictions, year_of_prediction, output_dir, filename, countries_to_drop)

    def _save_local_last_observations(self, df, output_dir, filename):
        """Original local save logic"""
        local_path = self._get_local_path(output_dir, filename)
        local_dir = os.path.dirname(local_path)
        try:
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            df.iloc[:, -1].to_csv(local_path)
            logger.info(f"Saved last observations locally: path={local_path}")
        except OSError as e:
            logger.error(f"Filesystem error saving last observations: path={local_path}, error={e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving last observations: path={local_path}, error={e}")
            raise

    def _save_gcp_last_observations(self, df, output_dir, filename):
        """GCP save logic for last observations"""
        blob_path = "<unset>"
        try:
            csv_content = df.iloc[:, -1].to_csv()

            internal_filename = self._generate_filename(filename, "internal")
            blob_path = self._get_gcs_path(output_dir, internal_filename)
            blob = self.bucket_int.blob(blob_path)

            blob.upload_from_string(csv_content, content_type="text/csv")
            logger.info(f"Saved last observations to GCS: bucket={self.bucket_name}, blob_path={blob_path}, filename={internal_filename}")
        except gapi_exceptions.GoogleAPIError as e:
            logger.error(f"GCS API error saving last observations: bucket={self.bucket_name}, blob_path={blob_path}, error={e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving last observations to GCS: bucket={self.bucket_name}, blob_path={blob_path}, error={e}")
            raise

    def _save_local_predictions(
        self,
        predictions,
        year_of_prediction,
        output_dir,
        filename,
        countries_to_drop=None,
    ):

        local_path = self._get_local_path(output_dir, f"{year_of_prediction}_{filename}.csv")
        local_dir = os.path.dirname(local_path)

        try:
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            if countries_to_drop is None:
                countries_to_drop = []

            predictions[~(predictions.index.isin(countries_to_drop))].to_csv(local_path, encoding="utf-8-sig")
            logger.info(f"Saved predictions locally: path={local_path}, year={year_of_prediction}, filename={filename}")
        except OSError as e:
            logger.error(f"Filesystem error saving predictions: path={local_path}, error={e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving predictions locally: path={local_path}, error={e}")
            raise

    def _save_gcp_predictions(
        self,
        predictions,
        year_of_prediction,
        output_dir,
        filename,
        countries_to_drop=None,
    ):
        """GCP save logic for predictions"""
        if countries_to_drop is None:
            countries_to_drop = []

        blob_path = "<unset>"
        try:
            filtered_predictions = predictions[~(predictions.index.isin(countries_to_drop))]
            csv_content = filtered_predictions.to_csv(encoding="utf-8-sig")

            internal_filename = self._generate_filename(filename, "internal", year_of_prediction)
            blob_path = self._get_gcs_path(output_dir, internal_filename)
            blob = self.bucket_int.blob(blob_path)

            blob.upload_from_string(csv_content, content_type="text/csv; charset=utf-8")
            logger.info(
                f"Saved predictions to GCS: bucket={self.bucket_name}, blob_path={blob_path},\
                filename={internal_filename}, year={year_of_prediction}"
            )
        except gapi_exceptions.GoogleAPIError as e:
            logger.error(f"GCS API error saving predictions: bucket={self.bucket_name}, blob_path={blob_path}, error={e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving predictions to GCS: bucket={self.bucket_name}, blob_path={blob_path}, error={e}")
            raise

        # Send files to Andromeda's bucket - only projections, no past data
        if "past_data" in filename and self.config["pubsub"]["enable_notifications"]:
            external_filename = self._generate_filename(filename, "external", year_of_prediction)
            self.save_to_external_bucket(predictions, output_dir, external_filename)

            # Send protobuf notification after successful save
            file_type = self._get_file_type_from_filename(filename)
            external_path = f"gs://{self.bucket_name_external}/{self.dir_external}/{external_filename}"
            self._send_file_notification(file_type, external_path, year_of_prediction)

    def _get_file_type_from_filename(self, filename):
        """Determine file type from filename"""
        file_type_mapping = {
            "Age_Gender": "AGE_AND_GENDER",
            "16-64": "EDUCATION_16_TO_64",
            "16-74": "EDUCATION_16_TO_74",
        }

        for pattern, file_type in file_type_mapping.items():
            if pattern in filename:
                return file_type

        return "FILE_TYPE_INVALID"

    def _send_file_notification(self, file_type, gcs_path, year):
        """Send protobuf notification when file is saved to GCS"""
        should_notify = (
            self.message_serializer is not None and self.data_source.strip().lower() == "eurostat" and self.config["pubsub"]["enable_notifications"]
        )
        if not should_notify:
            return
        try:
            filename = os.path.basename(gcs_path)
            external_path = f"gs://{self.bucket_name_external}/{self.dir_external}/{filename}"

            success = self.message_serializer.send_modelling_output_notification(file_type=file_type, gcs_path=external_path, year=year)
            if success:
                logger.info(f"Sent protobuf notification for {file_type} file: {gcs_path}")
            else:
                logger.warning(f"Failed to send protobuf notification for {file_type} file")
        except Exception as e:
            logger.error(f"Error sending protobuf notification: {e}")
