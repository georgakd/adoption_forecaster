"""
This module provides a utility function to load CSV files either from Google Cloud Storage or from a local folder
into pandas DataFrames. It supports both Education and Age_Gender namespaces.
"""

import pandas as pd
from google.auth import default
from google.cloud import storage
import io
import os
import shutil
import re

from adoption_forecaster.config.logging_config import get_logger
from adoption_forecaster.services.gcp_clients import emulator_enabled, get_credentials

logger = get_logger(__name__)


class Loader:
    def __init__(self, namespace, mode="local", config=None):
        self.namespace = namespace
        self.mode = mode
        self.bucket_name = config["defaults"]["bucket_name"]
        self.project_id = config["defaults"]["project_id"]
        self.config = config

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.abspath(os.path.join(script_dir, ".."))
        self.data_source = config["gcs"]["data_source"]

        if self.mode == "local":
            self._setup_local_input_folder()
        elif self.mode == "prod":
            try:
                if emulator_enabled():
                    self.client = storage.Client(credentials=get_credentials(config), project=self.project_id)
                else:
                    credentials, project = default()
                    self.client = storage.Client(credentials=credentials, project=self.project_id)
                    self.client = storage.Client(project=self.project_id)
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"Initialized GCS client for project={self.project_id}, bucket={self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to initialize GCS client in Loader: project_id={self.project_id}, bucket_name={self.bucket_name}, error={e}")
                raise

    def _setup_local_input_folder(self):
        """Create input folder structure for local development"""
        local_data_dir = self.config["defaults"].get("local_data_dir")
        if local_data_dir:
            # Override: read in place from <local_data_dir>/<Namespace>/input, nothing is moved out of the package dir
            self.local_path = os.path.join(local_data_dir, self.namespace, "input")
            os.makedirs(self.local_path, exist_ok=True)
            logger.info(f"Local path set to: {self.local_path}")
            return

        # Copy files from main directory to {module}/input, where module : Age_Gender or Education
        source_input = self.base_path  # Main directory where main.py is
        dest_input = os.path.join(self.base_path, self.namespace, "input")

        os.makedirs(dest_input, exist_ok=True)

        # Get file names from config
        module_config = self.config["modules"][self.namespace.lower()]
        expected_files = list(module_config["files"].values())

        base_names = [pattern.replace(".csv", "") for pattern in expected_files]  # remove .csv part to search with prefix in bucket

        if os.path.exists(source_input):
            for file in os.listdir(source_input):
                file_matches = False
                for base_name in base_names:
                    if file.startswith(base_name) and file.endswith(".csv"):
                        file_matches = True
                        break
                if file_matches:

                    source_file = os.path.join(source_input, file)
                    dest_file = os.path.join(dest_input, file)

                    if os.path.isfile(source_file):
                        if not os.path.exists(dest_file) or os.path.getmtime(source_file) > os.path.getmtime(dest_file):
                            try:
                                shutil.copy2(source_file, dest_file)
                                logger.info(f"Copied {file} to {dest_input}")
                                logger.info(f"Successfully copied {file}")

                                os.remove(source_file)
                                logger.info(f"Deleted {file} from source dir")

                            except Exception as e:
                                logger.error(f"Error copying {file}: {e}")
                        else:
                            logger.info(f"File {file} already exists and is up to date")
                    else:
                        logger.warning(f"Skipping directory: {file}")

        self.local_path = dest_input
        logger.info(f"Local path set to: {self.local_path}")
        logger.info(f"Expected files for {self.namespace}: {expected_files}")

    def read_csv(self, filename):
        """Read CSV file from local or GCS based on mode"""
        if self.mode == "local":
            return self._read_local(filename)
        else:
            return self._read_gcs(filename)

    def _read_local(self, filename):
        """Read CSV file from local filesystem"""
        file_path = os.path.join(self.local_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {filename} not found in {self.local_path}")

        logger.info(f"Reading {filename} from local filesystem")
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading CSV from local path={file_path}: {e}")
            raise

    def _read_gcs(self, filename):
        """Read CSV file from Google Cloud Storage"""
        try:
            if self.namespace in ["Education", "Age_Gender"]:
                file_path = f"{self.data_source}/{self.namespace}/input/{filename}"
            else:
                raise ValueError("Namespace must be either 'Education' or 'Age_Gender'")

            blob = self.bucket.blob(file_path)

            if not blob.exists():
                raise FileNotFoundError(f"File {file_path} not found in bucket {self.bucket_name}")

            content = blob.download_as_text()
            df = pd.read_csv(io.StringIO(content))
            logger.info(f"Successfully loaded {file_path} from GCS")
            return df

        except Exception as e:
            logger.error(f"Error loading {filename} from GCS: {str(e)}")
            raise

    def read_csv_timestamped(self, base_filename):
        """Read CSV file with timestamp from local or GCS based on mode"""
        if self.mode == "local":
            return self._read_local_timestamped(base_filename)
        else:
            return self._read_gcs_timestamped(base_filename)

    def _read_local_timestamped(self, base_filename):
        """Read timestamped CSV file from local filesystem"""

        try:
            # Define timestamp pattern for local files (handles both colons and underscores in time)
            # Pattern matches: filename_YYYY-MM-DDTHH:MM:SS+ZZZZ.csv or filename_YYYY-MM-DDTHH_MM_SS+ZZZZ.csv
            timestamp_pattern = re.compile(rf"^{re.escape(base_filename)}_(\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}[_:]\d{{2}}[_:]\d{{2}}[+-]\d{{4}})\.csv$")
            matching_files = []
            for file in os.listdir(self.local_path):
                if timestamp_pattern.match(file):
                    matching_files.append(file)

            if not matching_files:
                raise FileNotFoundError(f"No timestamped file found for {base_filename} in {self.local_path}")

            # Sort by timestamp to get the most recent file
            def extract_timestamp(filename):
                match = timestamp_pattern.match(filename)
                # Convert underscores back to colons for proper timestamp sorting
                return match.group(1).replace("_", ":")

            matching_files.sort(key=extract_timestamp, reverse=True)
            latest_file = matching_files[0]

            file_path = os.path.join(self.local_path, latest_file)
            logger.info(f"Reading timestamped file {latest_file} from local filesystem")
            return pd.read_csv(file_path)

        except Exception as e:
            logger.error(f"Error loading timestamped file {base_filename} from local filesystem: {str(e)}")
            raise

    def _read_gcs_timestamped(self, base_filename):
        """Read timestamped CSV file from Google Cloud Storage"""
        try:
            if self.namespace in ["Education", "Age_Gender"]:
                prefix = f"{self.data_source}/{self.namespace}/input/"
                blobs = self.bucket.list_blobs(prefix=prefix)

                matching_files = []
                for blob in blobs:
                    filename = os.path.basename(blob.name)
                    # Skip files in old_datasets folder
                    if "old_datasets" in blob.name:
                        continue
                    if filename.startswith(base_filename) and filename.endswith(".csv"):
                        matching_files.append(blob)

                if not matching_files:
                    raise FileNotFoundError(f"No timestamped file found for {base_filename} in GCS")

                # Sort by timestamp to get the most recent file
                def extract_timestamp(blob):
                    import re

                    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})")
                    match = timestamp_pattern.search(blob.name)
                    return match.group(1) if match else "0000-00-00T00:00:00+0000"

                # Sort by timestamp (most recent first)
                matching_files.sort(key=extract_timestamp, reverse=True)
                latest_blob = matching_files[0]

                content = latest_blob.download_as_text()
                df = pd.read_csv(io.StringIO(content))
                logger.info(f"Successfully loaded {latest_blob.name} from GCS")
                return df
            else:
                raise ValueError("Namespace must be either 'Education' or 'Age_Gender'")

        except Exception as e:
            logger.error(f"Error loading timestamped file {base_filename} from GCS: {str(e)}")
            raise
