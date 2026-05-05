import requests
import pandas as pd
import re
import os
from datetime import datetime
from google.cloud import storage
from pyjstat import pyjstat
from google.api_core import exceptions as gapi_exceptions

from quotas_forecasting.config.logging_config import get_logger

logger = get_logger(__name__)

TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})")


class EurostatFetcher:
    def __init__(self, dataset_codes, countries, years, config=None):
        self.dataset_codes = dataset_codes
        self.countries = countries
        self.years = years
        self.config = config
        self.ages_15_74 = [f"{age} years" for age in range(15, 75)]

    def fetch_catalogue(self):
        url = self.config["eurostat"]["api"]["catalogue_url"]
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            datasets = []
            for line in lines:
                parts = line.split("\t")
                if len(parts) >= 3 and parts[2].strip('"') == "dataset":
                    code = parts[1].strip('"')
                    title = parts[3].strip('"') if len(parts) > 3 else None
                    datasets.append((code, title))
            return datasets
        except Exception as e:
            logger.error(f"Failed to fetch Eurostat catalogue: {e}")
            return []

    def check_dataset_exists(self, target, datasets):
        target = target.lower()
        match = [title for code, title in datasets if code.lower() == target]
        return bool(match)

    def fetch_dataset(self, code):
        url = self.config["eurostat"]["api"]["data_url_template"].format(code=code)
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            updated_datetime = data.get("updated", "")
            results = pyjstat.from_json_stat(resp.json())
            return pd.DataFrame(results[0]), updated_datetime
        except Exception as e:
            logger.error(f"Failed to fetch dataset {code}: {e}")
            return pd.DataFrame(), ""

    def filter_total_population_16_64(self, df, sex):
        ages_16_64 = [f"{age} years" for age in range(16, 65)]
        df_filtered = df[
            (df["Time"].isin(self.years))
            & (df["Geopolitical entity (reporting)"].isin(self.countries))
            & (df["Sex"] == sex)
            & (df["Age class"].isin(ages_16_64))
        ]
        return df_filtered[["Time", "Geopolitical entity (reporting)", "Sex", "Age class", "value"]]

    def filter_total_population_16_74(self, df, sex):
        df_filtered = df[
            (df["Time"].isin(self.years))
            & (df["Geopolitical entity (reporting)"].isin(self.countries))
            & (df["Sex"] == sex)
            & (df["Age class"].isin(self.ages_15_74))
        ]
        df_filtered = df_filtered.drop(["Time frequency", "Unit of measure", "Sex"], axis=1)
        return df_filtered

    def filter_by_education(self, df, unit, indicator, individual_type):
        return df[
            (df["Information society indicator"].isin(indicator))
            & (df["Time"].isin(self.years))
            & (df["Individual type"] == individual_type)
            & (df["Geopolitical entity (reporting)"].isin(self.countries))
            & (df["Unit of measure"] == unit)
        ]

    def filter_education_distribution(self, df, age_group, sex, edu_levels):
        return df[
            (df["Age class"].isin(age_group))
            & (df["Sex"] == sex)
            & (df["Geopolitical entity (reporting)"].isin(self.countries))
            & (df["International Standard Classification of Education (ISCED 2011)"].isin(edu_levels))
        ]

    def filter_by_age_group(self, df, individual_type, unit, indicator):
        return df[
            (df["Time"].isin(self.years))
            & (df["Geopolitical entity (reporting)"].isin(self.countries))
            & (df["Individual type"].isin(individual_type))
            & (df["Unit of measure"] == unit)
            & (df["Information society indicator"] == indicator)
        ]

    def parse_individual_type(self, s):
        gender_order_map = {"Males": 0, "Females": 1, "Total": 2}
        gender_order = 2
        for g in gender_order_map:
            if g in s:
                gender_order = gender_order_map[g]
                break
        match = re.search(r"(\d+)\s+to\s+(\d+)", s)
        age_start = int(match.group(1)) if match else 0
        return (gender_order, age_start)

    def sort_by_gender_age(self, df):
        sort_keys = [(country, *self.parse_individual_type(ind_type)) for country, ind_type in df.index]
        df = df.copy()
        df["__sort_key__"] = sort_keys
        df = df.sort_values("__sort_key__").drop(columns="__sort_key__")
        return df


class EurostatApp:
    def __init__(self, last_year_of_observations, config):
        self.config = config
        self.dataset_codes = config["eurostat"]["dataset_codes"]
        self.years_list = [
            str(y)
            for y in range(
                self.config["prediction"]["time_series"]["start_year"],
                last_year_of_observations + 1,
            )
        ]
        self.countries_list = self.config["eurostat"]["countries"]

        # File names from config
        self.education_files_cfg = self.config["modules"]["education"]["files"]
        self.education_files = list(self.education_files_cfg.values())
        self.gender_age_files_cfg = self.config["modules"]["age_gender"]["files"]
        self.gender_age_files = list(self.gender_age_files_cfg.values())

        self.last_year_of_observations = last_year_of_observations
        self.fetcher = EurostatFetcher(self.dataset_codes, self.countries_list, self.years_list, self.config)

        # GCS configuration
        self.bucket_name = self.config["defaults"]["bucket_name"]
        self.project_id = self.config["defaults"]["project_id"]
        data_source = self.config["gcs"]["data_source"]
        input_subfolder = self.config["gcs"]["input_subfolder"]
        self.gcs_base_path_education = f"{data_source}/Education/{input_subfolder}"
        self.gcs_base_path_agegender = f"{data_source}/Age_Gender/{input_subfolder}"

        self.dataframes = {}

        logger.info("Fetching ISOC_CI_IFP_IU data for reuse across methods...")
        self.isoc_data, self.isoc_timestamp = self.fetcher.fetch_dataset(self.dataset_codes[0])
        logger.info(f"ISOC_CI_IFP_IU data fetched with timestamp: {self.isoc_timestamp}")

        logger.info("Fetching EDAT_LFS_9903 data for reuse across methods...")
        self.edat_data, self.edat_timestamp = self.fetcher.fetch_dataset(self.dataset_codes[1])
        logger.info(f"EDAT_LFS_9903 data fetched with timestamp: {self.edat_timestamp}")

        logger.info("Fetching DEMO_PJAN data for reuse across methods...")
        self.population_data, self.population_timestamp = self.fetcher.fetch_dataset(self.dataset_codes[2])
        logger.info(f"Population data fetched with timestamp: {self.population_timestamp}")

    def run(self):
        self.process_total_population_by_age_range()
        self.process_internet_by_education()
        self.process_population_distribution()
        self.process_online_by_age_and_gender()

    def list_gcs_files(self, bucket_name=None, prefix=""):
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name or self.bucket_name)
        it = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in it]

    def extract_only_timestamped_filenames(self, files_list):
        """
        Extract timestamped filenames and map them to their base names.

        Args:
            files_list (list): List of full GCS paths

        Returns:
            dict: Mapping of base filename (without .csv) to timestamp
        """
        files_with_timestamp = {}

        for full_path in files_list:
            if "old_datasets/" in full_path:
                continue

            filename = os.path.basename(full_path)
            m = TIMESTAMP_RE.search(filename)
            if not m:
                continue

            ts = m.group(1)
            base = filename.replace(f"_{ts}.csv", "")

            if base not in files_with_timestamp or ts > files_with_timestamp[base]:
                files_with_timestamp[base] = ts

        return files_with_timestamp

    def get_latest_file_in_folder(self, gcs_base_path, base_filename=None):
        try:
            files_list = self.list_gcs_files(prefix=gcs_base_path)
            csv_files = [f for f in files_list if f.endswith(".csv") and "old_datasets" not in f]
            if not csv_files:
                return None

            if base_filename:
                ts_re = re.compile(rf"^{re.escape(base_filename)}_(\d{{4}}-\d{{2}}-\d{{2}}T\d{{2}}:\d{{2}}:\d{{2}}[+-]\d{{4}})\.csv$")
                matching_files = [f for f in csv_files if ts_re.match(os.path.basename(f))]
                if not matching_files:
                    return None
                csv_files = matching_files

            csv_files.sort(
                key=lambda x: (
                    TIMESTAMP_RE.search(os.path.basename(x)).group(1) if TIMESTAMP_RE.search(os.path.basename(x)) else "0000-00-00T00:00:00+0000"
                )
            )
            return os.path.basename(csv_files[-1])

        except gapi_exceptions.GoogleAPIError as e:
            logger.error(f"Failed to get latest file in {gcs_base_path}: {str(e)}")
            return None

    def _should_process_group(self, filenames, updated_datetime, files_list, gcs_base_path):
        """
        Determine if a group of files should be processed.

        Args:
            filenames (list): List of filenames (without .csv extension from config)
            updated_datetime (str): New timestamp from API
            files_list (dict): Mapping of base names to existing timestamps
            gcs_base_path (str): GCS base path

        Returns:
            bool: True if processing is needed, False otherwise
        """
        bases = filenames

        any_missing = any(self.get_latest_file_in_folder(gcs_base_path, base) is None for base in bases)

        any_stale = any(files_list.get(base, "NOT_FOUND") != updated_datetime for base in bases)

        return files_list == {} or any_missing or any_stale

    def upload_to_gcs_with_path(self, df, filename, gcs_base_path, encoding="utf-8-sig"):
        try:
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(self.bucket_name)
            gcs_path = f"{gcs_base_path}/{filename}"
            blob = bucket.blob(gcs_path)
            csv_content = df.to_csv(encoding=encoding)
            blob.upload_from_string(csv_content, content_type=f"text/csv; charset={encoding}")
            logger.info(f"Successfully uploaded {filename} to GCS: gs://{self.bucket_name}/{gcs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {filename} to GCS: {str(e)}")
            return False

    def move_file_to_old_datasets(self, filename, gcs_base_path):
        try:
            logger.info(f"move_file_to_old_datasets - Starting move for {filename}")
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(self.bucket_name)

            source_path = f"{gcs_base_path}/{filename}"
            source_blob = bucket.blob(source_path)

            if not source_blob.exists():
                logger.info(f"Source file {filename} doesn't exist, no need to move")
                return True

            dest_path = f"{gcs_base_path}/old_datasets/{filename}"
            bucket.copy_blob(source_blob, bucket, dest_path)
            source_blob.delete()
            logger.info(f"Moved {filename} to old_datasets folder: gs://{self.bucket_name}/{dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to move {filename} to old_datasets: {str(e)}")
            return False

    def process_total_population_by_age_range(self):
        filename = self.education_files_cfg["total_population_by_age_range"]
        base = filename
        df, updated_datetime = self.population_data, self.population_timestamp

        existing_file = self.get_latest_file_in_folder(self.gcs_base_path_education, base)
        if (
            getattr(self, "files_list_education", {}) == {}
            or updated_datetime != getattr(self, "files_list_education", {}).get(filename, "NOT_FOUND")
            or existing_file is None
        ):
            if existing_file:
                self.move_file_to_old_datasets(existing_file, self.gcs_base_path_education)

            results = []
            for age_label, filter_fn in [
                ("16-64", lambda d: self.fetcher.filter_total_population_16_64(d, "Total")),
                ("16-74", lambda d: self.fetcher.filter_total_population_16_74(d, "Total")),
            ]:
                df_filtered = filter_fn(df)
                grouped_df = df_filtered.groupby(["Geopolitical entity (reporting)", "Time"], as_index=False)["value"].sum()
                total_pops = grouped_df.pivot(
                    index="Geopolitical entity (reporting)",
                    columns="Time",
                    values="value",
                )
                total_pops.index.set_names(["country"], inplace=True)
                total_pops["age_range"] = age_label
                total_pops = total_pops.reset_index().set_index(["country", "age_range"])
                results.append(total_pops)

            combined = pd.concat(results).reset_index()
            combined = combined.set_index(["country", "age_range"])
            self.dataframes[filename] = (combined, updated_datetime)
        else:
            self.dataframes[filename] = (pd.DataFrame([]), updated_datetime)

    def process_internet_by_education(self):
        df, updated_datetime = self.isoc_data, self.isoc_timestamp

        ratios = self.education_files_cfg["internet_penetration_population"]
        ed_filenames = [ratios]

        if self._should_process_group(
            ed_filenames,
            updated_datetime,
            getattr(self, "files_list_education", {}),
            self.gcs_base_path_education,
        ):
            for base in ed_filenames:
                old = self.get_latest_file_in_folder(self.gcs_base_path_education, base)
                if old:
                    self.move_file_to_old_datasets(old, self.gcs_base_path_education)

            types = [
                "Individuals with no or low formal education",
                "Individuals with medium formal education",
                "Individuals with high formal education",
            ]
            indicator = ["Last internet use: in the last 12 months"]
            unit = "Percentage of individuals"

            results = []
            for level in types:
                filtered = self.fetcher.filter_by_education(df, unit, indicator, level)
                results.append(filtered)

            results_all = pd.concat(results).pivot(
                index=["Geopolitical entity (reporting)", "Individual type"],
                columns="Time",
                values="value",
            )
            results_all.index.set_names(["country", "Education level"], inplace=True)
            mapping = {
                "Individuals with no or low formal education": 1,
                "Individuals with medium formal education": 2,
                "Individuals with high formal education": 3,
            }
            results_all.index = results_all.index.set_levels(results_all.index.levels[1].map(mapping), level="Education level")
            self.dataframes[ratios] = (results_all, updated_datetime)
        else:
            for fname in ed_filenames:
                self.dataframes[fname] = (pd.DataFrame([]), updated_datetime)

    def process_population_distribution(self):
        filename = self.education_files_cfg["population_ratios"]
        base = filename
        df, updated_datetime = self.edat_data, self.edat_timestamp

        if getattr(self, "files_list_education", {}) == {} or updated_datetime != getattr(self, "files_list_education", {}).get(
            filename, "NOT_FOUND"
        ):
            old_filename = self.get_latest_file_in_folder(self.gcs_base_path_education, base)
            if old_filename:
                self.move_file_to_old_datasets(old_filename, self.gcs_base_path_education)

            age_group = ["From 15 to 64 years", "From 15 to 74 years"]
            sex = "Total"
            levels = [
                "Less than primary, primary and lower secondary education (levels 0-2)",
                "Upper secondary and post-secondary non-tertiary education (levels 3 and 4)",
                "Tertiary education (levels 5-8)",
            ]
            filtered = self.fetcher.filter_education_distribution(df, age_group, sex, levels)
            result = filtered.pivot(
                index=[
                    "Geopolitical entity (reporting)",
                    "Age class",
                    "International Standard Classification of Education (ISCED 2011)",
                ],
                columns="Time",
                values="value",
            )
            result.index.set_names(["country", "Age group", "Education level"], inplace=True)
            mapping = {
                "Less than primary, primary and lower secondary education (levels 0-2)": 1,
                "Upper secondary and post-secondary non-tertiary education (levels 3 and 4)": 2,
                "Tertiary education (levels 5-8)": 3,
            }
            result.index = result.index.set_levels(result.index.levels[2].map(mapping), level="Education level")
            self.dataframes[filename] = (result, updated_datetime)
        else:
            self.dataframes[filename] = (pd.DataFrame([]), updated_datetime)

    def process_online_by_age_and_gender(self):
        filename = self.gender_age_files_cfg["age_gender"]
        base = filename.rsplit(".", 1)[0]
        df, updated_datetime = self.isoc_data, self.isoc_timestamp

        existing_file = self.get_latest_file_in_folder(self.gcs_base_path_agegender, base)
        files_list = getattr(self, "files_list_agegender", {})
        if files_list == {} or updated_datetime != files_list.get(filename, "NOT_FOUND") or existing_file is None:
            if existing_file:
                self.move_file_to_old_datasets(existing_file, self.gcs_base_path_agegender)

            types = [
                "Males, 16 to 24 years old",
                "Females, 16 to 24 years old",
                "Males, 25 to 34 years old",
                "Females, 25 to 34 years old",
                "Males 35 to 44 years old",
                "Females 35 to 44 years old",
                "Males 45 to 54 years old",
                "Females 45 to 54 years old",
                "Males 55 to 64 years old",
                "Females 55 to 64 years old",
                "Males 65 to 74 years old",
                "Females 65 to 74 years old",
            ]
            unit = "Percentage of individuals"
            indicator = "Last internet use: in the last 12 months"
            filtered = self.fetcher.filter_by_age_group(df, types, unit, indicator)
            result = filtered.pivot(
                index=["Geopolitical entity (reporting)", "Individual type"],
                columns="Time",
                values="value",
            )
            result.index.set_names(["country", "Individual type"], inplace=True)
            result = self.fetcher.sort_by_gender_age(result)
            self.dataframes[filename] = (result, updated_datetime)
        else:
            self.dataframes[filename] = (pd.DataFrame([]), updated_datetime)

    def upload_all_to_gcs(self):

        logger.info("Starting upload to Google Cloud Storage...")

        upload_results = {}
        result_details = {}

        # Upload education files
        logger.info("Uploading Education-related files...")
        for filename in self.education_files:
            if filename in self.dataframes and not self.dataframes[filename][0].empty:
                try:
                    base = filename.rsplit(".", 1)[0]
                    actual_name = f"{base}_{self.dataframes[filename][1]}.csv"
                    success = self.upload_to_gcs_with_path(
                        self.dataframes[filename][0],
                        actual_name,
                        self.gcs_base_path_education,
                    )
                    upload_results[filename] = success
                    result_details[filename] = {
                        "status": "uploaded" if success else "failed",
                        "gcs": f"gs://{self.bucket_name}/{self.gcs_base_path_education}/{actual_name}",
                    }
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    upload_results[filename] = False
                    result_details[filename] = {"status": "failed", "gcs": None}
            elif filename in self.dataframes and self.dataframes[filename][0].empty:
                base = filename.rsplit(".", 1)[0]
                existing_file = self.get_latest_file_in_folder(self.gcs_base_path_education, base)
                if existing_file:
                    logger.info(f"DataFrame for {filename} already exists in GCS")
                    upload_results[filename] = True
                    result_details[filename] = {
                        "status": "already_exists",
                        "gcs": f"gs://{self.bucket_name}/{self.gcs_base_path_education}/{existing_file}",
                    }
                else:
                    logger.warning(f"DataFrame for {filename} is empty and file not found in GCS, skipping upload")
                    upload_results[filename] = False
                    result_details[filename] = {
                        "status": "skipped_no_data",
                        "gcs": None,
                    }
            else:
                logger.warning(f"DataFrame for {filename} not found in memory, skipping upload")
                upload_results[filename] = False
                result_details[filename] = {
                    "status": "skipped_not_in_memory",
                    "gcs": None,
                }

        # Upload age/gender files
        logger.info("Uploading Age/Gender-related files...")
        for filename in self.gender_age_files:
            if filename in self.dataframes and not self.dataframes[filename][0].empty:
                try:
                    base = filename.rsplit(".", 1)[0]
                    actual_name = f"{base}_{self.dataframes[filename][1]}.csv"
                    success = self.upload_to_gcs_with_path(
                        self.dataframes[filename][0],
                        actual_name,
                        self.gcs_base_path_agegender,
                    )
                    upload_results[filename] = success
                    result_details[filename] = {
                        "status": "uploaded" if success else "failed",
                        "gcs": f"gs://{self.bucket_name}/{self.gcs_base_path_agegender}/{actual_name}",
                    }
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    upload_results[filename] = False
                    result_details[filename] = {"status": "failed", "gcs": None}
            elif filename in self.dataframes and self.dataframes[filename][0].empty:
                base = filename.rsplit(".", 1)[0]
                existing_file = self.get_latest_file_in_folder(self.gcs_base_path_agegender, base)
                if existing_file:
                    logger.info(f"DataFrame for {filename} already exists in GCS")
                    upload_results[filename] = True
                    result_details[filename] = {
                        "status": "already_exists",
                        "gcs": f"gs://{self.bucket_name}/{self.gcs_base_path_agegender}/{existing_file}",
                    }
                else:
                    logger.warning(f"DataFrame for {filename} is empty and file not found in GCS, skipping upload")
                    upload_results[filename] = False
                    result_details[filename] = {
                        "status": "skipped_no_data",
                        "gcs": None,
                    }
            else:
                logger.warning(f"DataFrame for {filename} not found in memory, skipping upload")
                upload_results[filename] = False
                result_details[filename] = {
                    "status": "skipped_not_in_memory",
                    "gcs": None,
                }

        # Summary
        successful_uploads = sum(upload_results.values())
        total_files = len(self.education_files) + len(self.gender_age_files)

        logger.info(f"Upload Summary: {successful_uploads}/{total_files} files uploaded successfully")
        logger.info(f"Education files: {sum(upload_results.get(f, False) for f in self.education_files)}/{len(self.education_files)}")
        logger.info(f"Gender/Age files: {sum(upload_results.get(f, False) for f in self.gender_age_files)}/{len(self.gender_age_files)}")

        logger.info("Upload Results:")
        for fname, meta in result_details.items():
            if meta["gcs"]:
                logger.info(f"  {fname}: {meta['status']} -> {meta['gcs']}")
            else:
                logger.info(f"  {fname}: {meta['status']}")

        if successful_uploads != total_files:
            logger.warning("Some files failed to upload. Check the logs above for details.")
        return upload_results


class EurostatDataFetcher:
    def __init__(self, last_year_of_observations=None, config=None):
        if last_year_of_observations is None:
            last_year_of_observations = datetime.now().year - 1

        self.last_year_of_observations = last_year_of_observations
        self.config = config
        self.app = EurostatApp(last_year_of_observations=last_year_of_observations, config=self.config)

        data_source = self.config["gcs"]["data_source"]
        input_subfolder = self.config["gcs"]["input_subfolder"]
        self.gcs_base_path_education = f"{data_source}/Education/{input_subfolder}"
        self.gcs_base_path_agegender = f"{data_source}/Age_Gender/{input_subfolder}"

    def fetch_data(self):
        try:
            logger.info("Starting Eurostat data fetching and processing...")
            logger.info(f"Last year of historic data: {self.last_year_of_observations}")
            logger.info(f"Data range: {self.config['prediction']['time_series']['start_year']} to {self.last_year_of_observations}")

            files_list_education = self.app.list_gcs_files(prefix=self.gcs_base_path_education)
            files_list_agegender = self.app.list_gcs_files(prefix=self.gcs_base_path_agegender)
            self.app.files_list_education = self.app.extract_only_timestamped_filenames(files_list_education)
            self.app.files_list_agegender = self.app.extract_only_timestamped_filenames(files_list_agegender)

            # processing and upload
            self.app.run()
            logger.info("Uploading files to Google Cloud Storage...")
            upload_results = self.app.upload_all_to_gcs()

            logger.info("Upload Results:")
            for filename, success in upload_results.items():
                status = "Success" if success else "Failed"
                logger.info(f"  {filename}: {status}")

            successful_uploads = sum(upload_results.values())
            total_files = len(upload_results)
            if successful_uploads == total_files:
                logger.info("All data fetching, processing, and GCS upload completed successfully!")
                return True
            else:
                logger.error(f"Some files failed to upload. {successful_uploads}/{total_files} files uploaded successfully.")
                return False

        except Exception as e:
            logger.error(f"Error during Eurostat data fetching: {e}")
            return False
