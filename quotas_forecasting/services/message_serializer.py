"""
Message Serializer for protobuf communication with Andromeda
"""

import json
from datetime import datetime, timezone
from typing import Optional
from .modelling_files_pb2 import ModellingOutputFile, FileType
from google.cloud import pubsub_v1
from google.oauth2 import service_account
from google.cloud import secretmanager

from quotas_forecasting.config.logging_config import get_logger

logger = get_logger(__name__)


class MessageSerializer:
    """Handles serialization and publishing of protobuf messages to Andromeda"""

    def __init__(self, sender_name: str = "quotas_forecasting", config=None):
        """
        Initialize the message serializer

        Args:
            sender_name: Name of this service (quotas_forecasting)
            topic_name: Pub/Sub topic_name to publish to  (Andromeda created)
            project_id: GCP project_id of subscriber , here it is Andromeda
        """

        self.sender_name = sender_name
        self.topic_name = config["pubsub"]["topic_name"]
        self.sub_project_id = config["pubsub"]["sub_project_id"]
        self.project_id = config["defaults"]["project_id"]
        self.service_acc = config["pubsub"]["service_account"]
        logger.info(f"FileWriter init - topic_name: {self.topic_name}")
        logger.info(f"FileWriter init - service_account: {self.service_acc}")

        if self.topic_name and self.sub_project_id and pubsub_v1:
            try:
                if self.service_acc:
                    credentials = self._get_credentials_from_secret_manager()
                    self.publisher = pubsub_v1.PublisherClient(credentials=credentials)
                    logger.info(f"Using service account from Secret Manager: {credentials.service_account_email}")
                else:
                    raise ValueError("Service account email is required for Pub/Sub publishing")

                self.topic_path = self.publisher.topic_path(self.sub_project_id, self.topic_name)
                logger.info(f"Initialized Pub/Sub publisher for topic: {self.topic_name} in project: {self.sub_project_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Pub/Sub publisher: {e}")
                self.publisher = None
                self.topic_path = None
        else:
            self.publisher = None
            self.topic_path = None

    def _get_credentials_from_secret_manager(self):
        """Get service account credentials from Secret Manager"""
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_name = f"projects/{self.project_id}/secrets/quotas-forecasting-sa-key/versions/latest"
            logger.error("SECRET_NAME_USED=%s", secret_name)

            logger.info(f"Accessing secret: {secret_name}")
            response = client.access_secret_version(request={"name": secret_name})
            key_data = json.loads(response.payload.data.decode("UTF-8"))

            credentials = service_account.Credentials.from_service_account_info(key_data)
            logger.info("Successfully retrieved credentials from Secret Manager")
            return credentials

        except Exception as e:
            logger.error(f"Failed to get credentials from Secret Manager: {e}")
            raise

    def create_message(
        self,
        file_type: str,
        gcs_path: str,
        year: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Create a protobuf message for modelling output file.

        Args:
            filetype: Type of the file("AGE_AND_GENDER", "EDUCATION_16_TO_64", "EDUCATION_16_TO_74")
            gcs_path: GCS path to the file
            year: Projection year
            timestamp: Optional timestamp (UTC?)

        Returns:
                Serialized protobuf message as bytes or None if protobuf not available

        """

        if ModellingOutputFile is None:
            logger.warning("Protobuf not available, skipping message creation...")
            return None

        try:

            message = ModellingOutputFile()

            if timestamp is None:
                timestamp = datetime.now(tz=timezone.utc)
            message.time.FromDatetime(timestamp)

            message.sender = self.sender_name
            message.gcs_path = gcs_path
            message.projection_year = year

            try:
                message.file_type = FileType.Value(file_type)
            except ValueError as e:
                logger.error(f" Invalid file type {file_type}: {e}")

            encoded_message = message.SerializeToString()
            logger.info(f"Created protobuf message for {file_type} file")

            return encoded_message

        except Exception as e:
            logger.error(f"Error creating protobuf message: {e}")

            return None

    def send_modelling_output_notification(self, file_type: str, gcs_path: str, year: int) -> bool:
        """
        Send a notification about a modelling output file to Andromeda

        Args:
            file_type: Type of file
            gcs_path: GCS path to the file
            year: Project year

        Returns:
                True is message sent successfully, False otherwise.
        """

        encoded_message = self.create_message(
            file_type=file_type,
            gcs_path=gcs_path,
            year=year,
        )

        if encoded_message is None:
            return False

        logger.info(f"PROTOBUFF_MESSAGE: {encoded_message.hex()}")
        logger.info(f"FILE_TYPE: {file_type}")
        logger.info(f"GCS_PATH: {gcs_path}")
        logger.info(f"YEAR: {year}")
        logger.info(f"SENDER: {self.sender_name}")

        if self.publisher and self.topic_path:
            try:
                future = self.publisher.publish(self.topic_path, encoded_message)
                message_id = future.result()
                logger.info(f"Published message to Pub/Sub topic {self.topic_name}: {message_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to publish to Pub/Sub: {e}")
                return False

        else:
            logger.warning("Pub/Sub not configured - message created but not sent")
            return False
