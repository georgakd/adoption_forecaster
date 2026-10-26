"""Credential selection shared by the GCS and Pub/Sub clients."""

import json
import os

from google.auth.credentials import AnonymousCredentials
from google.cloud import secretmanager
from google.oauth2 import service_account

from adoption_forecaster.config.logging_config import get_logger

logger = get_logger(__name__)

EMULATOR_ENV_VARS = ("STORAGE_EMULATOR_HOST", "PUBSUB_EMULATOR_HOST")


def emulator_enabled():
    return any(os.environ.get(var) for var in EMULATOR_ENV_VARS)


def get_credentials(config):
    """Anonymous credentials when a GCS or Pub/Sub emulator is configured, otherwise the Secret Manager service account."""
    if emulator_enabled():
        logger.info("Emulator host set, using anonymous credentials")
        return AnonymousCredentials()
    return _credentials_from_secret_manager(config["defaults"]["project_id"])


def _credentials_from_secret_manager(project_id):
    """Get service account credentials from Secret Manager"""
    try:
        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{project_id}/secrets/quotas-forecasting-sa-key/versions/latest"
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
