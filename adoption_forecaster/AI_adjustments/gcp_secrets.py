"""
Fetch secrets from GCP Secret Manager.
"""

from google.cloud import secretmanager
from adoption_forecaster.config.logging_config import get_logger

logger = get_logger(__name__)


def get_openai_api_key(project_id: str, secret_name: str) -> str:
    """
    Fetch OPENAI_API_KEY  from GCP Secret Manager.
    """

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

    try:
        logger.info(f"Accessing secret: {name}")
        response = client.access_secret_version(request={"name": name})
        key = response.payload.data.decode("UTF-8").strip()
        logger.info("Successfully retrieved OpenAI API key from Secret Manager")
        return key
    except Exception as e:
        logger.error(f" Failed to access OpenAI API key from Secret Manager: {e}")
        raise RuntimeError(
            f"Could not access OpenAI API key from Secret Manager (secret: {name}). "
            "Check GCP credentials and that the secret exists and is accessible."
        ) from e
