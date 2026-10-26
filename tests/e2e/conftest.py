"""Shared fixtures for the prod-path tests against the GCS and Pub/Sub emulators (see docker-compose.emulators.yml)."""

import copy
import re
from pathlib import Path

import pytest
from google.api_core import exceptions as gapi_exceptions
from google.auth.credentials import AnonymousCredentials
from google.cloud import pubsub_v1, storage

from adoption_forecaster.config.config_loader import load_config
from adoption_forecaster.synthetic.writers import write_age_gender_inputs, write_education_inputs
from tests.e2e.emulators import DATA_SOURCE


@pytest.fixture
def emulator_config():
    config = copy.deepcopy(load_config())
    config["gcs"]["data_source"] = DATA_SOURCE
    config["pubsub"]["enable_notifications"] = True
    config["defaults"]["bucket_name"] = "adoption-forecaster-data"
    config["defaults"]["bucket_name_external"] = "universe-bucket-staging"
    return config


def _gcs_style(name):
    """Local writers use `_` in the time part; GCS inputs carry colons."""
    return re.sub(r"T(\d{2})_(\d{2})_(\d{2})", r"T\1:\2:\3", name)


@pytest.fixture
def emulator_env(emulator_config, world, tmp_path):
    cfg = emulator_config
    client = storage.Client(project=cfg["defaults"]["project_id"], credentials=AnonymousCredentials())
    buckets = {}
    for name in (cfg["defaults"]["bucket_name"], cfg["defaults"]["bucket_name_external"]):
        try:
            client.get_bucket(name).delete(force=True)
        except gapi_exceptions.NotFound:
            pass
        buckets[name] = client.create_bucket(name)

    project, topic_id = cfg["pubsub"]["sub_project_id"], cfg["pubsub"]["topic_name"]
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    topic_path = publisher.topic_path(project, topic_id)
    sub_path = subscriber.subscription_path(project, f"{topic_id}-test-sub")
    for fn, path in ((subscriber.delete_subscription, sub_path), (publisher.delete_topic, topic_path)):
        try:
            fn(request={"subscription" if "subscriptions" in path else "topic": path})
        except gapi_exceptions.NotFound:
            pass
    publisher.create_topic(request={"name": topic_path})
    subscriber.create_subscription(request={"name": sub_path, "topic": topic_path})

    for namespace, writer in (("Education", write_education_inputs), ("Age_Gender", write_age_gender_inputs)):
        for path in writer(world, tmp_path / namespace):
            buckets[cfg["defaults"]["bucket_name"]].blob(f"{DATA_SOURCE}/{namespace}/input/{_gcs_style(Path(path).name)}").upload_from_filename(str(path))

    yield {"config": cfg, "client": client, "subscriber": subscriber, "sub_path": sub_path}
    subscriber.close()
